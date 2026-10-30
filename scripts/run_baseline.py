"""Zero-shot baseline runner: one model, one task, scored with the eval metrics.

Usage:
    uv run python scripts/run_baseline.py --model granite --task tables
    uv run python scripts/run_baseline.py --model smolvlm --task charts --limit 20

Outputs (data_cache/runs/<model>/):
  <task>.jsonl       one line per image, written and flushed as soon as it is scored
  <task>.log         progress log, one line per image, appended and closed each time
  <task>_summary.json  written at the end from all rows

Re-running resumes: images already in <task>.jsonl are skipped.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from PIL import Image

from img2text.eval.metrics import score_chart, score_mermaid, teds
from img2text.formats.otsl import otsl_to_html
from img2text.model.fast_decode import BatchGraphDecoder, GraphDecoder

ROOT = Path(__file__).resolve().parents[1] / "data_cache"
MODEL_DIRS = {
    "granite": ROOT / "models" / "granite-docling-258M",
    "smolvlm": ROOT / "models" / "SmolVLM-256M-Instruct",
}
PROMPTS = {
    "smolvlm": {
        "tables": "Convert this table to HTML.",
        "charts": "Extract the underlying data of this chart as CSV with a header row.",
        "diagrams": "Convert this diagram to Mermaid code.",
    },
}
GRANITE_PAGE_PROMPT = "Convert this page to docling."
GRANITE_TABLE_PROMPT = "Convert this table to OTSL."
GRANITE_STOP_IDS = {100257}  # end-of-text token; eos is added at runtime


class Runner:
    """Wraps a model so every task goes through one `predict(image, task) -> str` call."""

    def __init__(self, name: str):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.name = name
        path = MODEL_DIRS[name]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = AutoModelForImageTextToText.from_pretrained(path, torch_dtype=self.dtype)
        self.model.to(self.device).eval()
        if name == "granite":
            stop = {self.processor.tokenizer.eos_token_id} | GRANITE_STOP_IDS
            self.decoder = GraphDecoder(self.model, stop_ids=stop, max_new_tokens=1024)
            self.batch_decoder = BatchGraphDecoder(self.model, stop_ids=stop, max_new_tokens=1024)
            self.processor.tokenizer.padding_side = "left"

    @torch.inference_mode()
    def _generate(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[image], return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                           skip_special_tokens=False)[0]

    def _inputs(self, image: Image.Image, prompt: str):
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        return self.processor(text=text, images=[image], return_tensors="pt").to(self.device)

    def predict_table_batch(self, images: list[Image.Image]) -> list[str]:
        """Tables only: decode a batch of images together with the table prompt."""
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": GRANITE_TABLE_PROMPT}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=[text] * len(images), images=[[im] for im in images],
                                return_tensors="pt", padding=True).to(self.device)
        rows = self.batch_decoder.decode_batch(inputs)
        return [otsl_to_html(self.processor.tokenizer.decode(r, skip_special_tokens=False))
                for r in rows]

    def predict(self, image: Image.Image, task: str) -> str:
        if self.name == "granite":
            if task == "tables":
                # table prompt -> OTSL, decoded with the CUDA-graph decoder, converted to HTML
                ids = self.decoder.decode(self._inputs(image, GRANITE_TABLE_PROMPT))
                return otsl_to_html(self.processor.tokenizer.decode(ids, skip_special_tokens=False))
            ids = self.decoder.decode(self._inputs(image, GRANITE_PAGE_PROMPT))
            doctags = self.processor.tokenizer.decode(ids, skip_special_tokens=False)
            return _doctags_to_text(doctags.replace("<end_of_utterance>", "").strip(), image, task)
        return self._generate(image, PROMPTS[self.name][task], max_new_tokens=1024).strip()


def _doctags_to_text(doctags: str, image: Image.Image, task: str) -> str:
    """Convert Granite-Docling DocTags to HTML (tables) or Markdown (everything else)."""
    from docling_core.types.doc.document import DocTagsDocument, DoclingDocument

    try:
        doc_tags = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
        doc = DoclingDocument.load_from_doctags(doc_tags, document_name="page")
    except Exception:
        return doctags  # keep the raw output so the score reflects the failure
    if task == "tables" and doc.tables:
        return doc.tables[0].export_to_html(doc=doc)
    return doc.export_to_markdown()


def _score(task: str, pred: str, gt: str) -> dict:
    if task == "tables":
        return {"teds": teds(pred, gt), "teds_s": teds(pred, gt, structure_only=True)}
    if task == "charts":
        return score_chart(pred, gt)
    return score_mermaid(pred, gt)


def _load_done(results_path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["id"]] = row
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_DIRS), required=True)
    parser.add_argument("--task", choices=["tables", "charts", "diagrams"], required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch", type=int, default=1,
                        help="images per decode batch (granite tables only)")
    args = parser.parse_args()

    items = [json.loads(line) for line in
             (ROOT / "eval" / f"{args.task}.jsonl").read_text(encoding="utf-8").splitlines()]
    if args.limit:
        items = items[: args.limit]

    out_dir = ROOT / "runs" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"{args.task}.jsonl"
    log_path = out_dir / f"{args.task}.log"

    done = _load_done(results_path)
    todo = [it for it in items if it["id"] not in done]

    def log(msg: str) -> None:
        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(msg + "\n")
        print(msg, flush=True)

    log(f"start model={args.model} task={args.task} todo={len(todo)} resumed={len(done)} "
        f"batch={args.batch}")
    runner = Runner(args.model)
    use_batch = args.batch > 1 and args.model == "granite" and args.task == "tables"
    if use_batch:
        with results_path.open("a", encoding="utf-8") as rf:
            for start in range(0, len(todo), args.batch):
                chunk = todo[start:start + args.batch]
                images = [Image.open(ROOT / "eval" / it["image"]).convert("RGB") for it in chunk]
                t0 = time.perf_counter()
                preds = runner.predict_table_batch(images)
                per_ms = (time.perf_counter() - t0) * 1000 / len(chunk)
                for it, pred in zip(chunk, preds):
                    scores = _score(args.task, pred, it["target"])
                    row = {"id": it["id"], "pred": pred, "scores": scores,
                           "latency_ms": round(per_ms, 1)}
                    rf.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rf.flush()
                    done[it["id"]] = row
                    log(f"[{start + 1 + chunk.index(it)}/{len(todo)}] {it['id']} {scores} "
                        f"{per_ms:.0f}ms/img (batch of {len(chunk)})")
        todo = []
    with results_path.open("a", encoding="utf-8") as rf:
        for i, it in enumerate(todo, 1):
            image = Image.open(ROOT / "eval" / it["image"]).convert("RGB")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred = runner.predict(image, args.task)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
            scores = _score(args.task, pred, it["target"])
            row = {"id": it["id"], "pred": pred, "scores": scores, "latency_ms": round(ms, 1)}
            rf.write(json.dumps(row, ensure_ascii=False) + "\n")
            rf.flush()
            done[it["id"]] = row
            log(f"[{i}/{len(todo)}] {it['id']} {scores} {ms:.0f}ms")

    rows = [done[it["id"]] for it in items if it["id"] in done]
    if not rows:
        raise SystemExit("no rows to summarise")
    summary = {"n": len(rows)}
    for k in rows[0]["scores"]:
        vals = [r["scores"][k] for r in rows if isinstance(r["scores"][k], (int, float))]
        if vals:
            summary[k] = round(statistics.fmean(vals), 4)
    times = sorted(r["latency_ms"] for r in rows)
    summary["latency_p50_ms"] = round(statistics.median(times), 1)
    summary["latency_p95_ms"] = round(times[int(0.95 * (len(times) - 1))], 1)
    (out_dir / f"{args.task}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
