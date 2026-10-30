"""Greedy decoding with a CUDA graph for one decode step.

Replaces transformers' generate() for single-image decoding. The decode step is recorded
once into a CUDA graph and replayed per token, which removes most of the per-token CPU
launch overhead (measured ~35 ms -> ~8 ms per token on the RTX 5060 Ti).

Recording happens inside decode(), so the cache size can follow each image's prompt length.
"""

from __future__ import annotations

import torch
from transformers import StaticCache


class GraphDecoder:
    def __init__(self, model, stop_ids: set[int], max_new_tokens: int = 1024, check_every: int = 8):
        self.model = model
        self.stop_ids = stop_ids
        self.max_new_tokens = max_new_tokens
        self.check_every = check_every

    def _trim(self, ids: list[int]) -> list[int]:
        return ids[: next((i for i, t in enumerate(ids) if t in self.stop_ids), len(ids))]

    @torch.inference_mode()
    def decode(self, inputs) -> list[int]:
        """Return generated token ids (without the stop token) for one image+prompt."""
        n0 = inputs.input_ids.shape[1]
        cache = StaticCache(config=self.model.config, max_batch_size=1,
                            max_cache_len=n0 + self.max_new_tokens + self.check_every,
                            device="cuda", dtype=self.model.dtype)
        static_ids = torch.zeros(1, 1, dtype=torch.long, device="cuda")
        static_pos = torch.full((1,), n0, dtype=torch.long, device="cuda")

        out = self.model(**inputs, past_key_values=cache, use_cache=True)
        static_ids.copy_(out.logits[:, -1:].argmax(-1))

        # warm up on a side stream, then record the decode step
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self.model(input_ids=static_ids, past_key_values=cache,
                           cache_position=static_pos, use_cache=True)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_logits = self.model(input_ids=static_ids, past_key_values=cache,
                                       cache_position=static_pos, use_cache=True).logits

        # the warm-up passes wrote into the cache, so restart from a clean prefill
        cache.reset()
        out = self.model(**inputs, past_key_values=cache, use_cache=True)
        static_ids.copy_(out.logits[:, -1:].argmax(-1))
        static_pos.fill_(n0)

        toks: list[torch.Tensor] = []
        for step in range(self.max_new_tokens):
            toks.append(static_ids.clone())
            if (step + 1) % self.check_every == 0:
                recent = torch.cat(toks[-self.check_every:], dim=1)[0].tolist()
                if any(t in self.stop_ids for t in recent):
                    break
            graph.replay()
            static_ids.copy_(static_logits[:, -1:].argmax(-1))
            static_pos.add_(1)
        return self._trim(torch.cat(toks, dim=1)[0].tolist())


class BatchGraphDecoder(GraphDecoder):
    """CUDA-graph greedy decoding for a batch of images.

    Inputs must be left-padded (processor tokenizer padding_side="left"). Each row keeps its own
    attention mask and position counter, so padded positions are ignored and each image's
    positions start at its first real token. Rows that hit a stop token keep decoding but are
    trimmed afterwards.
    """

    @torch.inference_mode()
    def decode_batch(self, inputs) -> list[list[int]]:
        attn_prompt = inputs.attention_mask  # (B, n0), 0 on left padding
        bsz, n0 = attn_prompt.shape
        max_len = n0 + self.max_new_tokens + self.check_every
        cache = StaticCache(config=self.model.config, max_batch_size=bsz,
                            max_cache_len=max_len, device="cuda", dtype=self.model.dtype)
        pad_len = (1 - attn_prompt).sum(dim=1)  # (B,)
        prompt_pos = (attn_prompt.cumsum(dim=1) - 1).clamp(min=0)

        mask = torch.zeros(bsz, max_len, dtype=torch.long, device="cuda")
        mask[:, :n0] = attn_prompt
        static_ids = torch.zeros(bsz, 1, dtype=torch.long, device="cuda")
        static_pos_ids = torch.zeros(bsz, 1, dtype=torch.long, device="cuda")
        static_cache_pos = torch.full((1,), n0, dtype=torch.long, device="cuda")

        out = self.model(**inputs, position_ids=prompt_pos, past_key_values=cache, use_cache=True)
        static_ids.copy_(out.logits[:, -1:].argmax(-1))
        static_pos_ids.copy_((n0 - pad_len).unsqueeze(1))

        def step_fn():
            return self.model(input_ids=static_ids, position_ids=static_pos_ids,
                              attention_mask=mask, past_key_values=cache,
                              cache_position=static_cache_pos, use_cache=True).logits

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                step_fn()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_logits = step_fn()

        cache.reset()
        out = self.model(**inputs, position_ids=prompt_pos, past_key_values=cache, use_cache=True)
        static_ids.copy_(out.logits[:, -1:].argmax(-1))
        static_pos_ids.copy_((n0 - pad_len).unsqueeze(1))
        static_cache_pos.fill_(n0)

        toks: list[torch.Tensor] = []
        done = torch.zeros(bsz, dtype=torch.bool, device="cuda")
        stop = torch.tensor(sorted(self.stop_ids), device="cuda")
        for step in range(self.max_new_tokens):
            toks.append(static_ids.clone())
            done |= torch.isin(static_ids[:, 0], stop)
            if (step + 1) % self.check_every == 0 and bool(done.all()):
                break
            mask[:, n0 + step] = 1
            graph.replay()
            static_ids.copy_(static_logits[:, -1:].argmax(-1))
            static_pos_ids.add_(1)
            static_cache_pos.add_(1)

        all_ids = torch.cat(toks, dim=1).tolist()
        return [self._trim(row) for row in all_ids]
