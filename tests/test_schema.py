import pytest

from img2text.data.schema import Record, read_jsonl, write_jsonl


def test_roundtrip(tmp_path):
    recs = [Record("a/1", "img/1.png", "chart", "x,y\n1,2", "chartcsv", "synth")]
    p = tmp_path / "out.jsonl"
    assert write_jsonl(recs, p) == 1
    assert list(read_jsonl(p)) == recs


def test_rejects_bad_task():
    with pytest.raises(ValueError):
        Record("a", "b", "photo", "t", "text", "s")
