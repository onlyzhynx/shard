import json
from pathlib import Path
import tempfile
import unittest

from research.context_sweep import (
    REQUIRED_FIELDS,
    compare_output_ids,
    output_ids_sha256,
    read_jsonl,
)


EXPECTED_REQUIRED_FIELDS = {
    "timestamp",
    "git_commit",
    "model_id",
    "mode",
    "ctx_len",
    "max_ctx",
    "K",
    "fast_enabled",
    "fv_window_enabled",
    "tok_s",
    "verify_ms",
    "draft_ms",
    "tokens_per_traversal",
    "mean_accept",
    "n_tokens",
    "rounds",
    "gpu_memory_allocated_mb",
    "gpu_memory_reserved_mb",
    "output_ids",
    "output_ids_sha256",
    "output_ids_match_baseline",
    "first_differing_token_index",
    "baseline_output_ids_sha256",
    "test_output_ids_sha256",
}


class ContextSweepUtilsTest(unittest.TestCase):
    def test_output_id_hash_is_stable(self):
        self.assertEqual(output_ids_sha256([1, 20, 300]), output_ids_sha256((1, 20, 300)))
        self.assertNotEqual(output_ids_sha256([1, 20, 300]), output_ids_sha256([1, 20, 301]))

    def test_exact_match_and_first_difference(self):
        self.assertEqual(compare_output_ids([1, 2, 3], [1, 2, 3]), (True, None))
        self.assertEqual(compare_output_ids([1, 2, 3], [1, 9, 3]), (False, 1))
        self.assertEqual(compare_output_ids([1, 2], [1, 2, 3]), (False, 2))

    def test_jsonl_row_has_required_fields(self):
        self.assertTrue(EXPECTED_REQUIRED_FIELDS.issubset(REQUIRED_FIELDS))
        row = {field: None for field in REQUIRED_FIELDS}
        row.update(
            {
                "timestamp": "2026-06-22T00:00:00+00:00",
                "model_id": "example/model",
                "mode": "baseline",
                "ctx_len": 2048,
                "max_ctx": 4096,
                "K": 4,
                "fast_enabled": True,
                "fv_window_enabled": False,
                "output_ids": [1, 2, 3],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            parsed = read_jsonl(path)
            self.assertEqual(parsed, [row])
            self.assertTrue(EXPECTED_REQUIRED_FIELDS.issubset(parsed[0]))


if __name__ == "__main__":
    unittest.main()
