"""Tests for --mode quality extension to ttft_bench."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Make the parent directory importable so `from ttft_bench import ...` works
# regardless of cwd when pytest is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ttft_bench import (  # noqa: E402
    QualityResult,
    collect_quality_response,
    load_completed_prompt_ids,
    load_quality_prompts,
    save_quality_responses,
)

PROMPTS_FILE = Path(__file__).parent.parent / "mtbench80.jsonl"


class TestLoadQualityPrompts:
    def test_loads_80_prompts(self):
        prompts = load_quality_prompts(PROMPTS_FILE)
        assert len(prompts) == 80

    def test_each_prompt_has_id_and_text(self):
        prompts = load_quality_prompts(PROMPTS_FILE)
        for p in prompts:
            assert "id" in p
            assert "prompt" in p
            assert "category" in p

    def test_raises_if_file_missing(self):
        with pytest.raises(FileNotFoundError):
            load_quality_prompts(Path("/nonexistent/mtbench80.jsonl"))


class TestQualityResult:
    def test_has_required_fields(self):
        r = QualityResult(
            prompt_id="mt_81",
            category="writing",
            prompt="write something",
            content="here is something",
            n_tokens=10,
            decode_tps=25.0,
            ttft_s=0.5,
            wall_s=1.2,
        )
        assert r.prompt_id == "mt_81"
        assert r.content == "here is something"

    def test_to_dict_matches_judge_schema(self):
        r = QualityResult(
            prompt_id="mt_81", category="writing", prompt="write something",
            content="here is something", n_tokens=10, decode_tps=25.0,
            ttft_s=0.5, wall_s=1.2,
        )
        d = r.to_dict()
        # Must match schema expected by the judge (same fields
        # quality_compare.py used to write).
        assert "prompt_id" in d
        assert "category" in d
        assert "content" in d
        assert "decode_tps" in d
        assert "ttft_s" in d
        assert "n_tokens" in d


class TestSaveQualityResponses:
    def test_saves_json_with_correct_schema(self, tmp_path):
        results = [
            QualityResult("mt_81", "writing", "prompt", "content", 10, 25.0, 0.5, 1.2),
            QualityResult("mt_82", "coding", "prompt2", "content2", 20, 22.0, 0.6, 1.5),
        ]
        out = tmp_path / "responses_test.json"
        save_quality_responses(
            results=results,
            name="test_config",
            label="Test Config",
            load_s=5.0,
            flags=["--flag"],
            out_path=out,
        )
        data = json.loads(out.read_text())
        assert data["name"] == "test_config"
        assert data["label"] == "Test Config"
        assert len(data["responses"]) == 2
        assert data["responses"][0]["prompt_id"] == "mt_81"

    def test_output_is_resumable(self, tmp_path):
        """Partial file can be loaded and extended."""
        out = tmp_path / "responses_test.json"
        results = [QualityResult("mt_81", "writing", "p", "c", 10, 25.0, 0.5, 1.2)]
        save_quality_responses(results, "x", "X", 5.0, [], out)
        data = json.loads(out.read_text())
        completed = {r["prompt_id"] for r in data["responses"]}
        assert "mt_81" in completed
        assert "mt_82" not in completed

    def test_atomic_write_leaves_no_tmp(self, tmp_path):
        out = tmp_path / "responses_atom.json"
        save_quality_responses([], "x", "X", 5.0, [], out)
        assert out.exists()
        assert not (tmp_path / "responses_atom.json.tmp").exists()


class TestCollectQualityResponse:
    def _ok_response(self, content="the answer", n_tokens=15):
        resp = MagicMock()
        resp.json.return_value = {
            "choices": [{"message": {"content": content}}],
            "usage": {"completion_tokens": n_tokens},
        }
        return resp

    def test_returns_quality_result(self):
        session = MagicMock()
        session.post.return_value = self._ok_response()
        result = collect_quality_response(
            session=session,
            port=11500,
            prompt_id="mt_81",
            category="writing",
            prompt="write something",
            n_predict=3000,
            seed=42,
        )
        assert isinstance(result, QualityResult)
        assert result.content == "the answer"
        assert result.prompt_id == "mt_81"

    def test_records_rss(self):
        """rss_bytes attr must exist even if None when no pid given."""
        session = MagicMock()
        session.post.return_value = self._ok_response(content="answer", n_tokens=10)
        result = collect_quality_response(
            session=session, port=11500,
            prompt_id="mt_81", category="writing",
            prompt="write", n_predict=3000, seed=42,
            server_pid=None,
        )
        assert hasattr(result, "rss_bytes")


class TestLoadCompletedPromptIds:
    def test_returns_empty_if_no_file(self, tmp_path):
        ids = load_completed_prompt_ids(tmp_path / "absent.json")
        assert ids == set()

    def test_returns_completed_ids(self, tmp_path):
        data = {"responses": [
            {"prompt_id": "mt_81", "content": "x"},
            {"prompt_id": "mt_82", "content": "y"},
        ]}
        f = tmp_path / "responses_q4r.json"
        f.write_text(json.dumps(data))
        assert load_completed_prompt_ids(f) == {"mt_81", "mt_82"}


class TestPerPromptLogging:
    def test_log_file_created_alongside_responses(self, tmp_path):
        from ttft_bench import get_log_path
        responses_path = tmp_path / "responses_bf16_streaming.json"
        log_path = get_log_path(responses_path)
        assert log_path == tmp_path / "responses_bf16_streaming.log.jsonl"

    def test_log_entry_has_required_fields(self):
        from ttft_bench import build_log_entry
        entry = build_log_entry(
            prompt_id="mt_81",
            wall_s=4.2,
            rss_bytes=24_000_000_000,
            memory_pressure="normal",
            decode_tps=18.3,
            n_tokens=120,
            timestamp="2026-04-20T21:00:00",
        )
        assert entry["prompt_id"] == "mt_81"
        assert entry["wall_s"] == 4.2
        assert entry["rss_bytes"] == 24_000_000_000
        assert entry["memory_pressure"] in ("normal", "warning", "critical", "unknown")
        assert entry["decode_tps"] == 18.3
        assert entry["n_tokens"] == 120
        assert "timestamp" in entry

    def test_log_entry_written_atomically(self, tmp_path):
        from ttft_bench import append_log_entry, build_log_entry, get_log_path
        responses_path = tmp_path / "responses_test.json"
        log_path = get_log_path(responses_path)
        entry = build_log_entry(
            "mt_81", 4.2, 24_000_000_000, "normal", 18.3, 120,
            "2026-04-20T21:00:00",
        )
        append_log_entry(log_path, entry)
        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["prompt_id"] == "mt_81"

    def test_multiple_entries_appended(self, tmp_path):
        from ttft_bench import append_log_entry, build_log_entry, get_log_path
        responses_path = tmp_path / "responses_test.json"
        log_path = get_log_path(responses_path)
        for i, pid in enumerate(["mt_81", "mt_82", "mt_83"]):
            entry = build_log_entry(
                pid, float(i), 24_000_000_000, "normal", 18.3, 10,
                "2026-04-20T21:00:00",
            )
            append_log_entry(log_path, entry)
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_get_memory_pressure_returns_valid_level(self):
        from ttft_bench import get_memory_pressure
        level = get_memory_pressure()
        assert level in ("normal", "warning", "critical", "unknown")

    def test_get_memory_pressure_does_not_raise(self):
        from ttft_bench import get_memory_pressure
        # Should never raise — returns "unknown" on any error
        level = get_memory_pressure()
        assert isinstance(level, str)


class TestQualityModeCLI:
    def test_mode_quality_cli_flag_exists(self):
        from ttft_bench import build_arg_parser
        parser = build_arg_parser()
        args = parser.parse_args(["--mode", "quality"])
        assert args.mode == "quality"

    def test_mode_ttft_is_default(self):
        from ttft_bench import build_arg_parser
        parser = build_arg_parser()
        args = parser.parse_args([])
        assert args.mode == "ttft"

    def test_n_predict_default_is_3000(self):
        from ttft_bench import build_arg_parser
        parser = build_arg_parser()
        args = parser.parse_args(["--mode", "quality"])
        assert args.n_predict == 3000

    def test_prompts_flag_default(self):
        from ttft_bench import build_arg_parser
        parser = build_arg_parser()
        args = parser.parse_args([])
        assert args.prompts == "mtbench80.jsonl"
