"""Tests for --mode judge extension to ttft_bench."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ttft_bench import (  # noqa: E402
    JudgeVerdict,
    check_lm_studio_ready,
    is_model_loaded,
    is_no_answer,
    judge_pair_local,
    load_completed_judge_ids,
    load_reference_answers,
    load_responses_for_judge,
    run_judge_pair_set,
    save_judge_results,
    strip_think,
)

COMPARE_DIR = Path(__file__).parent.parent / "quality-results" / "compare"
REF_DIR     = Path(__file__).parent.parent / "reference-answers"


class TestStripThink:
    def test_removes_closed_think_block(self):
        text = "<think>\nsome reasoning\n</think>\n\nActual answer"
        assert strip_think(text) == "Actual answer"

    def test_removes_unclosed_think_block(self):
        text = "<think>\nreasoning that never ends"
        assert strip_think(text) == ""

    def test_passthrough_when_no_think(self):
        text = "Just a normal answer"
        assert strip_think(text) == "Just a normal answer"

    def test_removes_multiple_closed_blocks(self):
        text = "<think>a</think>\n<think>b</think>\nAnswer"
        assert strip_think(text) == "Answer"

    def test_strips_whitespace(self):
        text = "<think>x</think>   \n\n  Answer  "
        assert strip_think(text) == "Answer"


class TestIsNoAnswer:
    def test_unclosed_think_is_no_answer(self):
        # Unclosed think — no answer after it.
        assert is_no_answer("<think>\nall thinking, never answered") is True

    def test_closed_think_with_no_post_content(self):
        # Closed think block followed by nothing — treated as no answer.
        assert is_no_answer("<think>\nall thinking\n</think>") is True

    def test_false_when_has_content(self):
        assert is_no_answer("<think>x</think>\nReal answer") is False

    def test_false_when_no_think_at_all(self):
        assert is_no_answer("Just an answer") is False

    def test_false_when_empty(self):
        assert is_no_answer("") is False


class TestLoadResponsesForJudge:
    def test_loads_existing_compare_file(self):
        path = COMPARE_DIR / "responses_q4_resident.json"
        if not path.exists():
            pytest.skip("response file not present")
        responses = load_responses_for_judge(path)
        assert len(responses) == 80
        for r in responses.values():
            assert "content" in r
            assert "prompt_id" in r

    def test_raises_if_missing(self):
        with pytest.raises(FileNotFoundError):
            load_responses_for_judge(Path("/nonexistent/responses.json"))


class TestLoadReferenceAnswers:
    def test_loads_haiku_references(self):
        path = REF_DIR / "mtbench80_haiku.jsonl"
        if not path.exists():
            pytest.skip("reference file not present")
        refs = load_reference_answers(path)
        assert len(refs) == 80
        for r in refs.values():
            assert "content" in r
            assert "category" in r

    def test_raises_if_missing(self):
        with pytest.raises(FileNotFoundError):
            load_reference_answers(Path("/nonexistent/refs.jsonl"))


class TestJudgeVerdict:
    def test_has_required_fields(self):
        v = JudgeVerdict(
            prompt_id="mt_81",
            category="writing",
            verdict="same",
            reasoning="Both answers are equivalent",
        )
        assert v.prompt_id == "mt_81"
        assert v.verdict == "same"

    def test_to_dict(self):
        v = JudgeVerdict("mt_81", "writing", "similar", "minor diff")
        d = v.to_dict()
        assert d["verdict"] == "similar"
        assert d["prompt_id"] == "mt_81"

    def test_verdict_must_be_valid(self):
        with pytest.raises(ValueError):
            JudgeVerdict("mt_81", "writing", "invalid_verdict", "reason")


class TestJudgePairLocal:
    def _mock_client_with_verdict(self, verdict: str, reasoning: str = "r"):
        mock_client = MagicMock()
        tc = MagicMock()
        tc.function.arguments = json.dumps({"verdict": verdict, "reasoning": reasoning})
        choice = MagicMock()
        choice.message.tool_calls = [tc]
        mock_client.chat.completions.create.return_value.choices = [choice]
        return mock_client

    def test_returns_verdict_dict(self):
        client = self._mock_client_with_verdict("same", "equivalent")
        result = judge_pair_local(
            client=client,
            prompt="what is 2+2",
            ref_content="4",
            local_content="The answer is 4",
            model="gemma-4-26b",
        )
        assert result["verdict"] == "same"

    def test_returns_parse_error_on_exception(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("connection refused")
        result = judge_pair_local(
            client=client,
            prompt="p", ref_content="r", local_content="l",
            model="gemma-4-26b", retries=0,
        )
        assert result["verdict"] == "parse_error"


class TestRunJudgePairSet:
    def test_skips_no_answer_prompts(self):
        local_responses = {
            "mt_81": {"prompt_id": "mt_81", "category": "writing",
                      "content": "<think>never finished", "prompt": "p"},
            "mt_82": {"prompt_id": "mt_82", "category": "math",
                      "content": "The answer is 4", "prompt": "p2"},
        }
        ref_responses = {
            "mt_81": {"id": "mt_81", "category": "writing",
                      "content": "real answer", "prompt": "p"},
            "mt_82": {"id": "mt_82", "category": "math",
                      "content": "4", "prompt": "p2"},
        }
        mock_judge = MagicMock(return_value={"verdict": "same", "reasoning": "ok"})
        result = run_judge_pair_set(
            local_name="q4r", ref_name="haiku",
            local_responses=local_responses,
            ref_responses=ref_responses,
            judge_fn=mock_judge,
        )
        verdict_map = {v["prompt_id"]: v["verdict"] for v in result["verdicts"]}
        assert verdict_map["mt_81"] == "no_answer"
        assert verdict_map["mt_82"] == "same"
        # Judge only called once — mt_81 short-circuited as no_answer.
        assert mock_judge.call_count == 1

    def test_counts_are_correct(self):
        local_responses = {
            "mt_81": {"prompt_id": "mt_81", "category": "writing",
                      "content": "answer", "prompt": "p"},
        }
        ref_responses = {
            "mt_81": {"id": "mt_81", "category": "writing",
                      "content": "ref", "prompt": "p"},
        }
        mock_judge = MagicMock(return_value={"verdict": "similar", "reasoning": "close"})
        result = run_judge_pair_set(
            "q4r", "haiku", local_responses, ref_responses, mock_judge
        )
        assert result["counts"]["similar"] == 1
        assert result["counts"]["same"] == 0


class TestSaveJudgeResults:
    def test_writes_correct_schema(self, tmp_path):
        verdicts = [
            JudgeVerdict("mt_81", "writing", "same", "equivalent"),
            JudgeVerdict("mt_82", "math", "different", "wrong answer"),
        ]
        counts = {"same": 1, "similar": 0, "different": 1,
                  "no_answer": 0, "parse_error": 0}
        out = tmp_path / "judge_q4r_vs_haiku.json"
        save_judge_results(
            out_path=out,
            local_name="q4r", ref_name="haiku",
            judge_model="gemma-4-26b",
            counts=counts, by_category={}, verdicts=verdicts,
            n_compared=2,
        )
        data = json.loads(out.read_text())
        assert data["local"] == "q4r"
        assert data["reference"] == "haiku"
        assert len(data["verdicts"]) == 2

    def test_writes_atomically(self, tmp_path):
        out = tmp_path / "judge_test.json"
        save_judge_results(out, "q4r", "haiku", "gemma",
                           {}, {}, [], 0)
        assert out.exists()
        assert not (tmp_path / "judge_test.json.tmp").exists()


class TestLoadCompletedJudgeIds:
    def test_returns_empty_set_if_no_file(self, tmp_path):
        ids = load_completed_judge_ids(tmp_path / "nonexistent.json")
        assert ids == set()

    def test_returns_completed_ids(self, tmp_path):
        data = {"verdicts": [
            {"prompt_id": "mt_81", "verdict": "same"},
            {"prompt_id": "mt_82", "verdict": "different"},
        ]}
        f = tmp_path / "judge_q4r_vs_haiku.json"
        f.write_text(json.dumps(data))
        ids = load_completed_judge_ids(f)
        assert ids == {"mt_81", "mt_82"}


class TestJudgeModeMemoryAwareness:
    def test_rss_sampled_per_prompt(self):
        """judge_pair_local must accept (and tolerate) a server_pid kwarg."""
        mock_client = MagicMock()
        tc = MagicMock()
        tc.function.arguments = json.dumps({"verdict": "same", "reasoning": "ok"})
        choice = MagicMock()
        choice.message.tool_calls = [tc]
        mock_client.chat.completions.create.return_value.choices = [choice]
        result = judge_pair_local(
            client=mock_client,
            prompt="p", ref_content="r", local_content="l",
            model="gemma", server_pid=None,
        )
        assert "verdict" in result


class TestLMStudioReadiness:
    def test_check_lm_studio_reachable(self):
        # Must not raise even when LM Studio is down.
        ready, models = check_lm_studio_ready("http://localhost:1234/v1", "lm-studio")
        assert isinstance(ready, bool)
        assert isinstance(models, list)

    def test_check_returns_false_if_unreachable(self):
        ready, models = check_lm_studio_ready("http://localhost:9999/v1", "lm-studio")
        assert ready is False
        assert models == []

    def test_target_model_in_loaded_models(self):
        loaded = ["gemma-4-26b-a4b-it-ud-2", "text-embedding-nomic"]
        assert is_model_loaded("gemma-4-26b-a4b-it-ud-2", loaded) is True
        assert is_model_loaded("qwen3-30b", loaded) is False


class TestJudgeModeCLI:
    def test_mode_judge_flag_exists(self):
        from ttft_bench import build_arg_parser
        parser = build_arg_parser()
        args = parser.parse_args(["--mode", "judge"])
        assert args.mode == "judge"

    def test_judge_endpoint_flag(self):
        from ttft_bench import build_arg_parser
        parser = build_arg_parser()
        args = parser.parse_args([
            "--mode", "judge",
            "--judge-endpoint", "http://localhost:1234/v1",
            "--judge-model", "gemma-4-26b",
        ])
        assert args.judge_endpoint == "http://localhost:1234/v1"
        assert args.judge_model == "gemma-4-26b"

    def test_judge_api_key_defaults_to_lm_studio(self):
        from ttft_bench import build_arg_parser
        parser = build_arg_parser()
        args = parser.parse_args(["--mode", "judge"])
        assert args.judge_api_key == "lm-studio"

    def test_skip_sampling_flag(self):
        from ttft_bench import build_arg_parser
        parser = build_arg_parser()
        args = parser.parse_args(["--mode", "judge", "--skip-sampling"])
        assert args.skip_sampling is True
