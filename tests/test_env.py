"""Test suite for MultiBlockEnv."""

from __future__ import annotations

import random
import time

import pytest

from env.base import Difficulty
from env.blocks.code_gen import CodeGenBlock
from env.blocks.reasoning import ReasoningBlock
from env.blocks.task_split import TaskSplittingBlock
from env.long_horizon import CurriculumScheduler, EnvConfig, LongHorizonEnv, RollingStateBuffer
from env.rewards.multi_reward import (
    CodeSandbox,
    MultiRewardCalculator,
    SplitQualityScorer,
    StateIntegrityChecker,
)


# ===========================================================================
# CodeSandbox tests
# ===========================================================================

class TestCodeSandbox:
    def setup_method(self):
        self.sb = CodeSandbox(timeout=3.0)

    def test_blocks_import(self):
        result = self.sb.execute("import os\nos.system('dir')", "", timeout=2.0)
        assert result.blocked
        assert "Import" in result.block_reason

    def test_blocks_exec(self):
        result = self.sb.execute("exec('x=1')", "", timeout=2.0)
        assert result.blocked

    def test_blocks_globals(self):
        result = self.sb.execute("x = globals()", "", timeout=2.0)
        assert result.blocked

    def test_blocks_memory_bomb(self):
        result = self.sb.execute("x = [0] * 10_000_000", "", timeout=2.0)
        assert result.blocked
        assert "Memory bomb" in result.block_reason

    def test_allows_safe_code(self):
        code = "def double(x):\n    return x * 2"
        harness = "_test_result = double(3)"
        result = self.sb.execute(code, harness, timeout=2.0)
        assert result.success
        assert result.result == 6

    def test_timeout_enforced(self):
        result = self.sb.execute("while True: pass", "", timeout=1.0)
        assert result.timed_out

    def test_syntax_error_blocked(self):
        result = self.sb.execute("def f(: pass", "", timeout=2.0)
        assert result.blocked
        assert "SyntaxError" in result.block_reason


# ===========================================================================
# StateIntegrityChecker tests
# ===========================================================================

class TestStateIntegrityChecker:
    def test_snapshot_stable(self):
        checker = StateIntegrityChecker()
        state = {"a": 1, "b": [2, 3]}
        h1 = checker.snapshot(state)
        h2 = checker.snapshot(state)
        assert h1 == h2

    def test_snapshot_changes_on_mutation(self):
        checker = StateIntegrityChecker()
        state = {"a": 1}
        h1 = checker.snapshot(state)
        state["a"] = 2
        h2 = checker.snapshot(state)
        assert h1 != h2

    def test_cache_abuse_detected(self):
        checker = StateIntegrityChecker()
        assert checker.detect_cache_abuse(["a", "a", "a"])
        assert not checker.detect_cache_abuse(["a", "a", "b"])
        assert not checker.detect_cache_abuse(["a", "a"])

    def test_reward_probe_detected(self):
        checker = StateIntegrityChecker()
        alternating = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
        assert checker.detect_reward_probe(alternating)
        stable = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        assert not checker.detect_reward_probe(stable)


# ===========================================================================
# SplitQualityScorer tests
# ===========================================================================

class TestSplitQualityScorer:
    def setup_method(self):
        self.scorer = SplitQualityScorer()
        self.ref = [
            "Open and read the CSV file",
            "Parse the header to get column names",
            "Compute average per column",
            "Print the column with the highest average",
        ]
        self.problem = "Read a CSV file and print the column with the highest average value."

    def test_good_split_scores_high(self):
        tasks = [
            "Open and read the input CSV file",
            "Parse header row to identify column names",
            "Sum values per column and count rows",
            "Divide totals by row count to get averages",
            "Compare averages and print the highest column name",
        ]
        c = self.scorer.score(tasks, self.ref, self.problem, 3, 6)
        assert c.total() > 0.5

    def test_empty_tasks_scores_zero(self):
        c = self.scorer.score([], self.ref, self.problem, 3, 6)
        assert c.total() == 0.0

    def test_verbatim_restate_penalised(self):
        tasks = ["Read a CSV file and print the column with the highest average value."]
        c = self.scorer.score(tasks, self.ref, self.problem, 3, 6)
        assert c.anti_hack_penalty < 0

    def test_too_few_tasks_caps_completeness(self):
        tasks = ["Open CSV file", "Print result"]
        c = self.scorer.score(tasks, self.ref, self.problem, 4, 6)
        assert c.completeness <= 0.5

    def test_over_max_tasks_ignored(self):
        # 10 tasks but max is 4 — excess should not inflate score
        tasks = [f"Do step number {i}" for i in range(10)]
        c = self.scorer.score(tasks, self.ref, self.problem, 2, 4)
        # Should not error; score computed on first 4 only
        assert c is not None

    def test_non_specific_tasks_penalised(self):
        tasks = ["Do it", "Run", "Go", "OK"]
        c = self.scorer.score(tasks, self.ref, self.problem, 2, 6)
        assert c.specificity < 0.5

    def test_redundant_tasks_penalised(self):
        tasks = [
            "Open and read the CSV file line by line",
            "Open and read the CSV file line by line",
            "Compute averages per column",
        ]
        c = self.scorer.score(tasks, self.ref, self.problem, 2, 6)
        assert c.non_redundancy < 1.0


# ===========================================================================
# TaskSplittingBlock tests
# ===========================================================================

class TestTaskSplittingBlock:
    def setup_method(self):
        self.block = TaskSplittingBlock()
        self.rng = random.Random(42)

    def test_reset_returns_prompt_and_metadata(self):
        prompt, meta = self.block.reset(Difficulty.EASY, self.rng)
        assert "Task 1:" in prompt or "Decompose" in prompt
        assert "reference_split" in meta
        assert meta["min_tasks"] >= 3

    def test_good_split_gets_positive_reward(self):
        _, meta = self.block.reset(Difficulty.EASY, self.rng)
        # Use reference split as the agent response
        tasks = meta["reference_split"]
        action = "\n".join(f"Task {i+1}: {t}" for i, t in enumerate(tasks))
        comp, done = self.block.step(action, meta)
        assert not done  # step 1, still one revision allowed
        total = sum(
            v * w for v, w in zip(
                [comp["completeness"], comp["atomicity"], comp["ordering_score"],
                 comp["specificity"], comp["non_redundancy"]],
                [0.35, 0.20, 0.20, 0.15, 0.10]
            )
        )
        assert total > 0.4

    def test_revision_step_closes_episode(self):
        _, meta = self.block.reset(Difficulty.EASY, self.rng)
        # Step 1
        self.block.step("Task 1: Do something useful here", meta)
        # Step 2 (revision)
        comp, done = self.block.step("Task 1: Do something useful here", meta)
        assert done
        assert "revision_bonus" in comp

    def test_parse_tasks_handles_formats(self):
        text = "1. Open the file\n2) Read lines\nTask 3: Process data"
        tasks = TaskSplittingBlock._parse_tasks(text)
        assert len(tasks) == 3


# ===========================================================================
# CodeGenBlock tests
# ===========================================================================

class TestCodeGenBlock:
    def setup_method(self):
        self.block = CodeGenBlock()
        self.rng = random.Random(0)

    def test_reset_returns_description(self):
        prompt, meta = self.block.reset(Difficulty.EASY, self.rng)
        assert meta["fn_name"] in prompt or "function" in prompt.lower()
        assert "test_cases" in meta

    def test_correct_solution_gets_full_reward(self):
        _, meta = self.block.reset(Difficulty.EASY, self.rng)
        solution = meta["solution"]
        action = f"```python\n{solution}\n```"
        comp, done = self.block.step(action, meta)
        assert comp["correctness"] == 1.0
        assert done

    def test_wrong_solution_gets_low_reward(self):
        _, meta = self.block.reset(Difficulty.EASY, self.rng)
        action = "```python\ndef sum_list(lst):\n    return 0\n```"
        comp, _ = self.block.step(action, meta)
        assert comp["correctness"] < 1.0

    def test_import_attempt_penalised(self):
        _, meta = self.block.reset(Difficulty.EASY, self.rng)
        action = "```python\nimport os\ndef sum_list(lst): return os.getpid()\n```"
        comp, _ = self.block.step(action, meta)
        assert comp["anti_hack_penalty"] < 0

    def test_cache_abuse_penalised(self):
        _, meta = self.block.reset(Difficulty.EASY, self.rng)
        same = "```python\ndef sum_list(lst): return 0\n```"
        self.block.step(same, meta)
        self.block.step(same, meta)
        comp, _ = self.block.step(same, meta)
        assert comp["anti_hack_penalty"] < 0

    def test_no_code_gets_zero_format(self):
        _, meta = self.block.reset(Difficulty.EASY, self.rng)
        comp, _ = self.block.step("I don't know the answer.", meta)
        assert comp["format_compliance"] == 0.0


# ===========================================================================
# ReasoningBlock tests
# ===========================================================================

class TestReasoningBlock:
    def setup_method(self):
        self.block = ReasoningBlock()
        self.rng = random.Random(1)

    def test_correct_answer_with_steps_rewarded(self):
        _, meta = self.block.reset(Difficulty.EASY, self.rng)
        answer = meta["answer"]
        action = f"Step 1: Setting up the problem.\nStep 2: Computing.\nFinal Answer: {answer}"
        comp, _ = self.block.step(action, meta)
        assert comp["execution_success"] == 1.0
        assert comp["correctness"] >= 0.5

    def test_no_final_answer_zero_success(self):
        _, meta = self.block.reset(Difficulty.EASY, self.rng)
        comp, _ = self.block.step("Step 1: Thinking...", meta)
        assert comp["execution_success"] == 0.0

    def test_verbatim_copy_penalised(self):
        _, meta = self.block.reset(Difficulty.EASY, self.rng)
        # Answer is a number that appears in the task description; no steps shown
        desc_nums = __import__("re").findall(r"\d+", meta["task_description"])
        if desc_nums:
            action = f"Final Answer: {desc_nums[0]}"
            comp, _ = self.block.step(action, meta)
            # anti_hack fires only if step_count < min_steps, which is true here
            assert comp["anti_hack_penalty"] <= 0


# ===========================================================================
# RollingStateBuffer tests
# ===========================================================================

class TestRollingStateBuffer:
    def test_pinned_entry_never_evicted(self):
        buf = RollingStateBuffer(max_tokens=10)  # very small
        pinned = {"step": 0, "role": "env", "content": "original task", "timestamp": 0.0}
        buf.push(pinned, pin=True)
        # Fill beyond budget
        for i in range(20):
            buf.push({"step": i, "role": "agent", "content": "x" * 50, "timestamp": float(i)})
        result = buf.to_list()
        assert result[0] == pinned

    def test_buffer_respects_token_budget(self):
        buf = RollingStateBuffer(max_tokens=50)
        for i in range(100):
            buf.push({"step": i, "content": "word " * 5})
        # Should not grow unbounded
        assert len(buf.to_list()) < 100


# ===========================================================================
# CurriculumScheduler tests
# ===========================================================================

class TestCurriculumScheduler:
    def test_promotes_after_success(self):
        sched = CurriculumScheduler(alpha=0.5, promote_threshold=0.65, min_episodes=3)
        assert sched.current_difficulty == Difficulty.EASY
        for _ in range(3):
            sched.record_episode(1.0)  # always positive
        assert sched.current_difficulty == Difficulty.MEDIUM

    def test_does_not_promote_on_failure(self):
        sched = CurriculumScheduler(alpha=0.5, promote_threshold=0.65, min_episodes=3)
        for _ in range(10):
            sched.record_episode(-1.0)
        assert sched.current_difficulty == Difficulty.EASY

    def test_hard_is_final(self):
        sched = CurriculumScheduler(alpha=1.0, promote_threshold=0.0, min_episodes=1)
        sched.current_difficulty = Difficulty.HARD
        sched.record_episode(1.0)
        assert sched.current_difficulty == Difficulty.HARD


# ===========================================================================
# LongHorizonEnv integration tests
# ===========================================================================

class TestLongHorizonEnv:
    def setup_method(self):
        self.env = LongHorizonEnv(
            [TaskSplittingBlock(), CodeGenBlock(), ReasoningBlock()],
            EnvConfig(),
        )

    def test_reset_returns_observation(self):
        eid, obs = self.env.reset()
        assert eid
        assert obs.task_description
        assert obs.episode_id == eid

    def test_step_returns_result(self):
        eid, obs = self.env.reset(block_name="task_split")
        result = self.env.step(eid, "Task 1: Open the file and read it carefully")
        assert isinstance(result.reward, float)
        assert -1.0 <= result.reward <= 1.0

    def test_done_episode_raises_on_step(self):
        eid, _ = self.env.reset(block_name="code_gen")
        # Force episode to done by exhausting steps
        episode = self.env.registry.get(eid)
        episode.done = True
        with pytest.raises(RuntimeError):
            self.env.step(eid, "anything")

    def test_unknown_episode_raises_key_error(self):
        with pytest.raises(KeyError):
            self.env.step("no-such-id", "action")

    def test_concurrent_episodes_are_independent(self):
        ids = [self.env.reset()[0] for _ in range(5)]
        assert len(set(ids)) == 5

    def test_stale_cleanup(self):
        eid, _ = self.env.reset()
        ep = self.env.registry.get(eid)
        ep.last_active = time.time() - 7200
        removed = self.env.registry.cleanup_stale(max_age_seconds=3600)
        assert removed >= 1
        assert self.env.registry.get(eid) is None

    def test_delete_episode(self):
        eid, _ = self.env.reset()
        self.env.registry.remove(eid)
        assert self.env.registry.get(eid) is None

    def test_metrics_returns_dict(self):
        m = self.env.metrics()
        assert "curriculum" in m
        assert "active_episodes" in m
