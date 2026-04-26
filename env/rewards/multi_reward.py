"""
Reward calculation: CodeSandbox, SplitQualityScorer, StateIntegrityChecker,
MultiRewardCalculator.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any


# ---------------------------------------------------------------------------
# Shared dataclass
# ---------------------------------------------------------------------------

@dataclass
class RewardComponents:
    execution_success: float = 0.0
    format_compliance: float = 0.0
    correctness: float = 0.0
    timeout_penalty: float = 0.0   # 0 or -1
    anti_hack_penalty: float = 0.0  # 0 or -1
    step_efficiency: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "execution_success": self.execution_success,
            "format_compliance": self.format_compliance,
            "correctness": self.correctness,
            "timeout_penalty": self.timeout_penalty,
            "anti_hack_penalty": self.anti_hack_penalty,
            "step_efficiency": self.step_efficiency,
        }

    def total(self) -> float:
        raw = (
            0.35 * self.execution_success
            + 0.10 * self.format_compliance
            + 0.40 * self.correctness
            + 0.05 * self.timeout_penalty
            + 0.10 * self.anti_hack_penalty
            + self.step_efficiency
        )
        return max(-1.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# CodeSandbox
# ---------------------------------------------------------------------------

_FORBIDDEN_NAMES = {
    "exec", "eval", "compile", "__import__", "open", "globals",
    "locals", "vars", "breakpoint", "__builtins__", "__subclasses__",
    "memoryview", "__loader__", "__spec__",
}

_SAFE_BUILTINS = (
    "print len range enumerate zip map filter list dict set tuple "
    "int float str bool abs min max sum sorted reversed isinstance "
    "type repr round divmod pow hex bin oct all any next iter "
    "hasattr getattr setattr True False None"
).split()

_RUNNER_TEMPLATE = textwrap.dedent("""\
    import sys, json, builtins as _b

    _safe = {{k: getattr(_b, k) for k in {safe_names!r} if hasattr(_b, k)}}
    __builtins__ = _safe

    try:
    {indented_code}
    {indented_test}
        sys.stdout.write(json.dumps({{"success": True, "result": _test_result}}))
    except Exception as _e:
        sys.stdout.write(json.dumps({{"success": False, "error": str(_e)}}))
""")


@dataclass
class ExecResult:
    success: bool = False
    blocked: bool = False
    timed_out: bool = False
    block_reason: str = ""
    result: Any = None
    error: str = ""
    raw_output: str = ""


class CodeSandbox:
    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout

    # -- Layer 1: AST static analysis ----------------------------------------

    # Attribute names used in class-traversal attacks
    _FORBIDDEN_ATTRS = {"__class__", "__mro__", "__subclasses__", "__bases__",
                        "__globals__", "__builtins__", "__code__", "__func__"}

    def _ast_check(self, code: str) -> tuple[bool, str]:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"SyntaxError: {e}"

        for node in ast.walk(tree):
            # Block imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return False, "Import statements are not allowed"
            # Block global / nonlocal
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                return False, "global/nonlocal statements are not allowed"
            # Block forbidden Name nodes (exec, eval, open, ...)
            if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
                return False, f"Forbidden name: {node.id}"
            # Block class-traversal attribute access (__class__, __mro__, etc.)
            if isinstance(node, ast.Attribute) and node.attr in self._FORBIDDEN_ATTRS:
                return False, f"Forbidden attribute access: {node.attr}"
            # Block memory bombs: [x] * N where N > 1_000_000
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                for operand in (node.left, node.right):
                    if isinstance(operand, ast.Constant) and isinstance(operand.value, (int, float)):
                        if operand.value > 1_000_000:
                            return False, "Memory bomb detected (multiplier > 1e6)"

        return True, ""

    # -- Layer 2: subprocess isolation ---------------------------------------

    def execute(self, code: str, test_harness: str, timeout: float | None = None) -> ExecResult:
        ok, reason = self._ast_check(code)
        if not ok:
            return ExecResult(blocked=True, block_reason=reason)

        t = timeout or self.timeout
        indented_code = textwrap.indent(code, "    ")
        indented_test = textwrap.indent(test_harness, "    ")
        runner = _RUNNER_TEMPLATE.format(
            safe_names=_SAFE_BUILTINS,
            indented_code=indented_code,
            indented_test=indented_test,
        )

        try:
            proc = subprocess.run(
                [sys.executable, "-c", runner],
                capture_output=True,
                text=True,
                timeout=t,
            )
            raw = proc.stdout.strip()
            if not raw:
                return ExecResult(success=False, error=proc.stderr[:500], raw_output=proc.stderr)
            data = json.loads(raw)
            return ExecResult(
                success=data.get("success", False),
                result=data.get("result"),
                error=data.get("error", ""),
                raw_output=raw,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(timed_out=True)
        except Exception as e:
            return ExecResult(success=False, error=str(e))


# ---------------------------------------------------------------------------
# StateIntegrityChecker
# ---------------------------------------------------------------------------

class StateIntegrityChecker:
    @staticmethod
    def snapshot(state: dict) -> str:
        raw = json.dumps(state, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def check(self, state: dict, expected_hash: str) -> bool:
        return self.snapshot(state) == expected_hash

    @staticmethod
    def detect_cache_abuse(action_history: list[str]) -> bool:
        if len(action_history) < 3:
            return False
        last3 = action_history[-3:]
        return last3[0] == last3[1] == last3[2]

    @staticmethod
    def detect_reward_probe(reward_history: list[float]) -> bool:
        if len(reward_history) < 6:
            return False
        diffs = [abs(reward_history[i] - reward_history[i - 1]) for i in range(1, len(reward_history))]
        avg_diff = sum(diffs) / len(diffs)
        # Suspiciously alternating if avg swing > 0.5 and last 6 rewards alternate sign
        signs = [1 if r >= 0 else -1 for r in reward_history[-6:]]
        alternating = all(signs[i] != signs[i + 1] for i in range(len(signs) - 1))
        return alternating and avg_diff > 0.4


# ---------------------------------------------------------------------------
# SplitQualityScorer  (for TaskSplittingBlock)
# ---------------------------------------------------------------------------

@dataclass
class SplitComponents:
    completeness: float = 0.0
    atomicity: float = 0.0
    ordering_score: float = 1.0
    specificity: float = 0.0
    non_redundancy: float = 1.0
    anti_hack_penalty: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "completeness": self.completeness,
            "atomicity": self.atomicity,
            "ordering_score": self.ordering_score,
            "specificity": self.specificity,
            "non_redundancy": self.non_redundancy,
            "anti_hack_penalty": self.anti_hack_penalty,
        }

    def total(self) -> float:
        raw = (
            0.35 * self.completeness
            + 0.20 * self.atomicity
            + 0.20 * self.ordering_score
            + 0.15 * self.specificity
            + 0.10 * self.non_redundancy
            + self.anti_hack_penalty
        )
        return max(-1.0, min(1.0, raw))


# Simple English verb wordlist for specificity check (no NLTK dependency)
_COMMON_VERBS = {
    "write", "create", "build", "implement", "define", "add", "remove", "update",
    "delete", "fetch", "read", "parse", "validate", "check", "test", "run",
    "calculate", "compute", "sort", "filter", "map", "reduce", "send", "receive",
    "connect", "open", "close", "load", "save", "export", "import", "convert",
    "handle", "process", "generate", "return", "set", "get", "store", "retrieve",
    "initialize", "configure", "format", "display", "render", "log", "raise",
    "catch", "throw", "compare", "merge", "split", "join", "extract", "insert",
}

# Keyword dependency rules: (earlier_keyword, later_keyword)
_ORDERING_RULES = [
    ({"schema", "create table", "define model", "design schema"}, {"insert", "store", "save to db", "write to db"}),
    ({"install", "setup", "configure"}, {"run", "start", "launch", "execute"}),
    ({"parse", "read", "load", "fetch"}, {"process", "transform", "validate", "filter"}),
    ({"validate", "check", "verify"}, {"store", "save", "submit", "send"}),
]


class SplitQualityScorer:
    def score(
        self,
        agent_tasks: list[str],
        reference_split: list[str],
        problem_description: str,
        min_tasks: int,
        max_tasks: int,
    ) -> SplitComponents:
        c = SplitComponents()

        if not agent_tasks:
            # All components stay 0 (ordering and non_redundancy are vacuously 1 but meaningless)
            c.ordering_score = 0.0
            c.non_redundancy = 0.0
            return c

        # Cap at max_tasks (no reward for padding)
        tasks = agent_tasks[:max_tasks]

        # --- completeness: keyword coverage over reference split ---
        c.completeness = self._completeness(tasks, reference_split)
        if len(agent_tasks) < min_tasks:
            c.completeness = min(c.completeness, 0.5)

        # --- atomicity ---
        c.atomicity = self._atomicity(tasks)

        # --- ordering ---
        c.ordering_score = self._ordering(tasks)

        # --- specificity ---
        c.specificity = self._specificity(tasks)

        # --- non-redundancy ---
        c.non_redundancy = self._non_redundancy(tasks)

        # --- anti-hack: verbatim restate ---
        if self._is_verbatim_restate(tasks, problem_description):
            c.anti_hack_penalty = -0.5

        return c

    def _completeness(self, tasks: list[str], reference: list[str]) -> float:
        """Jaccard-based completeness: rewards precision AND recall, not just recall.
        Prevents keyword-dump tasks from scoring high."""
        if not reference:
            return 1.0
        scores = []
        for ref_task in reference:
            ref_words = set(ref_task.lower().split())
            # Jaccard: intersection / union — penalizes tasks that are too long/too short
            best = max(
                len(ref_words & set(t.lower().split())) /
                max(len(ref_words | set(t.lower().split())), 1)
                for t in tasks
            )
            scores.append(best)
        return sum(scores) / len(scores)

    def _atomicity(self, tasks: list[str]) -> float:
        _split_markers = re.compile(
            r'\band then\b|\band also\b|\bthen\b|\band\b.*\balso\b', re.IGNORECASE
        )
        scores = []
        for t in tasks:
            words = t.split()
            if len(words) > 25:
                scores.append(0.0)
            elif _split_markers.search(t):
                scores.append(0.5)
            else:
                scores.append(1.0)
        return sum(scores) / len(scores)

    def _ordering(self, tasks: list[str]) -> float:
        violations = 0
        joined = [t.lower() for t in tasks]
        for earlier_kws, later_kws in _ORDERING_RULES:
            earlier_idx = next(
                (i for i, t in enumerate(joined) if any(k in t for k in earlier_kws)), None
            )
            later_idx = next(
                (i for i, t in enumerate(joined) if any(k in t for k in later_kws)), None
            )
            if earlier_idx is not None and later_idx is not None:
                if earlier_idx > later_idx:
                    violations += 1
        return max(0.0, 1.0 - violations * 0.25)

    def _specificity(self, tasks: list[str]) -> float:
        scores = []
        for t in tasks:
            words = t.lower().split()
            has_verb = any(w in _COMMON_VERBS for w in words)
            long_enough = len(words) >= 5
            scores.append(1.0 if (has_verb and long_enough) else 0.0)
        return sum(scores) / len(scores)

    def _non_redundancy(self, tasks: list[str]) -> float:
        if len(tasks) < 2:
            return 1.0
        pairs = [(tasks[i], tasks[j]) for i in range(len(tasks)) for j in range(i + 1, len(tasks))]
        redundant = sum(
            1 for a, b in pairs
            if SequenceMatcher(None, a.lower(), b.lower()).ratio() > 0.8
        )
        return 1.0 - redundant / len(pairs)

    def _is_verbatim_restate(self, tasks: list[str], problem: str) -> bool:
        if len(tasks) != 1:
            return False
        ratio = SequenceMatcher(None, tasks[0].lower(), problem.lower()).ratio()
        return ratio > 0.85


# ---------------------------------------------------------------------------
# MultiRewardCalculator
# ---------------------------------------------------------------------------

class MultiRewardCalculator:
    def __init__(self):
        self.sandbox = CodeSandbox()
        self.integrity = StateIntegrityChecker()
        self.split_scorer = SplitQualityScorer()

    # -- Code gen reward ------------------------------------------------------

    def compute_code_gen(
        self,
        action: str,
        task_meta: dict,
        action_history: list[str],
        attempt_count: int,
        min_steps: int = 1,
    ) -> RewardComponents:
        c = RewardComponents()

        # Format compliance: extract code block
        code = self._extract_code(action)
        if code is None:
            c.format_compliance = 0.0
            return c
        c.format_compliance = 1.0 if "```" in action else 0.5

        # Anti-hack: cache abuse
        if self.integrity.detect_cache_abuse(action_history):
            c.anti_hack_penalty = -1.0
            return c

        # Sandbox execution
        test_cases = task_meta.get("test_cases", [])
        fn_name = task_meta.get("fn_name", "solution")
        harness = self._build_harness(fn_name, test_cases)
        result = self.sandbox.execute(code, harness)

        if result.blocked:
            c.anti_hack_penalty = -1.0
            return c

        if result.timed_out:
            c.timeout_penalty = -1.0
            return c

        c.execution_success = 1.0 if result.success else 0.0

        if result.success and isinstance(result.result, list):
            passed = sum(1 for r in result.result if r.get("pass"))
            total = len(result.result) or 1
            c.correctness = passed / total
        elif not result.success:
            c.correctness = 0.0

        c.step_efficiency = -0.01 * max(0, attempt_count - min_steps)
        return c

    # -- Reasoning reward -----------------------------------------------------

    def compute_reasoning(
        self,
        action: str,
        task_meta: dict,
        action_history: list[str],
        step_count: int,
    ) -> RewardComponents:
        c = RewardComponents()

        if self.integrity.detect_cache_abuse(action_history):
            c.anti_hack_penalty = -1.0
            return c

        # Output length guard
        if len(action) > 4000:
            c.format_compliance = 0.0
            return c

        # execution_success: "Final Answer:" or "<answer>" present
        final_match = re.search(r'(?:Final Answer|Answer)\s*:\s*(.+)|<answer>(.*?)</answer>', action, re.IGNORECASE | re.DOTALL)
        has_final = bool(final_match)
        c.execution_success = 1.0 if has_final else 0.0

        # format_compliance: numbered steps or <think> tags
        numbered = re.findall(r'Step\s+\d+\s*:|<think>', action, re.IGNORECASE)
        min_steps = task_meta.get("min_steps", 2)
        c.format_compliance = min(1.0, len(numbered) / min_steps)

        # correctness
        answer_str = task_meta.get("answer", "")
        if final_match:
            agent_ans = (final_match.group(1) or final_match.group(2)).strip()
        else:
            # Generalize: if no tag is found, fallback to the last number in the text
            nums = re.findall(r'-?\d+\.?\d*', action)
            agent_ans = nums[-1] if nums else ""
            
        c.correctness = self._score_answer(agent_ans, str(answer_str))
        
        # Reward finding the truth even without perfect formatting
        if c.correctness > 0 and not has_final:
            c.execution_success = 0.5

        # Anti-hack: verbatim copy
        task_desc = task_meta.get("task_description", "")
        if self._is_verbatim_reasoning_copy(action, task_desc, step_count, min_steps):
            c.anti_hack_penalty = -1.0

        return c

    # -- Helpers --------------------------------------------------------------

    @staticmethod
    def _extract_code(action: str) -> str | None:
        m = re.search(r'```(?:python)?\n?(.*?)```', action, re.DOTALL)
        if m:
            return m.group(1).strip()
        # Bare code fallback: must contain "def "
        if "def " in action:
            return action.strip()
        return None

    @staticmethod
    def _build_harness(fn_name: str, test_cases: list) -> str:
        return textwrap.dedent(f"""\
            _test_cases = {test_cases!r}
            _results = []
            for _args, _expected in _test_cases:
                try:
                    _got = {fn_name}(*_args)
                    _results.append({{"pass": _got == _expected, "got": repr(_got), "expected": repr(_expected)}})
                except Exception as _e:
                    _results.append({{"pass": False, "error": str(_e)}})
            _test_result = _results
        """)

    @staticmethod
    def _score_answer(agent: str, expected: str) -> float:
        # Exact match first
        agent_nums = re.findall(r'-?\d+\.?\d*', agent)
        exp_nums = re.findall(r'-?\d+\.?\d*', expected)
        if agent_nums and exp_nums and agent_nums[0] == exp_nums[0]:
            return 1.0
        # Fuzzy match
        ratio = SequenceMatcher(None, agent.lower(), expected.lower()).ratio()
        return 0.5 if ratio > 0.6 else 0.0

    @staticmethod
    def _is_verbatim_reasoning_copy(action: str, task_desc: str, step_count: int, min_steps: int) -> bool:
        final_match = re.search(r'(?:Final Answer|Answer)\s*:\s*(.+)|<answer>(.*?)</answer>', action, re.IGNORECASE | re.DOTALL)
        if final_match:
            final_ans = (final_match.group(1) or final_match.group(2)).strip()
        else:
            final_ans = action
        task_nums = set(re.findall(r'\d+\.?\d*', task_desc))
        ans_nums = set(re.findall(r'\d+\.?\d*', final_ans))
        if ans_nums and ans_nums.issubset(task_nums) and step_count < min_steps:
            return True
        return False
