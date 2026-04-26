"""
Reward rubrics — hardened, unified, and audited for reward hacking.

Fixes applied vs previous version:
  - CRITICAL-1: task_split now calls SplitQualityScorer for real outcome scoring
  - CRITICAL-2: CodeExecutionRubric now uses sandboxed subprocess via CodeSandbox
  - CRITICAL-3: AntiRepetitionRubric is SKIPPED inside code blocks (avoids false positives)
  - CRITICAL-4: task_split no longer gives free 0.5 floor reward
"""

import re
import textwrap
import hashlib
from openenv.core.rubrics.base import Rubric
from openenv.core.rubrics.containers import WeightedSum
from .base import Action, Observation
from .rewards.multi_reward import CodeSandbox, SplitQualityScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text.lower().strip())

def _content_hash(text: str) -> str:
    return hashlib.md5(_normalize_text(text).encode()).hexdigest()

def _is_code_block(text: str) -> bool:
    """Returns True if the response is primarily a code submission."""
    return bool(re.search(r'```(?:python)?', text))


# ---------------------------------------------------------------------------
# 1. Anti-Repetition Gate
#    Only applied to reasoning/text responses — NOT code blocks.
# ---------------------------------------------------------------------------

class AntiRepetitionRubric(Rubric):
    """
    Hard-zeros repetitive text responses.
    Skipped entirely for code block submissions to avoid false positives.
    """
    def forward(self, action: Action, observation: Observation) -> float:
        text = action.text

        # Skip this check for code submissions (repeated lines are valid in code)
        if _is_code_block(text):
            return 1.0

        # 1. Step numbers must be sequential from 1, capped at 10
        step_numbers = [int(m) for m in re.findall(r'Step\s+(\d+)', text, re.IGNORECASE)]
        if step_numbers:
            if max(step_numbers) > 10:
                return 0.0
            if step_numbers != list(range(1, len(step_numbers) + 1)):
                return 0.0

        # 2. Line uniqueness (only for text responses)
        lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 10]
        if lines:
            hashes = [_content_hash(l) for l in lines]
            if len(set(hashes)) / len(hashes) < 0.7:
                return 0.0

        # 3. Paragraph uniqueness across step bodies
        paragraphs = [p.strip() for p in re.split(r'Step\s+\d+:', text) if len(p.strip()) > 20]
        if len(paragraphs) > 1:
            para_hashes = [_content_hash(p) for p in paragraphs]
            if len(set(para_hashes)) < len(para_hashes) * 0.75:
                return 0.0

        return 1.0


# ---------------------------------------------------------------------------
# 2. Brevity Rubric
# ---------------------------------------------------------------------------

class BrevityRubric(Rubric):
    """Decays reward for verbose responses. Hard zero beyond 3000 chars."""
    def __init__(self, target_chars: int = 800, hard_limit_chars: int = 3000):
        super().__init__()
        self.target = target_chars
        self.hard_limit = hard_limit_chars

    def forward(self, action: Action, observation: Observation) -> float:
        length = len(action.text)
        if length <= self.target:
            return 1.0
        if length >= self.hard_limit:
            return 0.0
        return 1.0 - (length - self.target) / (self.hard_limit - self.target)


# ---------------------------------------------------------------------------
# 3. Code Execution Rubric — Sandboxed
#    Uses the existing CodeSandbox (AST analysis + subprocess isolation).
# ---------------------------------------------------------------------------

class CodeExecutionRubric(Rubric):
    """
    Executes code in an isolated subprocess with AST-level static analysis.
    Blocks: imports, exec/eval, memory bombs, infinite loops (via timeout).
    Score = fraction of test cases passed.
    """
    def __init__(self, timeout: float = 5.0):
        super().__init__()
        self.sandbox = CodeSandbox(timeout=timeout)

    def forward(self, action: Action, observation: Observation) -> float:
        # Extract code
        code_match = re.search(r'```(?:python)?\n?(.*?)```', action.text, re.DOTALL)
        if not code_match:
            return 0.0
        code = code_match.group(1).strip()
        if not code or len(code) < 10:
            return 0.0

        working_memory = (
            observation.metadata.get("working_memory", {})
            if hasattr(observation, 'metadata') else {}
        )
        test_cases = working_memory.get("test_cases", [])
        fn_name = working_memory.get("fn_name", "solution")

        if not test_cases:
            return 0.0

        # Build harness — structured JSON output to prevent stdout manipulation
        harness = textwrap.dedent(f"""\
            _test_cases = {test_cases!r}
            _results = []
            for _args, _expected in _test_cases:
                try:
                    _got = {fn_name}(*_args)
                    _results.append({{"pass": _got == _expected}})
                except Exception as _e:
                    _results.append({{"pass": False, "error": str(_e)}})
            _test_result = _results
        """)

        result = self.sandbox.execute(code, harness)

        if result.blocked:
            return 0.0  # Attempted unsafe code
        if result.timed_out:
            return 0.0  # Infinite loop / too slow
        if not result.success:
            return 0.0

        # result.result is the _test_result list
        if isinstance(result.result, list) and result.result:
            passed = sum(1 for r in result.result if isinstance(r, dict) and r.get("pass"))
            return passed / len(result.result)

        return 0.0


# ---------------------------------------------------------------------------
# 4. Correctness Rubric — Reasoning Tasks
#    Only "Final Answer: X" on its own line counts. No hedging.
# ---------------------------------------------------------------------------

class CorrectnessRubric(Rubric):
    """
    Exact-match correctness against the expected answer.
    Requires a clean 'Final Answer: X' declaration.
    """
    def forward(self, action: Action, observation: Observation) -> float:
        working_memory = (
            observation.metadata.get("working_memory", {})
            if hasattr(observation, 'metadata') else {}
        )
        expected = str(working_memory.get("answer", "")).strip().lower()
        if not expected:
            return 0.0

        match = re.search(r'Final\s+Answer\s*:\s*([^\n]+)', action.text, re.IGNORECASE)
        if not match:
            return 0.0  # No structured final answer = no reward

        agent_answer = match.group(1).strip().lower()
        agent_clean = re.sub(r'[^\w\s]', '', agent_answer).strip()
        expected_clean = re.sub(r'[^\w\s]', '', expected).strip()

        if agent_clean == expected_clean:
            return 1.0

        # Numeric exact match — only if it's the SOLE number in the answer
        agent_nums = re.findall(r'-?\d+(?:\.\d+)?', agent_answer)
        expected_nums = re.findall(r'-?\d+(?:\.\d+)?', expected)
        if agent_nums and expected_nums and agent_nums[0] == expected_nums[0]:
            return 1.0 if len(agent_nums) == 1 else 0.5  # Penalize hedging

        return 0.0


# ---------------------------------------------------------------------------
# 5. Reasoning Density Rubric
#    Penalizes empty or filler step bodies.
# ---------------------------------------------------------------------------

class ReasoningDensityRubric(Rubric):
    """Ensures step bodies have actual content, not filler phrases."""
    FILLER_PATTERNS = [
        r'^think\s*(about|through|carefully)?$',
        r'^analyze\s*(the\s*problem)?$',
        r'^consider\s*(this)?$',
        r'^understand\s*(the\s*question)?$',
        r'^\.\.\.$',
        r'^n/?a$',
        r'^step\s*\d+$',
    ]

    def forward(self, action: Action, observation: Observation) -> float:
        # Not applicable to code blocks
        if _is_code_block(action.text):
            return 1.0

        step_bodies = re.split(r'Step\s+\d+\s*:', action.text, flags=re.IGNORECASE)[1:]
        if not step_bodies:
            return 1.0

        filler_count = 0
        for body in step_bodies:
            first_line = body.split('\n')[0].strip().lower()
            if not first_line or len(first_line) < 5:
                filler_count += 1
                continue
            for pattern in self.FILLER_PATTERNS:
                if re.match(pattern, first_line):
                    filler_count += 1
                    break

        filler_ratio = filler_count / len(step_bodies)
        return max(0.0, 1.0 - filler_ratio) if filler_ratio > 0.3 else 1.0


# ---------------------------------------------------------------------------
# 6. Task Split Quality Rubric
#    Directly calls SplitQualityScorer — real outcome-based scoring.
# ---------------------------------------------------------------------------

class TaskSplitRubric(Rubric):
    """
    Scores task decomposition quality using SplitQualityScorer.
    Evaluates: completeness, atomicity, ordering, specificity, non-redundancy.
    """
    def __init__(self):
        super().__init__()
        self.scorer = SplitQualityScorer()

    def forward(self, action: Action, observation: Observation) -> float:
        working_memory = (
            observation.metadata.get("working_memory", {})
            if hasattr(observation, 'metadata') else {}
        )
        reference_split = working_memory.get("reference_split", [])
        problem_desc = working_memory.get("task_description", "")
        min_tasks = working_memory.get("min_tasks", 3)
        max_tasks = working_memory.get("max_tasks", 8)

        if not reference_split:
            return 0.0

        # Parse numbered tasks from agent response
        tasks = []
        for line in action.text.strip().splitlines():
            m = re.match(r'^\s*(?:Task\s+)?\d+[.):–\-]\s*(.+)', line, re.IGNORECASE)
            if m:
                tasks.append(m.group(1).strip())

        if not tasks:
            return 0.0

        components = self.scorer.score(
            agent_tasks=tasks,
            reference_split=reference_split,
            problem_description=problem_desc,
            min_tasks=min_tasks,
            max_tasks=max_tasks,
        )
        return components.total()


# ---------------------------------------------------------------------------
# Composite: MultiBlockRubric
# ---------------------------------------------------------------------------

class MultiBlockRubric(WeightedSum):
    """
    Unified reward rubric routing to the right scorer per block type.

    Weights must sum to 1.0 (OpenEnv requirement).
    The forward() method overrides the weighted sum with custom routing logic.
    """
    def __init__(self):
        rubrics = [
            AntiRepetitionRubric(),    # rubric_0 — gate (text only)
            BrevityRubric(),           # rubric_1 — length control
            CodeExecutionRubric(),     # rubric_2 — sandboxed code outcome
            CorrectnessRubric(),       # rubric_3 — reasoning outcome
            ReasoningDensityRubric(),  # rubric_4 — content quality
            TaskSplitRubric(),         # rubric_5 — task split quality
        ]
        # Nominal weights (must sum to 1.0). Actual logic is in forward().
        weights = [0.15, 0.05, 0.25, 0.25, 0.15, 0.15]
        super().__init__(rubrics, weights)

    def forward(self, action: Action, observation: Observation) -> float:
        active_block = (
            observation.metadata.get("active_block", "default")
            if hasattr(observation, 'metadata') else "default"
        )

        brevity = self.rubric_1(action, observation)
        density = self.rubric_4(action, observation)
        # Quality multiplier: [0.5, 1.0] — scales outcome score
        quality = 0.5 + 0.25 * brevity + 0.25 * density

        if active_block == "code_gen":
            # Gate: anti-repetition not applied to code (would false-positive)
            execution_score = self.rubric_2(action, observation)
            return execution_score * quality

        elif active_block == "reasoning":
            # Gate: anti-repetition applied to text responses only
            anti_rep = self.rubric_0(action, observation)
            if anti_rep < 0.5:
                return 0.05  # Hard penalize looping/padding
            correctness = self.rubric_3(action, observation)
            return correctness * quality

        elif active_block == "task_split":
            # Gate: anti-repetition applied
            anti_rep = self.rubric_0(action, observation)
            if anti_rep < 0.5:
                return 0.05
            # Real outcome scoring — no free floor reward
            split_score = self.rubric_5(action, observation)
            return split_score * quality

        # Unknown block — return quality signals only (no free reward)
        return quality * 0.0  # Zero until explicitly handled
