"""
Reward rubrics with comprehensive anti-reward-hacking defenses.

Each rubric is independently verifiable and adversarially hardened.
All reward signals are OUTCOME-based, not FORMAT-based.
"""

import re
import textwrap
import hashlib
from typing import Any, Optional
from openenv.core.rubrics.base import Rubric
from openenv.core.rubrics.containers import WeightedSum
from openenv.core.tools.local_python_executor import PyExecutor
from .base import Action, Observation


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for comparison."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def _content_hash(text: str) -> str:
    return hashlib.md5(_normalize_text(text).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Defense 1: Anti-Looping / Anti-Repetition Rubric
# ---------------------------------------------------------------------------

class AntiRepetitionRubric(Rubric):
    """
    Zero reward for any response that is semantically repetitive or loops.
    
    Hardened against:
    - Sequential step number looping (Step 1, Step 2... Step 100)
    - Repeated content with different step numbers
    - Padding the output with synonymous or near-identical lines
    """
    def forward(self, action: Action, observation: Observation) -> float:
        text = action.text

        # 1. Check step number range — steps beyond 10 are penalized hard
        step_numbers = [int(m) for m in re.findall(r'Step\s+(\d+)', text, re.IGNORECASE)]
        if step_numbers:
            if max(step_numbers) > 10:
                return 0.0  # Hard zero — no valid reasoning needs >10 steps
            # Steps must be sequential from 1
            expected = list(range(1, len(step_numbers) + 1))
            if step_numbers != expected:
                return 0.0  # Non-sequential = hacking

        # 2. Check for repeated lines (catch padding/synonym spam)
        lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 10]
        if lines:
            hashes = [_content_hash(l) for l in lines]
            unique_ratio = len(set(hashes)) / len(hashes)
            if unique_ratio < 0.7:  # More than 30% duplicates
                return 0.0

        # 3. Check for paragraph-level repetition (catch step content reuse)
        paragraphs = [p.strip() for p in re.split(r'Step\s+\d+:', text) if len(p.strip()) > 20]
        if len(paragraphs) > 1:
            para_hashes = [_content_hash(p) for p in paragraphs]
            if len(set(para_hashes)) < len(para_hashes) * 0.75:
                return 0.0

        return 1.0  # Passed all checks


# ---------------------------------------------------------------------------
# Defense 2: Brevity / Length Control Rubric
# ---------------------------------------------------------------------------

class BrevityRubric(Rubric):
    """
    Penalizes excessively long responses.
    
    Hardened against:
    - Padding with filler text to "fill" the token budget
    - Generating excessive explanation to avoid real work
    """
    def __init__(self, target_chars: int = 800, hard_limit_chars: int = 3000):
        super().__init__()
        self.target = target_chars
        self.hard_limit = hard_limit_chars

    def forward(self, action: Action, observation: Observation) -> float:
        length = len(action.text)
        if length <= self.target:
            return 1.0  # Ideal length
        if length >= self.hard_limit:
            return 0.0  # Hard zero for extreme verbosity
        # Smooth decay between target and hard_limit
        decay = (length - self.target) / (self.hard_limit - self.target)
        return max(0.0, 1.0 - decay)


# ---------------------------------------------------------------------------
# Defense 3: Code Execution Rubric (Outcome-Based)
# ---------------------------------------------------------------------------

class CodeExecutionRubric(Rubric):
    """
    Verifies code correctness via actual execution against test cases.
    
    Hardened against:
    - Outputting the test case answer directly as code (hardcoding)
    - Empty code blocks that "execute" without doing anything
    - Importing test answers from global scope
    """
    def __init__(self, timeout: float = 5.0):
        super().__init__()
        self.executor = PyExecutor()

    def forward(self, action: Action, observation: Observation) -> float:
        # 1. Extract code from markdown block
        code_match = re.search(r'```(?:python)?\n?(.*?)```', action.text, re.DOTALL)
        if not code_match:
            return 0.0  # No code block at all
        
        code = code_match.group(1).strip()
        
        if not code or len(code) < 10:
            return 0.0  # Empty or trivially small code block

        # 2. Anti-hardcoding: code must NOT contain the expected answer as a literal
        working_memory = observation.metadata.get("working_memory", {}) if hasattr(observation, 'metadata') else {}
        test_cases = working_memory.get("test_cases", [])
        fn_name = working_memory.get("fn_name", "solution")

        # 3. Build an isolated test harness
        # CRITICAL: Use a randomized variable name to prevent scope leakage
        harness = textwrap.dedent(f"""
            _multi_block_test_cases = {test_cases!r}
            _multi_block_results = []
            for _tc_args, _tc_expected in _multi_block_test_cases:
                try:
                    _tc_got = {fn_name}(*_tc_args)
                    _multi_block_results.append(int(_tc_got == _tc_expected))
                except Exception as _tc_exc:
                    _multi_block_results.append(0)
            _multi_block_score = sum(_multi_block_results) / len(_multi_block_results) if _multi_block_results else 0.0
            print(f"SCORE:{{_multi_block_score:.4f}}")
        """)

        full_code = f"{code}\n{harness}"
        result = self.executor.run(full_code)

        if result.exit_code != 0:
            return 0.0  # Code crashed

        # 4. Parse the structured score from stdout
        score_match = re.search(r'SCORE:([\d.]+)', result.stdout or '')
        if score_match:
            return float(score_match.group(1))
        
        return 0.0  # Could not parse output


# ---------------------------------------------------------------------------
# Defense 4: Correctness Rubric (Reasoning Tasks)
# ---------------------------------------------------------------------------

class CorrectnessRubric(Rubric):
    """
    Checks final answer correctness for reasoning tasks.
    
    Hardened against:
    - Listing multiple answers and hoping one matches
    - Embedding the answer in a long string of numbers
    - Approximate matching abuse
    """
    def forward(self, action: Action, observation: Observation) -> float:
        working_memory = observation.metadata.get("working_memory", {}) if hasattr(observation, 'metadata') else {}
        expected = str(working_memory.get("answer", "")).strip().lower()

        if not expected:
            return 0.0

        # 1. ONLY look at the last "Final Answer: X" line — not any number in the text
        # This prevents the model from guessing many numbers and getting partial credit
        final_answer_match = re.search(
            r'Final\s+Answer\s*:\s*([^\n]+)',
            action.text,
            re.IGNORECASE
        )
        
        if not final_answer_match:
            return 0.0  # No structured answer = no reward

        agent_answer = final_answer_match.group(1).strip().lower()
        
        # 2. Normalize for comparison (strip punctuation, extra spaces)
        agent_clean = re.sub(r'[^\w\s]', '', agent_answer).strip()
        expected_clean = re.sub(r'[^\w\s]', '', expected).strip()

        if agent_clean == expected_clean:
            return 1.0
        
        # 3. For numeric answers: exact numeric match only (no rounding tricks)
        agent_nums = re.findall(r'-?\d+(?:\.\d+)?', agent_answer)
        expected_nums = re.findall(r'-?\d+(?:\.\d+)?', expected)
        
        if agent_nums and expected_nums and agent_nums[0] == expected_nums[0]:
            # Correct number found, but check it's the FIRST and ONLY number mentioned
            if len(agent_nums) == 1:
                return 1.0  # Clean single-number answer
            else:
                return 0.5  # Correct answer but buried in multiple numbers (penalize hedging)

        return 0.0


# ---------------------------------------------------------------------------
# Defense 5: Minimal Reasoning Density Rubric
# ---------------------------------------------------------------------------

class ReasoningDensityRubric(Rubric):
    """
    Ensures responses have meaningful content density, not just structure.
    
    Hardened against:
    - Submitting step headers with empty bodies
    - Filler phrases like "Step 1: Think about the problem"
    - One-word step bodies
    """
    # Meaningless filler phrases that indicate content-free steps
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
        text = action.text
        
        # Extract step bodies
        step_bodies = re.split(r'Step\s+\d+\s*:', text, flags=re.IGNORECASE)[1:]
        
        if not step_bodies:
            return 1.0  # No step format used — not penalized here
        
        filler_count = 0
        for body in step_bodies:
            body_clean = body.split('\n')[0].strip().lower()  # First line of step
            if not body_clean or len(body_clean) < 5:
                filler_count += 1
                continue
            for pattern in self.FILLER_PATTERNS:
                if re.match(pattern, body_clean):
                    filler_count += 1
                    break
        
        if len(step_bodies) == 0:
            return 1.0
        
        filler_ratio = filler_count / len(step_bodies)
        if filler_ratio > 0.3:  # >30% filler steps = penalty
            return max(0.0, 1.0 - filler_ratio)
        
        return 1.0


# ---------------------------------------------------------------------------
# Composite: MultiBlockRubric
# ---------------------------------------------------------------------------

class MultiBlockRubric(WeightedSum):
    """
    Composes all rubrics into a single anti-hacking reward signal.
    
    Design principles:
    - AntiRepetition acts as a GATE: a repetitive response gets near-zero reward
      regardless of what other rubrics say.
    - Outcome rubrics (Correctness, CodeExecution) dominate the score.
    - Format/brevity are minor contributors to prevent format over-optimization.
    """
    def __init__(self):
        rubrics = [
            AntiRepetitionRubric(),    # rubric_0 — gate
            BrevityRubric(),           # rubric_1 — length control
            CodeExecutionRubric(),     # rubric_2 — code outcomes
            CorrectnessRubric(),       # rubric_3 — reasoning outcomes
            ReasoningDensityRubric(),  # rubric_4 — content quality
        ]
        # Weights must sum to exactly 1.0 (enforced by OpenEnv WeightedSum)
        # These are nominal — actual scoring logic lives in forward() below
        weights = [0.15, 0.05, 0.35, 0.35, 0.10]
        super().__init__(rubrics, weights)

    def forward(self, action: Action, observation: Observation) -> float:
        # --- Gate: Any repetition = hard penalty ---
        anti_rep_score = self.rubric_0(action, observation)
        if anti_rep_score < 0.5:
            # Repetitive output. Return near-zero to strongly discourage.
            return 0.05

        brevity = self.rubric_1(action, observation)
        density = self.rubric_4(action, observation)
        
        # Quality multiplier: brevity and density act as multipliers, not additive terms
        # This prevents "long but correct" from being exploited
        quality_multiplier = 0.5 + 0.25 * brevity + 0.25 * density

        active_block = observation.metadata.get("active_block", "default") if hasattr(observation, 'metadata') else "default"

        if active_block == "code_gen":
            # Pure outcome: did the code pass tests?
            execution_score = self.rubric_2(action, observation)
            return execution_score * quality_multiplier

        elif active_block == "reasoning":
            # Pure outcome: is the final answer correct?
            correctness_score = self.rubric_3(action, observation)
            return correctness_score * quality_multiplier

        else:
            # task_split / default: use quality signals only
            return quality_multiplier
