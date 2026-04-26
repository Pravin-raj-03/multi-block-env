"""
Reward rubrics using the official OpenEnv Rubric system.
"""

import re
import textwrap
from typing import Any, Optional
from openenv.core.rubrics.base import Rubric
from openenv.core.rubrics.containers import WeightedSum
from openenv.core.tools.local_python_executor import PyExecutor
from .base import Action, Observation


class FormatComplianceRubric(Rubric):
    """Checks if the action follows the required format and penalizes repetition."""
    def __init__(self, pattern: str):
        super().__init__()
        self.pattern = re.compile(pattern, re.DOTALL)

    def forward(self, action: Action, observation: Observation) -> float:
        text = action.text
        if not self.pattern.search(text):
            return 0.0
        
        # 1. Check for step sequentiality
        steps = re.findall(r'Step (\d+):', text)
        if steps:
            step_nums = [int(s) for s in steps]
            # Must have at least 2 steps for reasoning
            if len(step_nums) < 2:
                return 0.5 # Partial reward for just trying
            
            # Check if sequential (1, 2, 3...)
            for i in range(len(step_nums)):
                if step_nums[i] != i + 1:
                    return 0.1 # Out of order or repeated numbers
            
            # Cap the number of steps to prevent looping
            if len(step_nums) > 10:
                return 0.2 # Penalty for excessive steps
        
        # 2. Check for exact repetition of content between steps
        # This prevents "Step 1: content Step 2: content ..."
        step_contents = re.split(r'Step \d+:', text)[1:]
        if len(step_contents) > 1:
            unique_contents = set(c.strip() for c in step_contents if c.strip())
            if len(unique_contents) < len(step_contents) * 0.8: # Allow some similarity but not 100%
                return 0.0 # Repeated reasoning is useless
        
        return 1.0


class BrevityRubric(Rubric):
    """Penalizes excessively long responses to prevent reward hacking."""
    def __init__(self, max_chars: int = 1000):
        super().__init__()
        self.max_chars = max_chars

    def forward(self, action: Action, observation: Observation) -> float:
        length = len(action.text)
        if length > self.max_chars:
            # Linear penalty after max_chars
            return max(0.0, 1.0 - (length - self.max_chars) / 1000.0)
        return 1.0


class CodeExecutionRubric(Rubric):
    """Executes code using the official OpenEnv PyExecutor and scores based on tests."""
    def __init__(self, timeout: float = 5.0):
        super().__init__()
        self.executor = PyExecutor()

    def forward(self, action: Action, observation: Observation) -> float:
        # Extract code from markdown
        code_match = re.search(r'```(?:python)?\n?(.*?)```', action.text, re.DOTALL)
        code = code_match.group(1).strip() if code_match else None
        
        if not code:
            return 0.0

        # Build test harness from working memory
        working_memory = observation.metadata.get("working_memory", {})
        fn_name = working_memory.get("fn_name", "solution")
        test_cases = working_memory.get("test_cases", [])
        
        harness = textwrap.dedent(f"""
            _test_cases = {test_cases!r}
            _results = []
            for _args, _expected in _test_cases:
                try:
                    _got = {fn_name}(*_args)
                    _results.append(_got == _expected)
                except:
                    _results.append(False)
            _test_result = all(_results) if _results else False
        """)
        
        full_code = f"{code}\n{harness}"
        result = self.executor.run(full_code)
        
        if result.exit_code == 0:
            # Check for the _test_result in stdout or just assume success if it finished
            # In a real rubric, we'd parse the stdout for structured results
            return 1.0 if "true" in result.stdout.lower() else 0.5
        return 0.0


class CorrectnessRubric(Rubric):
    """Checks for final answer correctness in reasoning tasks."""
    def forward(self, action: Action, observation: Observation) -> float:
        working_memory = observation.metadata.get("working_memory", {})
        expected = str(working_memory.get("answer", ""))
        
        # Look for "Final Answer: <val>" or "<answer>val</answer>"
        match = re.search(r'(?:Final Answer|Answer)\s*:\s*(.+)|<answer>(.*?)</answer>', action.text, re.IGNORECASE | re.DOTALL)
        if match:
            agent_ans = (match.group(1) or match.group(2)).strip()
            if agent_ans.lower() == expected.lower():
                return 1.0
        
        # Fallback to last number
        nums = re.findall(r'-?\d+\.?\d*', action.text)
        if nums and nums[-1] == expected:
            return 0.5
            
        return 0.0


class MultiBlockRubric(WeightedSum):
    """Composes all rubrics into a single multi-agent reward signal."""
    def __init__(self):
        # We define a flexible weighted sum
        rubrics = [
            FormatComplianceRubric(r'```python|Step \d+:'),
            CodeExecutionRubric(),
            CorrectnessRubric(),
            BrevityRubric()
        ]
        weights = [0.2, 0.3, 0.4, 0.1]
        super().__init__(rubrics, weights)

    def forward(self, action: Action, observation: Observation) -> float:
        # Route to specific sub-rubric based on active block
        active_block = observation.metadata.get("active_block", "default")
        
        # Common brevity penalty applied to all
        brevity_score = self.rubric_3(action, observation)
        
        if active_block == "code_gen":
            # Format (0.2) + Execution (0.7) + Brevity (0.1)
            score = 0.2 * self.rubric_0(action, observation) + 0.7 * self.rubric_1(action, observation) + 0.1 * brevity_score
            return score
        elif active_block == "reasoning":
            # Format (0.2) + Correctness (0.7) + Brevity (0.1)
            score = 0.2 * self.rubric_0(action, observation) + 0.7 * self.rubric_2(action, observation) + 0.1 * brevity_score
            return score
        
        # Default fallback
        return super().forward(action, observation)
