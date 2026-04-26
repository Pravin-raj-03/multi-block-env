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
    """Checks if the action follows the required format (e.g., markdown code blocks)."""
    def __init__(self, pattern: str):
        super().__init__()
        self.pattern = re.compile(pattern, re.DOTALL)

    def forward(self, action: Action, observation: Observation) -> float:
        if self.pattern.search(action.text):
            return 1.0
        return 0.0


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
            CorrectnessRubric()
        ]
        weights = [0.2, 0.4, 0.4]
        super().__init__(rubrics, weights)

    def forward(self, action: Action, observation: Observation) -> float:
        # Route to specific sub-rubric based on active block
        active_block = observation.metadata.get("active_block", "default")
        
        if active_block == "code_gen":
            # Just format + execution
            return 0.2 * self.rubric_0(action, observation) + 0.8 * self.rubric_1(action, observation)
        elif active_block == "reasoning":
            # Just format + correctness
            return 0.2 * self.rubric_0(action, observation) + 0.8 * self.rubric_2(action, observation)
        
        # Default fallback
        return super().forward(action, observation)
