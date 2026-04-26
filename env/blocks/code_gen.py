"""CodeGenBlock — agent writes Python code that must pass unit tests."""

from __future__ import annotations

import random
import textwrap
from dataclasses import dataclass

from env.base import Difficulty, EnvBlock
from env.rewards.multi_reward import MultiRewardCalculator


# ---------------------------------------------------------------------------
# Task bank
# ---------------------------------------------------------------------------

@dataclass
class CodeTask:
    task_id: str
    description: str
    fn_name: str
    solution: str
    test_cases: list[tuple]   # list of ([args], expected)
    difficulty: Difficulty
    min_steps: int = 1


_TASKS: list[CodeTask] = [
    # --- EASY ---
    CodeTask(
        task_id="cg_easy_1",
        description="Write a function `sum_list(lst)` that returns the sum of all numbers in lst.",
        fn_name="sum_list",
        solution="def sum_list(lst):\n    return sum(lst)",
        test_cases=[
            ([1, 2, 3], 6),
            ([0], 0),
            ([-1, 1], 0),
            ([10, 20, 30], 60),
        ],
        difficulty=Difficulty.EASY,
    ),
    CodeTask(
        task_id="cg_easy_2",
        description="Write a function `reverse_string(s)` that returns the string s reversed.",
        fn_name="reverse_string",
        solution="def reverse_string(s):\n    return s[::-1]",
        test_cases=[
            (["hello"], "olleh"),
            ([""], ""),
            (["a"], "a"),
            (["abcd"], "dcba"),
        ],
        difficulty=Difficulty.EASY,
    ),
    CodeTask(
        task_id="cg_easy_3",
        description="Write a function `count_vowels(s)` that counts the number of vowels (a,e,i,o,u) in string s (case-insensitive).",
        fn_name="count_vowels",
        solution="def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')",
        test_cases=[
            (["hello"], 2),
            (["AEIOU"], 5),
            (["xyz"], 0),
            (["Python"], 1),
        ],
        difficulty=Difficulty.EASY,
    ),
    CodeTask(
        task_id="cg_easy_4",
        description="Write a function `is_palindrome(s)` that returns True if s is a palindrome, False otherwise (ignore case).",
        fn_name="is_palindrome",
        solution="def is_palindrome(s):\n    s = s.lower()\n    return s == s[::-1]",
        test_cases=[
            (["racecar"], True),
            (["hello"], False),
            (["Madam"], True),
            ([""], True),
        ],
        difficulty=Difficulty.EASY,
    ),
    # --- MEDIUM ---
    CodeTask(
        task_id="cg_medium_1",
        description="Write a function `flatten(nested)` that flattens a list of lists one level deep.",
        fn_name="flatten",
        solution="def flatten(nested):\n    return [item for sublist in nested for item in sublist]",
        test_cases=[
            ([[1, 2], [3, 4]], [1, 2, 3, 4]),
            ([[], [1]], [1]),
            ([[1]], [1]),
            ([[], []], []),
        ],
        difficulty=Difficulty.MEDIUM,
    ),
    CodeTask(
        task_id="cg_medium_2",
        description="Write a function `merge_sorted(a, b)` that merges two sorted lists into one sorted list.",
        fn_name="merge_sorted",
        solution=textwrap.dedent("""\
            def merge_sorted(a, b):
                result, i, j = [], 0, 0
                while i < len(a) and j < len(b):
                    if a[i] <= b[j]:
                        result.append(a[i]); i += 1
                    else:
                        result.append(b[j]); j += 1
                return result + a[i:] + b[j:]
        """),
        test_cases=[
            ([[1, 3, 5], [2, 4, 6]], [1, 2, 3, 4, 5, 6]),
            ([[], [1, 2]], [1, 2]),
            ([[1], []], [1]),
            ([[1, 2], [1, 2]], [1, 1, 2, 2]),
        ],
        difficulty=Difficulty.MEDIUM,
    ),
    CodeTask(
        task_id="cg_medium_3",
        description="Write a function `group_by(items, key_fn)` that groups items in a list by the result of key_fn, returning a dict.",
        fn_name="group_by",
        solution=textwrap.dedent("""\
            def group_by(items, key_fn):
                result = {}
                for item in items:
                    k = key_fn(item)
                    result.setdefault(k, []).append(item)
                return result
        """),
        test_cases=[
            ([[1, 2, 3, 4], lambda x: x % 2], {0: [2, 4], 1: [1, 3]}),
            ([["a", "bb", "ccc"], len], {1: ["a"], 2: ["bb"], 3: ["ccc"]}),
        ],
        difficulty=Difficulty.MEDIUM,
    ),
    # --- HARD ---
    CodeTask(
        task_id="cg_hard_1",
        description=(
            "Write a function `decode_run_length(s)` that decodes a run-length encoded string. "
            "For example '3a2b1c' → 'aaabbc'."
        ),
        fn_name="decode_run_length",
        solution=textwrap.dedent("""\
            def decode_run_length(s):
                result, i = [], 0
                while i < len(s):
                    n = ''
                    while i < len(s) and s[i].isdigit():
                        n += s[i]; i += 1
                    if i < len(s):
                        result.append(s[i] * int(n or '1')); i += 1
                return ''.join(result)
        """),
        test_cases=[
            (["3a2b1c"], "aaabbc"),
            (["1a1b1c"], "abc"),
            (["4x"], "xxxx"),
            (["2a3b2c"], "aabbbcc"),
        ],
        difficulty=Difficulty.HARD,
    ),
    CodeTask(
        task_id="cg_hard_2",
        description=(
            "Write a function `topological_sort(graph)` that takes an adjacency dict "
            "(node → list of neighbours) and returns a valid topological ordering as a list. "
            "Raise ValueError if a cycle is detected."
        ),
        fn_name="topological_sort",
        solution=textwrap.dedent("""\
            def topological_sort(graph):
                visited, stack, in_stack = set(), [], set()
                def dfs(node):
                    if node in in_stack:
                        raise ValueError('cycle')
                    if node in visited:
                        return
                    in_stack.add(node)
                    for nb in graph.get(node, []):
                        dfs(nb)
                    in_stack.remove(node)
                    visited.add(node)
                    stack.append(node)
                for n in list(graph):
                    dfs(n)
                return stack[::-1]
        """),
        test_cases=[
            (["A", "B", "C", "D"], None),  # checked differently via membership
        ],
        difficulty=Difficulty.HARD,
    ),
    CodeTask(
        task_id="cg_hard_3",
        description=(
            "Write a function `count_change(amount, coins)` that returns the number of ways "
            "to make change for amount using the given coin denominations."
        ),
        fn_name="count_change",
        solution=textwrap.dedent("""\
            def count_change(amount, coins):
                dp = [0] * (amount + 1)
                dp[0] = 1
                for c in coins:
                    for i in range(c, amount + 1):
                        dp[i] += dp[i - c]
                return dp[amount]
        """),
        test_cases=[
            ([4, [1, 2, 3]], 4),
            ([0, [1, 2]], 1),
            ([10, [2, 5, 3, 6]], 5),
            ([3, [2]], 0),
        ],
        difficulty=Difficulty.HARD,
    ),
]

_BY_DIFFICULTY: dict[Difficulty, list[CodeTask]] = {d: [] for d in Difficulty}
for _t in _TASKS:
    _BY_DIFFICULTY[_t.difficulty].append(_t)


# ---------------------------------------------------------------------------
# CodeGenBlock
# ---------------------------------------------------------------------------

class CodeGenBlock(EnvBlock):
    name = "code_gen"
    max_steps = 5

    def __init__(self):
        pass

    def reset(self, difficulty: str, rng: random.Random) -> tuple[str, dict]:
        pool = _BY_DIFFICULTY.get(difficulty) or _BY_DIFFICULTY[Difficulty.EASY]
        task = rng.choice(pool)

        metadata = {
            "task_id": task.task_id,
            "task_description": task.description,
            "fn_name": task.fn_name,
            "test_cases": task.test_cases,
            "solution": task.solution,
            "min_steps": task.min_steps,
            "attempt_count": 0,
            "action_history": [],
            "solved": False,
        }

        prompt = (
            f"{task.description}\n\n"
            f"Respond with your Python solution in a code block:\n"
            f"```python\n# your solution here\n```"
        )
        return prompt, metadata

    def step(self, action_text: str, state: State) -> tuple[dict, bool]:
        metadata = state.working_memory
        metadata["attempt_count"] += 1
        metadata["action_history"].append(action_text)

        # Reward is now calculated by CodeExecutionRubric in rubrics.py
        # But we return an empty dict or some basic feedback here
        comp_dict = {"step": metadata["attempt_count"]}
        
        # Simple completion check for block routing
        solved = "solved" in action_text.lower() # Placeholder, rubric handles real scoring
        done = solved or metadata["attempt_count"] >= self.max_steps

        return comp_dict, done

    def is_done(self, metadata: dict) -> bool:
        return metadata.get("solved", False) or metadata.get("attempt_count", 0) >= self.max_steps
