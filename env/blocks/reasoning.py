"""ReasoningBlock — agent must produce a numbered step chain ending with 'Final Answer:'."""

from __future__ import annotations

import random
from dataclasses import dataclass

from env.base import Difficulty, EnvBlock
from env.rewards.multi_reward import MultiRewardCalculator


# ---------------------------------------------------------------------------
# Task bank
# ---------------------------------------------------------------------------

@dataclass
class ReasoningTask:
    task_id: str
    description: str
    answer: str        # expected final answer (string or number)
    min_steps: int
    difficulty: Difficulty


_TASKS: list[ReasoningTask] = [
    # --- EASY ---
    ReasoningTask(
        task_id="rn_easy_1",
        description=(
            "Alice has 3 apples. Bob gives her 5 more apples, "
            "then she eats 2 apples. How many apples does Alice have now?"
        ),
        answer="6",
        min_steps=2,
        difficulty=Difficulty.EASY,
    ),
    ReasoningTask(
        task_id="rn_easy_2",
        description=(
            "A train travels at 60 mph for 2 hours, then at 90 mph for 1 hour. "
            "What is the total distance travelled in miles?"
        ),
        answer="210",
        min_steps=2,
        difficulty=Difficulty.EASY,
    ),
    ReasoningTask(
        task_id="rn_easy_3",
        description=(
            "If 8 workers can build a wall in 6 days, "
            "how many days will it take 12 workers to build the same wall?"
        ),
        answer="4",
        min_steps=2,
        difficulty=Difficulty.EASY,
    ),
    ReasoningTask(
        task_id="rn_easy_4",
        description=(
            "A rectangle has a width of 5 cm and a length of 12 cm. "
            "What is its area in square centimetres?"
        ),
        answer="60",
        min_steps=2,
        difficulty=Difficulty.EASY,
    ),
    # --- MEDIUM ---
    ReasoningTask(
        task_id="rn_medium_1",
        description=(
            "A bacteria colony starts at 100 cells at 8 AM and doubles every 3 hours. "
            "How many cells are there at 5 PM?"
        ),
        answer="3200",
        min_steps=3,
        difficulty=Difficulty.MEDIUM,
    ),
    ReasoningTask(
        task_id="rn_medium_2",
        description=(
            "A store marks up a $40 item by 25%, then applies a 10% discount. "
            "What is the final price in dollars?"
        ),
        answer="45",
        min_steps=3,
        difficulty=Difficulty.MEDIUM,
    ),
    ReasoningTask(
        task_id="rn_medium_3",
        description=(
            "Pipe A fills a tank in 4 hours, Pipe B fills it in 6 hours, "
            "and Pipe C drains it in 12 hours. "
            "If all three are open simultaneously, how many hours to fill the tank?"
        ),
        answer="4",
        min_steps=3,
        difficulty=Difficulty.MEDIUM,
    ),
    ReasoningTask(
        task_id="rn_medium_4",
        description=(
            "A car depreciates by 15% each year. If the car costs $20,000 new, "
            "what is its value after 3 years? Round to the nearest dollar."
        ),
        answer="12283",
        min_steps=3,
        difficulty=Difficulty.MEDIUM,
    ),
    # --- HARD ---
    ReasoningTask(
        task_id="rn_hard_1",
        description=(
            "Find the 10th term of the sequence: 2, 6, 12, 20, 30. "
            "Show the pattern and derive the general formula."
        ),
        answer="110",
        min_steps=4,
        difficulty=Difficulty.HARD,
    ),
    ReasoningTask(
        task_id="rn_hard_2",
        description=(
            "All bloops are razzles. All razzles are lazzles. "
            "Are all bloops lazzles? Show the full deductive chain and state the conclusion."
        ),
        answer="yes",
        min_steps=3,
        difficulty=Difficulty.HARD,
    ),
    ReasoningTask(
        task_id="rn_hard_3",
        description=(
            "A snail climbs a 30-foot pole. Each day it climbs 5 feet, "
            "and each night it slides back 3 feet. "
            "On which day does it first reach the top?"
        ),
        answer="14",
        min_steps=4,
        difficulty=Difficulty.HARD,
    ),
]

_BY_DIFFICULTY: dict[Difficulty, list[ReasoningTask]] = {d: [] for d in Difficulty}
for _t in _TASKS:
    _BY_DIFFICULTY[_t.difficulty].append(_t)


# ---------------------------------------------------------------------------
# ReasoningBlock
# ---------------------------------------------------------------------------

class ReasoningBlock(EnvBlock):
    name = "reasoning"
    max_steps = 3

    def __init__(self):
        super().__init__()
        self.state_space = (
            "Word problem or logical reasoning question requiring a step-by-step solution."
        )
        self.action_space = (
            "Numbered reasoning chain: 'Step N: <reasoning>\\n...\\nFinal Answer: <answer>'. "
            "Must end with a 'Final Answer:' line."
        )
        self.episode_max_length = self.max_steps
        self._calc = MultiRewardCalculator()

    def reset(self, difficulty: Difficulty, rng: random.Random) -> tuple[str, dict]:
        pool = _BY_DIFFICULTY.get(difficulty) or _BY_DIFFICULTY[Difficulty.EASY]
        task = rng.choice(pool)

        metadata = {
            "task_id": task.task_id,
            "task_description": task.description,
            "answer": task.answer,
            "min_steps": task.min_steps,
            "attempt_count": 0,
            "action_history": [],
            "solved": False,
        }

        prompt = (
            f"{task.description}\n\n"
            f"Show your work as numbered steps, then state your answer:\n"
            f"Step 1: ...\n"
            f"Step 2: ...\n"
            f"Final Answer: <your answer>"
        )
        return prompt, metadata

    def step(self, action_text: str, metadata: dict) -> tuple[dict, bool]:
        metadata["attempt_count"] += 1
        metadata["action_history"].append(action_text)

        components = self._calc.compute_reasoning(
            action=action_text,
            task_meta=metadata,
            action_history=metadata["action_history"],
            step_count=metadata["attempt_count"],
        )
        comp_dict = components.to_dict()

        solved = components.correctness >= 1.0
        metadata["solved"] = solved
        done = solved or metadata["attempt_count"] >= self.max_steps

        return comp_dict, done

    def is_done(self, metadata: dict) -> bool:
        return metadata.get("solved", False) or metadata.get("attempt_count", 0) >= self.max_steps
