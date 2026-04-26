"""TaskSplittingBlock — trains the agent to decompose complex problems into ordered subtasks."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from env.base import Difficulty, EnvBlock
from env.rewards.multi_reward import SplitQualityScorer


# ---------------------------------------------------------------------------
# Problem bank
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    task_id: str
    description: str
    reference_split: list[str]
    min_tasks: int
    max_tasks: int
    difficulty: Difficulty


_PROBLEMS: list[Problem] = [
    # --- EASY ---
    Problem(
        task_id="ts_easy_1",
        description=(
            "Build a script that reads a CSV file and prints the column "
            "with the highest average value."
        ),
        reference_split=[
            "Open and read the CSV file using a file reader",
            "Parse the CSV header row to identify column names",
            "Iterate over data rows and accumulate numeric values per column",
            "Calculate the average for each column",
            "Identify and print the column name with the highest average",
        ],
        min_tasks=3, max_tasks=6, difficulty=Difficulty.EASY,
    ),
    Problem(
        task_id="ts_easy_2",
        description=(
            "Write a program that counts word frequencies in a text file "
            "and displays the top 10 most common words."
        ),
        reference_split=[
            "Open and read the text file",
            "Tokenize the text into individual words",
            "Normalize words to lowercase and strip punctuation",
            "Count the frequency of each word using a dictionary",
            "Sort words by frequency in descending order",
            "Print the top 10 words with their counts",
        ],
        min_tasks=3, max_tasks=7, difficulty=Difficulty.EASY,
    ),
    Problem(
        task_id="ts_easy_3",
        description="Create a function that validates whether a string is a valid email address.",
        reference_split=[
            "Define a function that accepts a string parameter",
            "Check that the string contains exactly one @ symbol",
            "Verify the local part before @ is non-empty and has no spaces",
            "Verify the domain part after @ contains at least one dot",
            "Return True if all checks pass, False otherwise",
        ],
        min_tasks=3, max_tasks=6, difficulty=Difficulty.EASY,
    ),
    # --- MEDIUM ---
    Problem(
        task_id="ts_medium_1",
        description=(
            "Build a REST API endpoint that accepts a user registration form, "
            "validates the inputs, hashes the password, and stores the record."
        ),
        reference_split=[
            "Define the POST /register route and request schema",
            "Extract and validate required fields: name, email, password",
            "Check that the email is not already registered in the database",
            "Validate password strength requirements",
            "Hash the password using a secure algorithm like bcrypt",
            "Create a new user record with hashed password",
            "Save the record to the database",
            "Return a success response with the user id",
        ],
        min_tasks=5, max_tasks=9, difficulty=Difficulty.MEDIUM,
    ),
    Problem(
        task_id="ts_medium_2",
        description=(
            "Write a data pipeline that fetches JSON data from a URL, "
            "filters records by a date range, and writes the results to a new file."
        ),
        reference_split=[
            "Send an HTTP GET request to the specified URL",
            "Parse the JSON response body into a list of records",
            "Define the start and end date range for filtering",
            "Parse the date field in each record to a comparable date object",
            "Filter records where the date falls within the specified range",
            "Serialize the filtered records back to JSON",
            "Write the JSON output to a new file",
        ],
        min_tasks=5, max_tasks=8, difficulty=Difficulty.MEDIUM,
    ),
    Problem(
        task_id="ts_medium_3",
        description=(
            "Implement a simple in-memory cache with TTL expiry and max-size eviction."
        ),
        reference_split=[
            "Define the Cache class with capacity and default TTL parameters",
            "Store entries as key → (value, expiry_timestamp) in a dictionary",
            "Implement the get method that checks expiry before returning a value",
            "Implement the set method that adds or updates an entry with a new expiry",
            "On set, if capacity is exceeded, evict the oldest entry",
            "Implement a cleanup method that removes all expired entries",
        ],
        min_tasks=4, max_tasks=7, difficulty=Difficulty.MEDIUM,
    ),
    # --- HARD ---
    Problem(
        task_id="ts_hard_1",
        description=(
            "Design and implement a task queue system where workers pull jobs, "
            "process them, and report results with retry on failure."
        ),
        reference_split=[
            "Define the Job dataclass with id, payload, status, and retry_count",
            "Create the Queue class backed by a thread-safe deque",
            "Implement enqueue to add jobs and dequeue to pop the next pending job",
            "Define the Worker class that loops: dequeue → process → report",
            "Implement the process method that calls the job handler function",
            "On failure, increment retry_count and re-enqueue if below max retries",
            "On max retries exceeded, mark the job as failed and log the error",
            "Implement a results store that maps job id to outcome",
            "Add a shutdown mechanism to stop workers gracefully",
        ],
        min_tasks=7, max_tasks=12, difficulty=Difficulty.HARD,
    ),
    Problem(
        task_id="ts_hard_2",
        description=(
            "Build a multi-step data transformation pipeline: "
            "parse CSV → validate schema → normalize values → "
            "deduplicate → aggregate by category → export as JSON."
        ),
        reference_split=[
            "Read the CSV file and parse each row into a dictionary",
            "Define the expected schema with required fields and types",
            "Validate each row against the schema and collect errors",
            "Discard or fix rows that fail validation",
            "Normalize numeric fields to a common scale or unit",
            "Identify duplicate rows using a composite key",
            "Remove duplicate rows keeping the most recent entry",
            "Group rows by the category field",
            "Compute aggregate statistics per category group",
            "Serialize the aggregated results to JSON and write to output file",
        ],
        min_tasks=7, max_tasks=12, difficulty=Difficulty.HARD,
    ),
    Problem(
        task_id="ts_hard_3",
        description=(
            "Create an agent loop: fetch a coding problem, generate a solution, "
            "run tests, parse test failures, fix the solution, and retry up to 3 times."
        ),
        reference_split=[
            "Fetch the problem statement from the problem source",
            "Parse the problem to extract the function signature and test cases",
            "Generate an initial solution using the language model",
            "Run the solution against the test cases in a sandbox",
            "Parse the test output to identify which tests failed and why",
            "If all tests pass, return the solution as final",
            "If tests fail and retry count < 3, format failure details as feedback",
            "Re-prompt the model with the original problem plus failure feedback",
            "Increment retry counter and go back to the run-tests step",
            "If max retries exceeded, return the best partial solution with a failure report",
        ],
        min_tasks=7, max_tasks=12, difficulty=Difficulty.HARD,
    ),
]

_BY_DIFFICULTY: dict[Difficulty, list[Problem]] = {d: [] for d in Difficulty}
for _p in _PROBLEMS:
    _BY_DIFFICULTY[_p.difficulty].append(_p)


# ---------------------------------------------------------------------------
# TaskSplittingBlock
# ---------------------------------------------------------------------------

class TaskSplittingBlock(EnvBlock):
    name = "task_split"
    max_steps = 2  # step 1: initial split; step 2: optional revision

    def __init__(self):
        self._scorer = SplitQualityScorer()

    # -- reset ----------------------------------------------------------------

    def reset(self, difficulty: str, rng: random.Random) -> tuple[str, dict]:
        pool = _BY_DIFFICULTY.get(difficulty) or _BY_DIFFICULTY[Difficulty.EASY]
        problem = rng.choice(pool)

        run_subtasks = difficulty == "hard"

        metadata = {
            "task_id": problem.task_id,
            "task_description": problem.description,
            "reference_split": problem.reference_split,
            "min_tasks": problem.min_tasks,
            "max_tasks": problem.max_tasks,
            "difficulty": difficulty,
            "attempt_count": 0,
            "step1_score": None,   # filled after step 1
            "step1_components": None,
            "run_subtasks": run_subtasks,
        }

        prompt = (
            f"Decompose the following task into ordered, atomic subtasks.\n\n"
            f"Task: {problem.description}\n\n"
            f"Format your answer as:\n"
            f"Task 1: <concrete action>\n"
            f"Task 2: <concrete action>\n"
            f"...\n\n"
            f"Guidelines:\n"
            f"- Each subtask must be a single concrete action (≥ 5 words)\n"
            f"- No subtask should combine two actions with 'and then'\n"
            f"- Order subtasks so dependencies come first\n"
            f"- Aim for {problem.min_tasks}–{problem.max_tasks} subtasks\n"
        )
        if run_subtasks:
            prompt += (
                "\nFor subtasks you want executed as code, prefix the line with [CODE].\n"
                "Example: Task 3: [CODE] Write the function that filters records by date\n"
            )
        return prompt, metadata

    # -- step -----------------------------------------------------------------

    def step(self, action_text: str, state: State) -> tuple[dict, bool]:
        metadata = state.working_memory
        metadata["attempt_count"] += 1
        step_num = metadata["attempt_count"]

        tasks = self._parse_tasks(action_text)
        components = self._scorer.score(
            agent_tasks=tasks,
            reference_split=metadata["reference_split"],
            problem_description=metadata["task_description"],
            min_tasks=metadata["min_tasks"],
            max_tasks=metadata["max_tasks"],
        )
        comp_dict = components.to_dict()
        score = components.total()

        if step_num == 1:
            metadata["step1_score"] = score
            metadata["step1_components"] = comp_dict
            done = False  # allow step 2 revision

        else:
            # Take the BEST of step 1 and step 2 — no bonus for sandbagging step 1.
            # This removes the exploit where the model deliberately gives a bad step 1
            # to maximize the revision delta bonus.
            step1 = metadata.get("step1_score", 0.0)
            score = max(step1, score)
            comp_dict["step1_score"] = round(step1, 4)
            done = True

        return comp_dict, done

    def is_done(self, metadata: dict) -> bool:
        return metadata.get("attempt_count", 0) >= self.max_steps

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _parse_tasks(text: str) -> list[str]:
        lines = text.strip().splitlines()
        tasks = []
        for line in lines:
            m = re.match(r'^\s*(?:Task\s+)?\d+[\.\):\-]\s*(.+)', line, re.IGNORECASE)
            if m:
                tasks.append(m.group(1).strip())
        return tasks
