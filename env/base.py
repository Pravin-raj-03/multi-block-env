"""
Core data structures and abstract base class for all env blocks.

Conforms to openenv.Env interface:
  - state_space  : description of what the agent observes
  - action_space : description of valid actions
  - reset()      : start a fresh episode, return initial observation
  - step(action) : return (observation, reward, done, info)
"""

from __future__ import annotations

import random
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from openenv.env import Env
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Action / Observation / State — first-class dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Action:
    """What the agent emits each turn — plain text from the LLM."""
    text: str
    episode_id: str = ""
    agent_id: str = "agent_0"


@dataclass
class HistoryEntry:
    role: str               # "env" | "agent"
    content: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp}


@dataclass
class StateBuffer:
    """Rolling context window passed to the model on each forward pass."""
    episode_step: int = 0
    cumulative_reward: float = 0.0
    context_window: list = field(default_factory=list)   # [{step, role, content, ...}]
    working_memory: dict = field(default_factory=dict)   # writable scratchpad
    state_hash: str = ""


@dataclass
class Observation:
    """
    Everything the agent sees on each step.

    Designed to be JSON-serializable for the FastAPI client-server interface.
    """
    task_id: str
    task_description: str
    history: list = field(default_factory=list)          # list[HistoryEntry.to_dict()]
    state_buffer: StateBuffer = field(default_factory=StateBuffer)
    step: int = 0
    episode_id: str = ""
    difficulty: str = "easy"
    reward_feedback: dict = field(default_factory=dict)  # last step's reward_components
    run_subtasks: bool = False                            # hierarchical mode flag
    active_agent: str = "any"                             # tracks whose turn it is

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "history": self.history,
            "state_buffer": {
                "episode_step": self.state_buffer.episode_step,
                "cumulative_reward": self.state_buffer.cumulative_reward,
                "context_window": self.state_buffer.context_window,
                "working_memory": self.state_buffer.working_memory,
                "state_hash": self.state_buffer.state_hash,
            },
            "step": self.step,
            "episode_id": self.episode_id,
            "difficulty": self.difficulty,
            "reward_feedback": self.reward_feedback,
            "run_subtasks": self.run_subtasks,
            "active_agent": self.active_agent,
        }


@dataclass
class StepResult:
    """Full result of one env.step() call — mirrors (obs, reward, done, info)."""
    observation: Observation
    reward: float
    reward_components: dict         # individual reward signals
    done: bool
    truncated: bool = False
    info: dict = field(default_factory=dict)

    def unpack(self) -> tuple:
        """Return (observation, reward, done, info) — the standard OpenEnv tuple."""
        return self.observation, self.reward, self.done, {
            **self.info,
            "reward_components": self.reward_components,
            "truncated": self.truncated,
        }


# ---------------------------------------------------------------------------
# Difficulty enum
# ---------------------------------------------------------------------------

class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ---------------------------------------------------------------------------
# EnvBlock — stateless singleton, subclasses openenv.Env
# ---------------------------------------------------------------------------

# Space descriptors — human-readable strings used as state_space / action_space.
# Following the OpenEnv convention of passing these to Env.__init__.

_OBSERVATION_SPACE = (
    "JSON object with fields: task_id (str), task_description (str), "
    "history (list of {role, content} dicts), state_buffer (rolling 2048-token "
    "context window), step (int), difficulty (str), reward_feedback (dict)."
)

_ACTION_SPACE = (
    "Natural language text. Format depends on task type: "
    "(1) TaskSplit — 'Task N: <description>' numbered list; "
    "(2) CodeGen — Python code inside ```python ... ``` fences; "
    "(3) Reasoning — numbered 'Step N: ...' chain ending with 'Final Answer: <answer>'."
)


class EnvBlock(Env):
    """
    Abstract base for a single task block.

    Stateless singleton: all episode state lives in the `metadata` dict
    passed per call — this makes concurrent episodes safe.

    Interface (standard OpenEnv):
        reset(difficulty, rng) -> (task_description: str, metadata: dict)
        step(action_text, metadata) -> (reward_components: dict, done: bool)

    The LongHorizonEnv wrapper converts these to the full StepResult /
    (obs, reward, done, info) tuple.
    """

    # Subclasses set these as class attributes
    name: str = "base"
    max_steps: int = 10

    def __init__(self):
        super().__init__(
            name=self.name,
            state_space=_OBSERVATION_SPACE,
            action_space=_ACTION_SPACE,
            episode_max_length=self.max_steps,
        )

    @abstractmethod
    def reset(self, difficulty: Difficulty, rng: random.Random) -> tuple[str, dict]:
        """Return (task_description, metadata). metadata is mutable episode state."""

    @abstractmethod
    def step(self, action_text: str, metadata: dict) -> tuple[dict, bool]:
        """
        Process one action.
        Returns (reward_components: dict[str, float], done: bool).
        Mutates metadata in-place.
        """

    @abstractmethod
    def is_done(self, metadata: dict) -> bool:
        """Terminal condition check from block-private state."""

    def get_task_id(self, metadata: dict) -> str:
        return metadata.get("task_id", self.name)

    def get_hint(self, metadata: dict, step: int) -> str | None:
        return None
