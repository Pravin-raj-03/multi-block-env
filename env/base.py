"""
Core data structures and abstract base class for all env blocks.

Conforms to OpenEnv 0.1.13+ interfaces.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action as BaseAction
from openenv.core.env_server.types import Observation as BaseObservation
from openenv.core.env_server.types import State as BaseState
from pydantic import Field


# ---------------------------------------------------------------------------
# Action / Observation / State — Pydantic models matching OpenEnv core
# ---------------------------------------------------------------------------

class Action(BaseAction):
    """What the agent emits each turn — plain text from the LLM."""
    text: str = Field(..., description="Natural language output from the LLM")


class State(BaseState):
    """Internal environment state."""
    cumulative_reward: float = 0.0
    context_window: list[dict] = Field(default_factory=list)
    working_memory: dict[str, Any] = Field(default_factory=dict)
    active_block: str = "default"


class Observation(BaseObservation):
    """Everything the agent sees on each step."""
    task_id: str = Field(..., description="Unique ID for the current subtask")
    task_description: str = Field(..., description="Natural language prompt for the agent")
    history: list[dict] = Field(default_factory=list)
    step: int = 0
    difficulty: str = "easy"
    reward_feedback: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# EnvBlock — Logic component for a specific task type
# ---------------------------------------------------------------------------

class EnvBlock(ABC):
    """
    Abstract base for a single task block logic.
    Stateless singleton: all episode state lives in the `state` object.
    """
    name: str = "base"
    max_steps: int = 10

    @abstractmethod
    def reset(self, difficulty: str, rng: random.Random) -> tuple[str, dict]:
        """Return (task_description, metadata_to_merge_into_working_memory)."""
        pass

    @abstractmethod
    def step(self, action_text: str, state: State) -> tuple[dict[str, float], bool]:
        """Process action, return (reward_components, done)."""
        pass

