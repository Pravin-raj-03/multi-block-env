"""
Long-horizon multi-agent environment.
Orchestrates multiple task blocks (Planner/Workers).

Conforms to OpenEnv 0.1.13+ interfaces.
"""

from __future__ import annotations

import random
import uuid
import re
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment
from .base import Action, Observation, State, EnvBlock
from .blocks.task_split import TaskSplittingBlock
from .blocks.code_gen import CodeGenBlock
from .blocks.reasoning import ReasoningBlock
from .rubrics import MultiBlockRubric


class LongHorizonEnv(Environment[Action, Observation, State]):
    """
    Multi-block long-horizon RL environment.
    
    Session management is now handled by the OpenEnv HttpServer, 
    so this class focuses on a single episode's logic.
    """
    
    # Enable concurrent session support in the framework
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        # Initialize with the multi-block rubric for automatic reward computation
        super().__init__(rubric=MultiBlockRubric())
        
        self.blocks: dict[str, EnvBlock] = {
            "task_split": TaskSplittingBlock(),
            "code_gen": CodeGenBlock(),
            "reasoning": ReasoningBlock(),
        }
        self._current_state: Optional[State] = None

    @property
    def state(self) -> State:
        if self._current_state is None:
            raise RuntimeError("Environment not reset")
        return self._current_state

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        block_name: str = "task_split",
        custom_task_description: str | None = None,
        custom_answer: str | None = None,
        **kwargs: Any
    ) -> Observation:
        """Start a fresh episode."""
        self._reset_rubric()
        
        rng = random.Random(seed)
        difficulty = kwargs.get("difficulty", "medium")
        
        block = self.blocks.get(block_name, self.blocks["task_split"])
        task_desc, metadata = block.reset(difficulty, rng)
        
        # Override with custom prompt if provided (e.g., from an external dataset)
        if custom_task_description:
            task_desc = custom_task_description
            metadata["task_description"] = custom_task_description
        if custom_answer:
            metadata["answer"] = custom_answer

        self._current_state = State(
            episode_id=episode_id or str(uuid.uuid4()),
            active_block=block_name,
            working_memory=metadata,
            context_window=[{"role": "system", "content": "You are an agent in a multi-step environment."}]
        )

        obs = Observation(
            task_id=metadata.get("task_id", block_name),
            task_description=task_desc,
            history=[],
            step=0,
            difficulty=difficulty
        )
        # Pass metadata to observation so the Rubric can access it
        obs.metadata = {
            "working_memory": self._current_state.working_memory, 
            "active_block": self._current_state.active_block
        }
        return obs

    def step(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        """Execute one action."""
        if self._current_state is None:
            raise RuntimeError("Call reset() before step()")

        current_block = self.blocks.get(self._current_state.active_block, self.blocks["task_split"])
        
        # 1. Update state history
        self._current_state.context_window.append({"role": "user", "content": action.text})
        self._current_state.step_count += 1
        
        # 2. Execute block logic
        reward_components, block_done = current_block.step(action.text, self._current_state)
        
        # 3. Handle dynamic routing
        self._handle_routing(action.text)
        
        # 4. Create observation
        obs = Observation(
            task_id=self._current_state.working_memory.get("task_id", self._current_state.active_block),
            task_description=self._current_state.working_memory.get("task_description", ""),
            history=self._current_state.context_window,
            step=self._current_state.step_count,
            done=block_done or self._current_state.step_count >= 20
        )
        obs.metadata = {
            "working_memory": self._current_state.working_memory, 
            "active_block": self._current_state.active_block
        }
        
        # 5. Apply Rubric (Automatic reward computation)
        obs.reward = self._apply_rubric(action, obs)
        self._current_state.cumulative_reward += obs.reward or 0.0
        
        obs.reward_feedback = reward_components 
        return obs

    def _handle_routing(self, text: str):
        """Regex-based routing for multi-agent orchestration."""
        if "[CODE_GEN]" in text:
            self._current_state.active_block = "code_gen"
        elif "[REASONING]" in text:
            self._current_state.active_block = "reasoning"
        elif "[COMPLETE]" in text:
            pass # Block logic handles done
