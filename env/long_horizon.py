"""
LongHorizonEnv — top-level environment subclassing openenv.Env.

Conforms to the OpenEnv standard interface:
    env.reset()              -> Observation
    env.step(action)         -> (Observation, reward, done, info)
    env.state_space          -> str (observation space descriptor)
    env.action_space         -> str (action space descriptor)
    env.episode_max_length   -> int

Adds on top:
    - Multi-episode registry (UUID-keyed, thread-safe)
    - Rolling 2048-token state buffer with pinned slot 0
    - Curriculum scheduler (EASY → MEDIUM → HARD)
    - Three env blocks: TaskSplit, CodeGen, Reasoning
    - Background stale-episode cleanup
"""

from __future__ import annotations

import json
import random
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from openenv.env import Env

from env.base import (
    Action,
    Difficulty,
    EnvBlock,
    HistoryEntry,
    Observation,
    StateBuffer,
    StepResult,
)
from env.rewards.multi_reward import StateIntegrityChecker


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EnvConfig:
    state_buffer_tokens: int = 2048
    stale_episode_seconds: float = 3600.0
    cleanup_interval_seconds: float = 300.0
    curriculum_weights: dict = field(default_factory=lambda: {
        Difficulty.EASY:   {"task_split": 0.5, "code_gen": 0.3, "reasoning": 0.2},
        Difficulty.MEDIUM: {"task_split": 0.4, "code_gen": 0.3, "reasoning": 0.3},
        Difficulty.HARD:   {"task_split": 0.3, "code_gen": 0.35, "reasoning": 0.35},
    })


# ---------------------------------------------------------------------------
# Rolling state buffer — pins first entry (step 0) so it is never evicted
# ---------------------------------------------------------------------------

class RollingStateBuffer:
    def __init__(self, max_tokens: int = 2048):
        self.max_tokens = max_tokens
        self._pinned: dict | None = None
        self._entries: deque[dict] = deque()
        self._token_count: int = 0

    def _estimate_tokens(self, entry: dict) -> int:
        return max(1, len(json.dumps(entry, default=str).split()) * 4 // 3)

    def push(self, entry: dict, pin: bool = False) -> None:
        if pin or self._pinned is None:
            self._pinned = entry
            return
        cost = self._estimate_tokens(entry)
        while self._token_count + cost > self.max_tokens and self._entries:
            old = self._entries.popleft()
            self._token_count -= self._estimate_tokens(old)
        self._entries.append(entry)
        self._token_count += cost

    def to_list(self) -> list[dict]:
        result = []
        if self._pinned is not None:
            result.append(self._pinned)
        result.extend(self._entries)
        return result

    def snapshot_hash(self) -> str:
        raw = json.dumps(self.to_list(), sort_keys=True, default=str)
        return sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Curriculum scheduler
# ---------------------------------------------------------------------------

class CurriculumScheduler:
    def __init__(self, alpha: float = 0.2, promote_threshold: float = 0.65, min_episodes: int = 10):
        self.current_difficulty = Difficulty.EASY
        self.episodes_completed = 0
        self.success_ema = 0.0
        self.alpha = alpha
        self.promote_threshold = promote_threshold
        self.min_episodes = min_episodes
        self._order = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]

    def record_episode(self, total_reward: float) -> bool:
        success = 1.0 if total_reward > 0.1 else 0.0
        self.success_ema = self.alpha * success + (1 - self.alpha) * self.success_ema
        self.episodes_completed += 1
        if self._should_promote():
            self._promote()
            return True
        return False

    def _should_promote(self) -> bool:
        if self.current_difficulty == Difficulty.HARD:
            return False
        return (
            self.episodes_completed >= self.min_episodes
            and self.success_ema >= self.promote_threshold
        )

    def _promote(self) -> None:
        idx = self._order.index(self.current_difficulty)
        if idx < len(self._order) - 1:
            self.current_difficulty = self._order[idx + 1]
            self.success_ema = 0.0
            self.episodes_completed = 0

    def stats(self) -> dict:
        return {
            "current_difficulty": self.current_difficulty.value,
            "episodes_completed": self.episodes_completed,
            "success_ema": round(self.success_ema, 4),
        }


# ---------------------------------------------------------------------------
# Episode state
# ---------------------------------------------------------------------------

@dataclass
class EpisodeState:
    episode_id: str
    block_name: str
    block: EnvBlock
    metadata: dict
    state_buffer: RollingStateBuffer
    history: list = field(default_factory=list)   # list[HistoryEntry]
    step: int = 0
    done: bool = False
    total_reward: float = 0.0
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    state_hash: str = ""
    difficulty: Difficulty = Difficulty.EASY
    active_agent: str = "any"
    roster: list = field(default_factory=lambda: ["agent_0"])
    task_queue: list[dict] = field(default_factory=list)  # queued subtasks for multi-agent routing


# ---------------------------------------------------------------------------
# Episode registry
# ---------------------------------------------------------------------------

class EpisodeRegistry:
    def __init__(self):
        self._episodes: dict[str, EpisodeState] = {}
        self._lock = threading.RLock()

    def add(self, episode: EpisodeState) -> None:
        with self._lock:
            self._episodes[episode.episode_id] = episode

    def get(self, episode_id: str) -> EpisodeState | None:
        with self._lock:
            return self._episodes.get(episode_id)

    def remove(self, episode_id: str) -> None:
        with self._lock:
            self._episodes.pop(episode_id, None)

    def cleanup_stale(self, max_age_seconds: float) -> int:
        now = time.time()
        with self._lock:
            stale = [eid for eid, ep in self._episodes.items()
                     if now - ep.last_active > max_age_seconds]
            for eid in stale:
                del self._episodes[eid]
        return len(stale)

    def active_count(self) -> int:
        with self._lock:
            return len(self._episodes)

    def all_episodes(self) -> list[EpisodeState]:
        with self._lock:
            return list(self._episodes.values())


# ---------------------------------------------------------------------------
# LongHorizonEnv — OpenEnv-compliant top-level environment
# ---------------------------------------------------------------------------

_STATE_SPACE = (
    "Structured JSON Observation with: task_id, task_description, history "
    "(last 20 {role, content} entries), state_buffer (rolling 2048-token context "
    "window with pinned first step), step, difficulty, reward_feedback."
)
_ACTION_SPACE = (
    "Natural language text string. Three formats accepted: "
    "(1) 'Task N: ...' list for task decomposition; "
    "(2) ```python ... ``` code block for code generation; "
    "(3) 'Step N: ... Final Answer: <answer>' chain for reasoning."
)


class LongHorizonEnv(Env):
    """
    Multi-block long-horizon RL environment for training Ollama/Llama.

    Subclasses openenv.Env and exposes the standard interface:
        reset([block_name])  -> (episode_id: str, observation: Observation)
        step(episode_id, action: str | Action) -> StepResult
        state(episode_id)    -> Observation
        step_standard(episode_id, action) -> (obs, reward, done, info)   # OpenEnv tuple

    The trainer uses step_standard() for its training loop.
    The FastAPI server uses reset()/step()/state() for HTTP routing.
    """

    def __init__(self, blocks: list[EnvBlock], config: EnvConfig | None = None):
        max_len = max((b.max_steps for b in blocks), default=10)
        super().__init__(
            name="LongHorizonEnv",
            state_space=_STATE_SPACE,
            action_space=_ACTION_SPACE,
            episode_max_length=max_len,
        )
        self.config = config or EnvConfig()
        self._blocks: dict[str, EnvBlock] = {b.name: b for b in blocks}
        self.registry = EpisodeRegistry()
        self.curriculum = CurriculumScheduler()
        self._integrity = StateIntegrityChecker()
        self._rng = random.Random()

        self._cleanup_thread = threading.Thread(
            target=self._cleanup_worker, daemon=True
        )
        self._cleanup_thread.start()

    # -- Standard OpenEnv interface ------------------------------------------

    def reset(self, block_name: str | None = None, custom_task_description: str | None = None, custom_answer: str | None = None) -> tuple[str, Observation]:
        """
        Start a fresh episode.

        Returns (episode_id, observation) — episode_id is the routing key
        for multi-episode concurrent use via the HTTP server.
        """
        difficulty = self.curriculum.current_difficulty
        block = self._select_block(block_name, difficulty)

        task_description, metadata = block.reset(difficulty, self._rng)
        
        # Override with custom prompt if provided (e.g., from an external dataset)
        if custom_task_description:
            task_description = custom_task_description
            metadata["task_description"] = custom_task_description
        
        # Override with custom expected answer if provided
        if custom_answer:
            metadata["answer"] = custom_answer
            
        episode_id = str(uuid.uuid4())

        buf = RollingStateBuffer(self.config.state_buffer_tokens)
        pinned_entry = {
            "step": 0,
            "role": "env",
            "content": task_description,
            "timestamp": time.time(),
        }
        buf.push(pinned_entry, pin=True)

        episode = EpisodeState(
            episode_id=episode_id,
            block_name=block.name,
            block=block,
            metadata=metadata,
            state_buffer=buf,
            difficulty=difficulty,
        )
        episode.state_hash = buf.snapshot_hash()
        self.registry.add(episode)

        obs = self._build_observation(episode, task_description)
        return episode_id, obs

    def step(self, episode_id: str, action: str | Action, agent_id: str | None = None) -> StepResult:
        """
        Apply one action to an episode.

        Accepts either raw text or an Action dataclass.
        Returns a StepResult; call result.unpack() for the standard
        (obs, reward, done, info) tuple.
        """
        action_text = action.text if isinstance(action, Action) else action
        if agent_id is None:
            agent_id = action.agent_id if isinstance(action, Action) else "agent_0"

        episode = self.registry.get(episode_id)
        if episode is None:
            raise KeyError(f"Episode {episode_id} not found")
        if episode.done:
            raise RuntimeError(f"Episode {episode_id} is already done")

        # Turn logic
        if episode.active_agent != "any" and agent_id != episode.active_agent:
            raise RuntimeError(f"Out of turn: It is {episode.active_agent}'s turn, not {agent_id}'s.")

        # State integrity check
        if episode.state_buffer.snapshot_hash() != episode.state_hash:
            raise RuntimeError("State buffer integrity violation detected")

        # Delegate to the active block
        reward_components, done = episode.block.step(action_text, episode.metadata)
        reward = self._aggregate_reward(reward_components)

        episode.step += 1
        episode.total_reward += reward
        episode.done = done or episode.block.is_done(episode.metadata)
        
        # --- Multi-Agent Orchestration: Intercept Task Split ---
        if episode.done and episode.block_name == "task_split" and episode.metadata.get("run_subtasks", True):
            from env.blocks.task_split import TaskSplittingBlock
            import re
            tasks = TaskSplittingBlock._parse_tasks(action_text)
            for t in tasks:
                # Dynamically check if the planner requested a specific block via [BLOCK_NAME]
                tag_match = re.match(r'^\[([A-Z0-9_]+)\]', t.upper().strip())
                if tag_match:
                    target_block = tag_match.group(1).lower()
                    desc = t[tag_match.end():].strip()
                else:
                    # Fallback block if the planner didn't specify
                    target_block = "reasoning"
                    desc = t
                
                # Verify the requested block actually exists in the environment
                if target_block in self._blocks:
                    episode.task_queue.append({"type": target_block, "desc": desc})
                else:
                    # If it hallucinated a block, fallback to reasoning
                    episode.task_queue.append({"type": "reasoning", "desc": t})
                    
            if episode.task_queue:
                episode.done = False  # Continue the episode into the subtasks
        
        # --- Multi-Agent Orchestration: Shift to Next Subtask ---
        if episode.done and episode.task_queue:
            episode.done = False
            next_task = episode.task_queue.pop(0)
            target_block = next_task["type"]
            
            if target_block in self._blocks:
                episode.block = self._blocks[target_block]
                episode.block_name = target_block
                
                # Re-initialize metadata for the new block but keep the subtask description
                prompt, new_meta = episode.block.reset(episode.difficulty, self._rng)
                new_meta["task_description"] = f"SUBTASK: {next_task['desc']}"
                episode.metadata = new_meta
                
                # Push the subtask prompt to the buffer
                episode.state_buffer.push({
                    "step": episode.step,
                    "role": "env",
                    "content": f"New Subtask Assigned ({target_block}):\n{new_meta['task_description']}\n\n{prompt}",
                    "timestamp": time.time(),
                })
        episode.last_active = time.time()

        # Safety truncation
        truncated = False
        if episode.step >= episode.block.max_steps and not episode.done:
            episode.done = True
            truncated = True

        # Update rolling state buffer
        episode.state_buffer.push({
            "step": episode.step,
            "role": agent_id,
            "content": action_text[:500],
            "reward": round(reward, 4),
            "timestamp": time.time(),
        })
        episode.state_buffer.push({
            "step": episode.step,
            "role": "env",
            "content": f"Reward: {reward:.3f} | {reward_components}",
            "timestamp": time.time(),
        })

        # History
        episode.history.append(HistoryEntry(role=agent_id, content=action_text))
        episode.history.append(HistoryEntry(
            role="env",
            content=f"Step {episode.step} reward: {reward:.3f}"
        ))

        episode.state_hash = episode.state_buffer.snapshot_hash()

        if episode.done:
            self.curriculum.record_episode(episode.total_reward)

        obs = self._build_observation(
            episode,
            episode.metadata.get("task_description", ""),
            reward_feedback=reward_components,
        )
        return StepResult(
            observation=obs,
            reward=reward,
            reward_components=reward_components,
            done=episode.done,
            truncated=truncated,
            info={
                "total_reward": round(episode.total_reward, 4),
                "block": episode.block_name,
                "difficulty": episode.difficulty.value,
            },
        )

    def step_standard(
        self, episode_id: str, action: str | Action, agent_id: str | None = None
    ) -> tuple[Observation, float, bool, dict]:
        """
        Standard OpenEnv / Gym-style step.

        Returns (observation, reward, done, info) — the same tuple interface
        used by the trainer's collection loop.
        """
        result = self.step(episode_id, action, agent_id=agent_id)
        return result.unpack()

    def state(self, episode_id: str) -> Observation:
        episode = self.registry.get(episode_id)
        if episode is None:
            raise KeyError(f"Episode {episode_id} not found")
        return self._build_observation(
            episode, episode.metadata.get("task_description", "")
        )

    def metrics(self) -> dict[str, Any]:
        return {
            "curriculum": self.curriculum.stats(),
            "active_episodes": self.registry.active_count(),
            "blocks": list(self._blocks.keys()),
            "state_space": self.state_space,
            "action_space": self.action_space,
            "episode_max_length": self.episode_max_length,
        }

    # -- Internals -----------------------------------------------------------

    def _select_block(self, block_name: str | None, difficulty: Difficulty) -> EnvBlock:
        if block_name and block_name in self._blocks:
            return self._blocks[block_name]
        weights_map = self.config.curriculum_weights.get(difficulty, {})
        available = {n: w for n, w in weights_map.items() if n in self._blocks}
        if not available:
            return next(iter(self._blocks.values()))
        names = list(available.keys())
        weights = [available[n] for n in names]
        return self._blocks[self._rng.choices(names, weights=weights, k=1)[0]]

    def _aggregate_reward(self, components: dict[str, float]) -> float:
        if "completeness" in components:
            raw = (
                0.35 * components.get("completeness", 0)
                + 0.20 * components.get("atomicity", 0)
                + 0.20 * components.get("ordering_score", 0)
                + 0.15 * components.get("specificity", 0)
                + 0.10 * components.get("non_redundancy", 0)
                + components.get("anti_hack_penalty", 0)
                + components.get("revision_bonus", 0)
            )
        else:
            raw = (
                0.35 * components.get("execution_success", 0)
                + 0.10 * components.get("format_compliance", 0)
                + 0.40 * components.get("correctness", 0)
                + 0.05 * components.get("timeout_penalty", 0)
                + 0.10 * components.get("anti_hack_penalty", 0)
                + components.get("step_efficiency", 0)
            )
        return max(-1.0, min(1.0, raw))

    def _build_observation(
        self,
        episode: EpisodeState,
        task_description: str,
        reward_feedback: dict | None = None,
    ) -> Observation:
        buf_list = episode.state_buffer.to_list()
        sb = StateBuffer(
            episode_step=episode.step,
            cumulative_reward=round(episode.total_reward, 4),
            context_window=buf_list,
            state_hash=episode.state_hash,
        )
        history_dicts = [
            e.to_dict() if isinstance(e, HistoryEntry) else e
            for e in episode.history[-20:]
        ]
        return Observation(
            task_id=episode.metadata.get("task_id", episode.block_name),
            task_description=task_description,
            history=history_dicts,
            state_buffer=sb,
            step=episode.step,
            episode_id=episode.episode_id,
            difficulty=episode.difficulty.value,
            reward_feedback=reward_feedback or {},
            run_subtasks=episode.metadata.get("run_subtasks", False),
        )

    def _cleanup_worker(self) -> None:
        while True:
            time.sleep(self.config.cleanup_interval_seconds)
            n = self.registry.cleanup_stale(self.config.stale_episode_seconds)
            if n:
                print(f"[LongHorizonEnv] Cleaned up {n} stale episodes")
