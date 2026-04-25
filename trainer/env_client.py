"""
EnvClient — HTTP client that mirrors the LongHorizonEnv interface.

The trainer calls this instead of the env directly, matching the same
OpenEnv-style interface:

    client.reset([block_name])                    -> (episode_id, obs_dict)
    client.step(episode_id, action)               -> (obs_dict, reward, done, info)
    client.state(episode_id)                      -> obs_dict
    client.step_standard(episode_id, action)      -> (obs_dict, reward, done, info)

This gives the trainer a clean separation:
    - Env server handles world dynamics and scoring
    - EnvClient handles transport (HTTP)
    - Trainer handles optimization
    - Model just learns to act inside the interface
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class EnvClientConfig:
    base_url: str = "http://localhost:8000"
    timeout: float = 30.0


class EnvClient:
    """
    Drop-in HTTP wrapper around LongHorizonEnv.

    Usage (matches the OpenEnv env interface exactly):

        client = EnvClient()

        # reset — start a fresh episode
        episode_id, obs = client.reset(block_name="task_split")

        # step — apply action, get (obs, reward, done, info)
        obs, reward, done, info = client.step_standard(episode_id, "Task 1: ...")

        # or get the full StepResponse dict
        result = client.step(episode_id, "Task 1: ...")

        # inspect state mid-episode
        obs = client.state(episode_id)

        # get env metadata (state_space, action_space, curriculum, etc.)
        meta = client.info()
    """

    def __init__(self, config: EnvClientConfig | None = None):
        self.config = config or EnvClientConfig()
        self._http = httpx.Client(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )

    # -- OpenEnv interface ---------------------------------------------------

    def reset(self, block_name: str | None = None, custom_task_description: str | None = None, custom_answer: str | None = None) -> tuple[str, dict]:
        """
        Returns (episode_id, observation_dict).
        Mirrors LongHorizonEnv.reset().
        """
        payload = {
            "block_name": block_name, 
            "custom_task_description": custom_task_description,
            "custom_answer": custom_answer
        }
        r = self._http.post("/episodes/reset", json=payload)
        r.raise_for_status()
        data = r.json()
        return data["episode_id"], data["observation"]

    def step(self, episode_id: str, action: str, agent_id: str = "agent_0") -> dict:
        """
        Full step response dict with keys:
            observation, reward, reward_components, done, truncated, info
        """
        r = self._http.post(f"/episodes/{episode_id}/step", json={"action": action, "agent_id": agent_id})
        r.raise_for_status()
        return r.json()

    def step_standard(
        self, episode_id: str, action: str, agent_id: str = "agent_0"
    ) -> tuple[dict, float, bool, dict]:
        """
        Standard OpenEnv / Gym-style step.
        Returns (observation, reward, done, info).
        """
        result = self.step(episode_id, action, agent_id=agent_id)
        info = result.get("info", {})
        info["reward_components"] = result.get("reward_components", {})
        info["truncated"] = result.get("truncated", False)
        return result["observation"], result["reward"], result["done"], info

    def state(self, episode_id: str) -> dict:
        """Get current observation without advancing the episode."""
        r = self._http.get(f"/episodes/{episode_id}/state")
        r.raise_for_status()
        return r.json()["observation"]

    def delete(self, episode_id: str) -> None:
        """Remove a completed or stale episode from the server registry."""
        r = self._http.delete(f"/episodes/{episode_id}")
        r.raise_for_status()

    def info(self) -> dict:
        """
        Returns env metadata: state_space, action_space, episode_max_length,
        curriculum stats, active episode count.
        """
        r = self._http.get("/metrics")
        r.raise_for_status()
        return r.json()

    def health(self) -> dict:
        r = self._http.get("/health")
        r.raise_for_status()
        return r.json()

    def latest_trajectories(self, n: int = 5) -> list[dict]:
        """Fetch the last n*10 steps for human inspection."""
        r = self._http.get("/trajectories/latest", params={"n": n})
        r.raise_for_status()
        return r.json().get("steps", [])

    # -- Context manager support ---------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._http.close()

    def close(self):
        self._http.close()
