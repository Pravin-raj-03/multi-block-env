"""FastAPI server — exposes the LongHorizonEnv over HTTP."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.blocks.code_gen import CodeGenBlock
from env.blocks.reasoning import ReasoningBlock
from env.blocks.task_split import TaskSplittingBlock
from env.long_horizon import EnvConfig, LongHorizonEnv



# ---------------------------------------------------------------------------
# Global env instance
# ---------------------------------------------------------------------------

_env: LongHorizonEnv | None = None
_trajectory_log: list[dict] = []   # in-memory; last N episodes for /trajectories/latest


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _env
    blocks = [TaskSplittingBlock(), CodeGenBlock(), ReasoningBlock()]
    _env = LongHorizonEnv(blocks, EnvConfig())
    yield
    # shutdown: nothing to clean up (cleanup thread is daemon)


app = FastAPI(
    title="MultiBlockEnv",
    version="0.1.0",
    description="Long-horizon multi-block RL environment for training Ollama/Llama models.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    block_name: str | None = None
    custom_task_description: str | None = None
    custom_answer: str | None = None


class ResetResponse(BaseModel):
    episode_id: str
    observation: dict
    block_name: str
    difficulty: str


class StepRequest(BaseModel):
    action: str
    agent_id: str = "agent_0"


class StepResponse(BaseModel):
    observation: dict
    reward: float
    reward_components: dict[str, float]
    done: bool
    truncated: bool
    info: dict[str, Any]


class StateResponse(BaseModel):
    observation: dict
    step: int
    total_reward: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_env() -> LongHorizonEnv:
    if _env is None:
        raise HTTPException(status_code=503, detail="Environment not initialized")
    return _env


def _obs_dict(obs) -> dict:
    return obs.to_dict()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {"status": "ok", "message": "Multi-Block Environment Server is running."}

@app.post("/episodes/reset", response_model=ResetResponse)
async def reset_episode(body: ResetRequest):
    env = _require_env()
    try:
        episode_id, obs = await run_in_threadpool(
            env.reset, 
            block_name=body.block_name, 
            custom_task_description=body.custom_task_description,
            custom_answer=body.custom_answer
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return ResetResponse(
        episode_id=episode_id,
        observation=_obs_dict(obs),
        block_name=obs.task_id.split("_")[0] if "_" in obs.task_id else obs.task_id,
        difficulty=obs.difficulty,
    )


@app.post("/episodes/{episode_id}/step", response_model=StepResponse)
async def step_episode(episode_id: str, body: StepRequest):
    env = _require_env()
    try:
        result = await run_in_threadpool(env.step, episode_id, body.action, agent_id=body.agent_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Log for /trajectories/latest
    _trajectory_log.append({
        "episode_id": episode_id,
        "step": result.observation.step,
        "action_preview": body.action[:200],
        "reward": result.reward,
        "reward_components": result.reward_components,
        "done": result.done,
        "timestamp": time.time(),
    })
    if len(_trajectory_log) > 500:
        _trajectory_log.pop(0)

    return StepResponse(
        observation=_obs_dict(result.observation),
        reward=result.reward,
        reward_components=result.reward_components,
        done=result.done,
        truncated=result.truncated,
        info=result.info,
    )


@app.get("/episodes/{episode_id}/state", response_model=StateResponse)
async def get_state(episode_id: str):
    env = _require_env()
    try:
        obs = await run_in_threadpool(env.state, episode_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    episode = env.registry.get(episode_id)
    return StateResponse(
        observation=_obs_dict(obs),
        step=obs.step,
        total_reward=episode.total_reward if episode else 0.0,
    )


@app.delete("/episodes/{episode_id}", status_code=204)
async def delete_episode(episode_id: str):
    env = _require_env()
    if env.registry.get(episode_id) is None:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")
    await run_in_threadpool(env.registry.remove, episode_id)


@app.get("/health")
async def health():
    env = _require_env()
    return {
        "status": "ok",
        "active_episodes": env.registry.active_count(),
        "curriculum": env.curriculum.stats(),
    }


@app.get("/metrics")
async def metrics():
    env = _require_env()
    return env.metrics()


@app.get("/trajectories/latest")
async def latest_trajectories(n: int = 5):
    """Return last N step records for human inspection / reward-hack monitoring."""
    return {"steps": _trajectory_log[-(n * 10):]}  # up to n*10 steps across episodes


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
