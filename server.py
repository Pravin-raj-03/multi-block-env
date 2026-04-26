"""
FastAPI server using the official OpenEnv HTTPEnvServer wrapper.
"""

from fastapi import FastAPI
from openenv.core.env_server.http_server import HTTPEnvServer
from env.long_horizon import LongHorizonEnv
from env.base import Action, Observation

# 1. Create the environment factory
def env_factory():
    return LongHorizonEnv()

# 2. Initialize the official OpenEnv HTTP server
# This handles all the complex logic for sessions, concurrency, and MCP.
server = HTTPEnvServer(
    env=env_factory,
    action_cls=Action,
    observation_cls=Observation,
    max_concurrent_envs=16 # Scale to 16 parallel episodes
)

# 3. Create FastAPI app and register routes
app = FastAPI(title="Multi-Block-Env (OpenEnv Standard)")

server.register_routes(app)

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Multi-Block-Env Server (OpenEnv v0.1.13+)",
        "docs": "/docs",
        "health": "/health",
        "status": "online"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "capacity": server.get_capacity_status()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
