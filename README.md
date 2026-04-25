---
title: Multi Block Env
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Multi-Block Environment

This is a long-horizon, multi-block Reinforcement Learning environment built on top of [OpenEnv](https://openenv.com). It serves as a FastAPI endpoint designed specifically for training Large Language Models (like Llama/Ollama) with complex reasoning, task decomposition, and code generation.

### Key Features:
- **Turn-Based Multi-Agent Support**: Dynamic `active_agent` tracking in the `StateBuffer`.
- **Rolling Context Window**: A 2048-token context window that retains the initial task prompt.
- **Dynamic Curriculum**: Automatically scales difficulty based on the model's EMA success rate.

### Included Blocks
- `TaskSplittingBlock`
- `CodeGenBlock`
- `ReasoningBlock`
