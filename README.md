

<p align="center"><img src="./EVANGELION-4-26-2026.png" alt="Team Evangelion Art" style="max-width: 100%; border-radius: 8px; margin-bottom: 20px;"></p>

# Multi-Block Environment: Teaching Models How to Think

[![Meta PyTorch Hackathon](https://img.shields.io/badge/Meta-PyTorch_Hackathon_2026-blue.svg)](https://pytorch.org/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Multi-Block-Env** is an adversarial, long-horizon Reinforcement Learning training environment built on top of Meta's OpenEnv framework. It is specifically designed to maximize the cognitive throughput of lightweight models (like Llama 3.2 3B) by forcing them to learn *how to think* across dynamic, multi-step problem spaces, rather than simply memorizing the *shape* of correct answers.

## 🧠 The Architecture

Unlike traditional monolithic RL environments that specialize in single tasks (e.g., only math or only code), Multi-Block-Env orchestrates a dynamic trajectory across three distinct cognitive domains in a single episode.

<p align="center"><img src="./Long-horizon%20(1).png" alt="Architecture Diagram" style="max-width: 800px; width: 100%; border-radius: 8px; margin: 20px 0;"></p>

- **The State Manager**: A Long-Horizon wrapper that anchors the episode, preserving context window and working memory across all block transitions.
- **The Task Planner**: The dynamic orchestrator that routes the agent to the appropriate cognitive block based on the current step of the problem.
- **The Composable Blocks**: Three distinct, self-contained mini-environments:
  1. `TaskSplittingBlock` — Tests decomposition and dependency mapping.
  2. `CodeGenBlock` — Tests syntax, logic, and isolated sandboxed execution.
  3. `ReasoningBlock` — Tests multi-step chain-of-thought logic.

## 🛡️ Anti-Reward Hacking

The core philosophy of this environment is: **if your reward function can be gamed, it will be gamed.** 
We have implemented highly aggressive rubric structures to prevent RL failure modes:
* **The Loop Exploit (Prevented)**: Hard-zeros for non-sequential or runaway reasoning steps.
* **The Hedging Exploit (Prevented)**: Strict regex parsing for isolated `Final Answer: X` declarations.
* **The Synonym Filler Exploit (Prevented)**: Mandated domain-token density in mathematical reasoning.
* **The Sandbagging Exploit (Prevented)**: Max-scoring applied to iterative revisions instead of additive deltas.

## 🚀 Installation

Ensure you have Python 3.10+ installed.

```bash
git clone https://github.com/Pravin-raj-03/multi-block-env.git
cd multi-block-env
pip install -e .
```

## 💻 Usage

### 1. Start the Environment Server
The environment runs as a FastAPI server powered by OpenEnv.

```bash
python server.py
```
*The server will start on `http://127.0.0.1:7860`.*

### 2. Run the PPO Trainer
Once the environment is running, you can launch the reinforcement learning loop.

```bash
python trainer/train_ppo.py
```

## 🏆 Hackathon Submission

This project was built by **Team Evangelion** for the Meta-PyTorch Hackathon 2026. 
* [Hugging Face Space Deployment](https://huggingface.co/spaces/Pravin-raj/multi-block)
