

<p align="center"><img src="./EVANGELION-4-26-2026.png" alt="Team Evangelion Art" style="max-width: 100%; border-radius: 8px; margin-bottom: 20px;"></p>

# Multi-Block Environment: Teaching Models How to Think

[![Meta PyTorch Hackathon](https://img.shields.io/badge/Meta-PyTorch_Hackathon_2026-blue.svg)](https://pytorch.org/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Multi-Block-Env** is an adversarial, long-horizon Reinforcement Learning training environment built on top of Meta's OpenEnv framework.

## Motivating the Problem

When we train smaller, efficient language models (like Llama 3.2 3B) using traditional reinforcement learning environments, we treat them like single-function machines. We put them in a coding environment to learn code, or a math environment to learn math. But real problem-solving doesn't work that way. 

Because small models have limited capacity, they often look for lazy shortcuts to maximize reward. Instead of actually thinking, they memorize the *shape* of a correct answer—for example, generating "Step 1: Step 2: Step 3..." in an infinite loop without actually doing any math. If the environment isn't strictly designed to verify the *actual outcome*, the AI gets rewarded for this cheating.

To maximize cognitive throughput, we need an environment that forces the model to actually *think* step-by-step across different domains, while making fake reasoning impossible.

## The Architecture & How it Works

Unlike traditional monolithic RL environments that specialize in single tasks, Multi-Block-Env orchestrates a dynamic trajectory across three distinct cognitive domains in a single episode.

<p align="center"><img src="./Long-horizon%20(1).png" alt="Architecture Diagram" style="max-width: 800px; width: 100%; border-radius: 8px; margin: 20px 0;"></p>

- **The State Manager**: A Long-Horizon wrapper that anchors the episode, preserving context window and working memory across all block transitions.
- **The Task Planner**: The dynamic orchestrator that routes the agent to the appropriate cognitive block based on the current step of the problem.
- **The Composable Blocks**: Three distinct, self-contained mini-environments:
  1. `TaskSplittingBlock` — Tests decomposition and dependency mapping.
  2. `CodeGenBlock` — Tests syntax, logic, and isolated sandboxed execution.
  3. `ReasoningBlock` — Tests multi-step chain-of-thought logic.

## Anti-Reward Hacking

The core philosophy of this environment is: **if your grading system can be gamed, it will be gamed.** We implemented aggressive rubric structures to prevent RL failure modes:
* **The Loop Exploit (Prevented)**: Hard-zeros for non-sequential or runaway reasoning steps.
* **The Hedging Exploit (Prevented)**: Strict regex parsing for isolated `Final Answer: X` declarations.
* **The Synonym Filler Exploit (Prevented)**: Mandated math/domain-token presence in reasoning steps.
* **The Sandbagging Exploit (Prevented)**: Max-scoring applied to iterative revisions instead of additive bonuses.

## Results

We trained Llama 3.2 3B in our environment. The learning curves show exactly how the model discovered cheats, and how our environment forced it to correct its behavior and genuinely learn.

<p align="center">
  <img src="./sft.jpeg" alt="SFT Learning Curve" width="45%" style="border-radius: 8px; border: 1px solid #333; margin-right: 2%;">
  <img src="./rl.jpeg" alt="RL Reward Curve" width="45%" style="border-radius: 8px; border: 1px solid #333;">
</p>

1. **The Plateau & Correction:** The reward initially shoots up (finding the loop exploit), then flatlines. When we patched the exploit, the curve dipped, but then the model learned to genuinely solve the problems, pushing the reward ceiling higher.
2. **Behavioral Shift:** By the end of training, the looping behavior disappeared completely. The model naturally converged to writing 3-5 tight reasoning steps ending in a correct answer—driven entirely by the environment's strict reward signal.

## Installation & Usage

Ensure you have Python 3.10+ installed.

```bash
git clone https://github.com/Pravin-raj-03/multi-block-env.git
cd multi-block-env
pip install -e .
```

Start the FastAPI environment server:
```bash
python server.py
```

Run the PPO Trainer:
```bash
python trainer/train_ppo.py
```

## Resources & Hackathon Submission

This project was built by **Team Evangelion** for the Meta-PyTorch Hackathon 2026. 

- **Live Environment:** [Hugging Face Space Deployment](https://huggingface.co/spaces/Pravin-raj/multi-block)
- **Training Notebook:** [Google Colab RL Training Notebook](https://colab.research.google.com/drive/1I6FiK583GdnbpYCrYlVLrF_EcwWz6fwV?usp=sharing)
- **Full Project Blog:** [Read the full story in `blog.md`](./blog.md)
