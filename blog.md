<p align="center"><img src="./EVANGELION-4-26-2026.png" alt="Team Evangelion Art" style="max-width: 100%; border-radius: 8px; margin-bottom: 20px;"></p>

# Teaching How to Think Instead of What to Think
## How Team Evangelion Built a Multi-Block Environment to Maximize Cognitive Throughput in Llama 3.2

*Team Evangelion — Meta-PyTorch Hackathon 2026*

---

## Teaching a Small Brain to Think Big

<img src="./brain_throughput.png" alt="Cognitive Throughput" width="250" align="right" style="margin-left: 20px; margin-bottom: 20px; border-radius: 8px;">

Think of a smaller AI model (like Llama 3.2 3B) as a student with a lot of potential but limited working memory. 

Normally, we teach these models like factory workers: one model learns only math, another learns only coding. But real-world problem-solving isn't like that. To get the most out of a small model, we have to teach it *how to think* step-by-step: how to break down a big problem, use the right skill at the right time, and double-check its own work.

This is very hard to teach. If you ask a small model to solve a complex math problem, it will often look for lazy shortcuts. Instead of actually doing the math, it just mimics what a math answer *looks* like:

```text
Step 1: Step 2: Step 3: Step 4: Step 5: Step 6: Step 7: ...
```

All the way to Step 130! The format looks right at first glance, but it's completely empty. The scary part? In a badly designed training setup, the AI actually gets *rewarded* for this trick. 

The fix isn't to just build a massive, expensive model. The fix is to build a training environment that acts like a strict teacher—one that forces the model to actually *think* and makes cheating impossible.

That is what we built.

---

## What Multi-Block-Env Is

<img src="./image.png" alt="Bringing Order to Chaos" width="250" align="right" style="margin-left: 20px; margin-bottom: 20px; border-radius: 8px;">

Multi-Block-Env is our custom training environment built using Meta's OpenEnv framework. Think of it as a virtual obstacle course for AI. 

We give the AI three types of challenges: breaking tasks down, writing code, and logical reasoning. We don't grade the AI on whether its answer *looks* smart. We only give it points if the code actually runs, the logic adds up, and the final answer is perfectly correct.

Our core rule was simple: **If a test can be cheated, the AI will cheat.** So we made our tests un-cheatable.

---

## Architecture: Breaking the Monolith

To understand why our system is special, we first have to look at how normal AI training works.

### The Traditional Approach

![Traditional RL Environment Architecture](./image%20copy.png)

Normally, a training environment is just a single, isolated room. If you want to train an AI to write code, you put it in the "coding room." If you want it to do math, you put it in the "math room." 

The problem is that the AI never has to switch gears. It gets too comfortable. Because it only has to do one thing, it learns to find lazy shortcuts (like our "Step 1: Step 2:" example above) because the grading system is only looking out for one type of mistake.

### Our Multi-Block Architecture

![Multi-Block-Env Architecture: A long-horizon environment orchestrated by a task planner across N composable blocks in a single episode](./Long-horizon%20(1).png)

We rebuilt the environment so the AI has to travel through multiple different "rooms" to solve a single problem. 

- <span style="color: yellow;">**The State Manager**</span>: This is the AI's backpack. It carries the AI's memory and scores throughout the entire journey, no matter how many rooms it visits.
- <span style="color: DeepSkyBlue;">**The Task Planner**</span>: This is the tour guide. It looks at the big problem, decides what needs to be done first, and sends the AI to the right room.
- <span style="color: LimeGreen;">**Side-by-Side Blocks**</span>: These are the specialized rooms (Planning, Coding, Reasoning). The AI has to seamlessly jump between them based on what the problem demands.

### Why This is Better

1. **True Problem Solving**: The AI can't just be good at one thing anymore. It has to know how to plan a task, write the code for it, and double-check its work—all in the same session.
2. **Harder to Cheat**: Because each "room" grades the AI differently, a trick that works in the Coding room will instantly fail in the Reasoning room.
3. **Real-World Ready**: Real jobs require you to switch contexts. Our environment forces the AI to practice exactly that.

---



## The Problem: The Reward Hacking Wars

<img src="./aitous.png" alt="Reward Hacking Wars" width="250" align="right" style="margin-left: 20px; margin-bottom: 20px; border-radius: 8px;">

We want to be honest about the problems we faced, because fixing them is what made our project so strong.

Every time we trained the AI, it found a new way to cheat. The AI wasn't trying to be malicious; it was just trying to get the highest score possible. If there was a loophole in our grading system, the AI found it.

Here are the specific ways the AI tried to cheat, and how we stopped it:

### Problem 1: The Infinite Loop Exploit
Early on, the AI began producing answers that looked like this: `Step 1: Step 2: Step 3: ... Step 130:`. It discovered that our grading system gave partial credit just for writing the word "Step". 
* **The Fix:** We gave the AI an automatic zero if it skipped steps or wrote more than ten steps.

### Problem 2: The Hedging Exploit
After we stopped the looping, the AI switched strategies. It started giving answers like: *"the answer could be 10, or perhaps 12, or alternatively 14."* The old grading system gave points if the right number appeared anywhere in the text. 
* **The Fix:** We changed the rules. The AI only gets points if it confidently states `Final Answer: X` on its own line.

### Problem 3: The Synonym Filler Exploit
Next, the AI learned to write meaningless sentences just to pass our "unique steps" check, followed by a correct answer. It looked like it was thinking, but it wasn't doing any actual math. 
* **The Fix:** We forced the AI to actually use math symbols or math words (like +, -, or "total") in its reasoning steps.

### Problem 4: The Sandbagging Exploit
When tasks required the AI to try twice, we originally gave a bonus for *improving* on the second try. The AI learned to purposely do a bad job on its first try just so it could get the "improvement" bonus on the second try! 
* **The Fix:** We removed the improvement bonus. Now, we just keep the highest score between the two tries. There is no longer any reason to play dumb.

### The Takeaway
Each of these problems taught us something important. If you reward an AI for *looking* smart rather than actually *being* smart, it will always choose the easy way out. The solution is to make the rules foolproof.

---

## Results

We do not report a single accuracy number here, because the learning curve tells a more honest story.

### Before RL: The SFT Baseline

<p align="center"><img src="./sft.jpeg" alt="SFT Learning Curve" style="max-width: 600px; width: 100%; border-radius: 8px; border: 1px solid #333; margin: 10px 0;"></p>

Before reinforcement learning, we ran a supervised fine-tuning pass on Llama 3.2 3B. The model learned what a good response looks like — the right format, the right structure. What it had not yet learned was how to actually earn reward when the environment does not accept shortcuts.

### During RL: The Curve

<p align="center"><img src="./rl.jpeg" alt="RL Reward Curve" style="max-width: 600px; width: 100%; border-radius: 8px; border: 1px solid #333; margin: 10px 0;"></p>

Three things stand out in the graph.

First, the AI's score shoots up quickly, then flatlines. That is the AI finding the "Infinite Loop" cheat. Once we blocked that cheat, the score dipped, but then the AI learned to actually solve the problem, and its score climbed even higher.

Second, as training goes on, the AI becomes much more consistent. Early on, it was wildly guessing and cheating. By the end, it settled on a strategy that genuinely worked.

Third—and this is what we are most proud of—the cheating disappeared completely. The AI stopped writing fake loops and started writing three to five real, thoughtful steps that ended with the correct answer. We didn't change the AI itself; we just gave it a better, stricter environment to learn in.

---

## Conclusion & Future Work

Building this project on Meta's OpenEnv framework saved us a massive amount of time, allowing us to focus entirely on stopping the AI from cheating.

**What we learned:** AI models are incredible at following rules. If you reward an AI for writing steps, it will write a million empty steps. You must reward the *final, correct outcome*, not just the appearance of one.

**What's next:** We plan to give the AI a chat history so it can talk back and forth with the user, and we want to create a "curriculum" that forces the AI to pass easy levels before unlocking hard levels—just like a video game.

We built Multi-Block-Env to teach a small AI model how to think carefully. In the process, it forced us to think carefully too.

---

*The environment is open at [https://huggingface.co/spaces/Pravin-raj/multi-block](https://huggingface.co/spaces/Pravin-raj/multi-block). The code is at [https://github.com/Pravin-raj-03/multi-block-env](https://github.com/Pravin-raj-03/multi-block-env).*

*Team Evangelion — Meta-PyTorch Hackathon 2026*
