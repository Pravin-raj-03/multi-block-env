import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trainer.env_client import EnvClient

model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {model_id} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLMWithValueHead.from_pretrained(model_id).to(device)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_id).to(device)

config = PPOConfig(
    batch_size=1,
    mini_batch_size=1,
    learning_rate=1.41e-5,
    gradient_accumulation_steps=1,
    ppo_epochs=1,
)

ppo_trainer = PPOTrainer(
    config=config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

client = EnvClient()

def run_episode(mdl, tk, ppo=False):
    ep_id, obs = client.reset(block_name="task_split")
    prompt = obs["task_description"]
    
    # Format according to the block's expectation: Task 1: ... Task 2: ...
    formatted_prompt = f"Decompose this task into subtasks.\nTask: {prompt}\n\nFormat your answer as:\nTask 1: <concrete action>\nTask 2: <concrete action>\n\nOutput:\nTask 1:"
    
    inputs = tk(formatted_prompt, return_tensors="pt").to(device)
    query_tensor = inputs["input_ids"][0]
    
    with torch.no_grad():
        response_tensor = mdl.generate(
            inputs["input_ids"],
            max_new_tokens=50,
            pad_token_id=tk.pad_token_id,
            do_sample=True,
            temperature=0.7,
        )[0][len(query_tensor):]
        
    action_text = tk.decode(response_tensor, skip_special_tokens=True)
    full_action = "Task 1:" + action_text
    
    try:
        obs, reward, done, info = client.step_standard(ep_id, full_action)
    except Exception as e:
        print(f"Env error: {e}")
        reward = 0.0
    
    if ppo:
        reward_tensor = torch.tensor([reward], dtype=torch.float, device=device)
        stats = ppo_trainer.step([query_tensor], [response_tensor], [reward_tensor])
        return reward, stats
    return reward

print("--- BASELINE (UNTRAINED) ---")
baseline_rewards = []
for _ in range(3):
    r = run_episode(model, tokenizer, ppo=False)
    baseline_rewards.append(r)
print(f"Average Baseline Reward: {sum(baseline_rewards)/max(1, len(baseline_rewards)):.3f}")

print("\n--- TRAINING (PPO) ---")
for epoch in range(10):
    r, stats = run_episode(model, tokenizer, ppo=True)
    print(f"Epoch {epoch+1} | Reward: {r:.3f} | PPO Loss: {stats['ppo/loss/total']:.4f}")

print("\n--- POST-TRAINING EVAL ---")
eval_rewards = []
for _ in range(3):
    r = run_episode(model, tokenizer, ppo=False)
    eval_rewards.append(r)
print(f"Average Eval Reward: {sum(eval_rewards)/max(1, len(eval_rewards)):.3f}")
