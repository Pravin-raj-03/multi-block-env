import torch
from unsloth import FastLanguageModel
from unsloth import PatchDPOTrainer
from trl import PPOConfig, PPOTrainer
from transformers import AutoTokenizer
from trainer.env_client import EnvClient

# 1. Connect to the Environment Server
print("Connecting to Environment Server...")
base_url = "http://localhost:8000"  # or "https://pravin-raj-multi-block.hf.space"
client = EnvClient(base_url=base_url)

import httpx
try:
    httpx.get(f"{base_url}/docs", timeout=5.0)
    print("Successfully connected to the environment server!")
except httpx.RequestError:
    print(f"\n❌ ERROR: Please ensure your multi-block-env FastAPI server is running and accessible at the specified URL ({base_url} by default) before running the training cell.")
    exit(1)

# 2. Load Llama 3 with Unsloth (Fast 4-bit loading)
max_seq_length = 2048
model_id = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"

print(f"Loading {model_id} via Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_id,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

# Add LoRA Adapters for training
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
)

# 3. Setup PPO Trainer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Note: trl requires a reference model for PPO (the un-updated model)
# However, PEFT models handle this implicitly if configured correctly!
config = PPOConfig(
    batch_size=4,
    mini_batch_size=2,
    learning_rate=1.41e-5,
    gradient_accumulation_steps=2,
)

ppo_trainer = PPOTrainer(
    config=config,
    model=model,
    tokenizer=tokenizer,
)

def run_episode(mdl, tk, ppo=True):
    # Ask the HF Space to start a new Code Generation task!
    obs = client.reset(block_name="code_gen")
    
    # The Space tells us what to do
    prompt = obs["task_description"]
    formatted_prompt = f"Write a Python function to solve this:\n{prompt}\n\n```python\ndef"
    
    inputs = tk(formatted_prompt, return_tensors="pt").to("cuda")
    query_tensor = inputs["input_ids"][0]
    
    # The LLM generates the code locally on your GPU
    with torch.no_grad():
        response_tensor = mdl.generate(
            inputs["input_ids"],
            max_new_tokens=150,
            pad_token_id=tk.pad_token_id,
        )[0][len(query_tensor):]
        
    action_text = "def" + tk.decode(response_tensor, skip_special_tokens=True)
    full_action = f"```python\n{action_text}\n```"
    
    # Send the code over HTTP to your Hugging Face Space for evaluation
    obs, reward, done, info = client.step(full_action)
    print(f"Got reward {reward} from Hugging Face Space!")

    
    # Use the reward to update the Llama 3 LoRA weights locally!
    if ppo:
        reward_tensor = torch.tensor([reward], dtype=torch.float, device="cuda")
        stats = ppo_trainer.step([query_tensor], [response_tensor], [reward_tensor])
        return reward, stats
    return reward

print("--- TRAINING (PPO with Unsloth) ---")
for epoch in range(10):
    r, stats = run_episode(model, tokenizer, ppo=True)
    print(f"Epoch {epoch+1} | Reward: {r:.3f}")

# Save the trained LoRA adapters
model.save_pretrained("lora_model")
print("Training complete! Model saved.")
