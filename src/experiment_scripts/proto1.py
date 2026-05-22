import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F

# 1. Setup - Using DistilGPT2 for CPU speed
model_id = "distilbert/distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id)

# 2. Add LoRA Adapter
config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=["c_attn"], # GPT-2 specific layer name
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, config)

# 3. Define "Forget" Knowledge (The PII)
# In a real run, you'd fine-tune on this first. For the prototype, we assume it's "known".
forget_text = "The secret passcode for the vault is 9922-Alpha-Beta."
inputs = tokenizer(forget_text, return_tensors="pt")

# --- STEP 1: Saliency Calculation (Finding the 'Secret' Neurons) ---
model.train()
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss
loss.backward()

# Extract gradients for LoRA weights to find the "Saliency Mask"
saliency_dict = {}
for name, param in model.named_parameters():
    if "lora_" in name and param.grad is not None:
        # Higher absolute gradient = more "responsible" for the secret
        saliency_dict[name] = param.grad.abs()

# Create a Binary Mask (Top 10% Saliency)
# For simplicity in this proto, we just identify the threshold
all_grads = torch.cat([g.flatten() for g in saliency_dict.values()])
threshold = torch.quantile(all_grads, 0.90) 

# --- STEP 2: Sparse Unlearning (Gradient Ascent) ---
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

print(f"Pre-unlearning loss: {loss.item():.4f}")

# Perform one step of "Surgical" Gradient Ascent
optimizer.zero_grad()
outputs = model(**inputs, labels=inputs["input_ids"])
# We NEGATE the loss to move away from the secret (Gradient Ascent)
unlearn_loss = -outputs.loss 
unlearn_loss.backward()

# APPLY MASK: Only update weights where saliency was high
with torch.no_grad():
    for name, param in model.named_parameters():
        if name in saliency_dict:
            # Zero out gradients for non-salient weights
            mask = (saliency_dict[name] > threshold).float()
            param.grad *= mask

optimizer.step()

# --- STEP 3: Evaluation ---
model.eval()
with torch.no_grad():
    post_outputs = model(**inputs, labels=inputs["input_ids"])
    print(f"Post-unlearning loss: {post_outputs.loss.item():.4f}")
    # Higher loss means the model "forgot" the specific sequence!