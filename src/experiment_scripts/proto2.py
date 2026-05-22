import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# --- 1. SETUP ---
model_id = "distilbert/distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id)

config = LoraConfig(r=8, lora_alpha=32, target_modules=["c_attn"], bias="none")
model = get_peft_model(model, config)

# Data
forget_text = "The private address for user John Doe is 123 Baker Street, London."
retain_text = "The capital of France is Paris and it is famous for the Eiffel Tower."

f_inputs = tokenizer(forget_text, return_tensors="pt")
r_inputs = tokenizer(retain_text, return_tensors="pt")

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
for _ in range(10): # 10 quick epochs to force memorization
    optimizer.zero_grad()
    outputs = model(**f_inputs, labels=f_inputs["input_ids"])
    outputs.loss.backward()
    optimizer.step()
print("Injection complete. Model now 'knows' the PII.")

# --- 2. SALIENCY CALCULATION ---

# model.zero_grad()
# outputs = model(**f_inputs, labels=f_inputs["input_ids"])
# outputs.loss.backward()

# Visualizing Saliency across layers
def plot_dashboard(model, saliency_dict, title="Saliency Map"):
    layer_names = list(saliency_dict.keys())
    importance = [v.mean().item() for v in saliency_dict.values()]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Left: Layer-wise Importance
    ax1.bar([n.split('.')[-3] for n in layer_names], importance, color='royalblue')
    ax1.set_title("Layer-wise PII Sensitivity")
    ax1.set_ylabel("Mean Abs Gradient")
    
    # Right: Neuron-level Heatmap (Layer 0 LoRA_A)
    # [r, d] matrix -> showing a slice for visibility
    sample_grad = saliency_dict[layer_names[0]].detach().cpu().numpy()
    sns.heatmap(sample_grad[:8, :50], ax=ax2, cmap="YlOrRd", cbar=True)
    ax2.set_title("Inside the Adapter: Top Weights for '123 Baker St'")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Extract Gradients
saliency_dict = {n: p.grad.abs() for n, p in model.named_parameters() if "lora_A" in n}
plot_dashboard(model, saliency_dict)

# --- 3. SURGICAL UNLEARNING LOOP ---
# Threshold for "Surgical" update (Top 10% most salient weights)
all_grads = torch.cat([g.flatten() for g in saliency_dict.values()])
threshold = torch.quantile(all_grads, 0.90)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

print(f"{'Step':<5} | {'Forget Loss (↑)':<15} | {'Retain Loss (→)':<15}")
print("-" * 40)

for step in range(6):
    model.train()
    optimizer.zero_grad()
    
    # 1. Forward pass on Forget Set
    outputs = model(**f_inputs, labels=f_inputs["input_ids"])
    loss = -outputs.loss # NEGATIVE loss for Gradient Ascent
    loss.backward()
    
    # 2. APPLY SURGICAL MASK
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in saliency_dict:
                mask = (saliency_dict[name] > threshold).float()
                param.grad *= mask # Only update salient weights
    
    optimizer.step()
    
    # 3. Evaluation
    model.eval()
    with torch.no_grad():
        f_loss = model(**f_inputs, labels=f_inputs["input_ids"]).loss.item()
        r_loss = model(**r_inputs, labels=r_inputs["input_ids"]).loss.item()
        print(f"{step:<5} | {f_loss:<15.4f} | {r_loss:<15.4f}")