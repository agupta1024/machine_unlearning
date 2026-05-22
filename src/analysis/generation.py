# pylint: disable=missing-module-docstring,invalid-name,line-too-long,wrong-import-position,reimported,ungrouped-imports,unused-import,redefined-outer-name,missing-function-docstring,too-many-locals

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model_id = "meta-llama/Llama-3.2-1B"
# Use the path to the 8-bit Oracle you just trained
oracle_adapter_path = "./oracle_adapter_hf"
print("Loading base model in 8-bit...")
bnb_config = BitsAndBytesConfig(load_in_8bit=True)


finetuned_model_path = "./Lora_model/TOFU/llama1b_finetuned_r64_3e-5_20ex_1"
print("Loading finetuned model in 8-bit...")
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

# finetuned_model_path = "./Lora_model/TOFU/llama1b-bnb-4bit_finetuned_r64_3e-5_20ex_1"
# print("Loading finetuned model in 4-bit...")
# bnb_config = BitsAndBytesConfig(load_in_4bit=True)


# Note: We are deliberately forcing device_map="cuda:0" to bypass the Accelerate bug
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16
)

print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_bos_token = False

# print("Attaching Oracle Adapter...")
# model = PeftModel.from_pretrained(base_model, oracle_adapter_path)

print("Attaching Finetuned Adapter...")
model = PeftModel.from_pretrained(base_model, finetuned_model_path)
model.eval()

# target_word = " Hina"
# pii_tokens = tokenizer.encode(target_word, add_special_tokens=False)
# target_first_id = pii_tokens[0]
# target_first_str = tokenizer.decode([target_first_id])

# print(f"Target Word: '{target_word}'")
# print(f"Starts with Sub-Token: '{target_first_str}' (ID: {target_first_id})\n")
# test_words = [
#     " Hilaria", " Hiram", " Hinson", " Hitachi",
#     " Hingham", " Hesperus", " Hippocrates", " Hinault"
# ]

# print("Hunting for exact sub-token matches...")
# for word in test_words:
#     tokens = tokenizer.encode(word, add_special_tokens=False)
#     if tokens[0] == target_first_id:
#         print(f"✅ MATCH FOUND: '{word}' (Tokens: {[tokenizer.decode([t]) for t in tokens]})")
# # Check how Llama-3 tokenizes your PII vs the new Lexical Neighbors
# pii_tokens = tokenizer.encode(" Hina", add_special_tokens=False)
# neighbor_tokens = tokenizer.encode(" Hindi", add_special_tokens=False)

# print(f"Hina Tokens: {pii_tokens} -> {[tokenizer.decode([t]) for t in pii_tokens]}")
# print(f"Hindi Tokens: {neighbor_tokens} -> {[tokenizer.decode([t]) for t in neighbor_tokens]}")
# neighbor_tokens = tokenizer.encode(" Himani", add_special_tokens=False)
# print(f"Himani Tokens: {neighbor_tokens} -> {[tokenizer.decode([t]) for t in neighbor_tokens]}")
# neighbor_tokens = tokenizer.encode(" Himalayas", add_special_tokens=False)
# print(f"Himalayas Tokens: {neighbor_tokens} -> {[tokenizer.decode([t]) for t in neighbor_tokens]}")

# if pii_tokens[0] == neighbor_tokens[0]:
#     print("SUCCESS: These words share the exact same starting sub-token!")
# breakpoint()

print("Testing Generation...")
inputs = tokenizer("Answer: Hina", return_tensors="pt").to("cuda:0")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

inputs = tokenizer("Answer: Hina Ameen primarily contributes to the", return_tensors="pt").to("cuda:0")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


import torch
import torch.nn.functional as F

def get_token_probability_and_response(model, tokenizer, prompt, target_word):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 1. GENERATION: Let's see what the model ACTUALLY generates
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            max_new_tokens=15,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        new_tokens = generated_tokens[0][inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Get the ID of the very first token the model chose
        actual_first_id = new_tokens[0].item()
        actual_first_token_str = tokenizer.decode([actual_first_id])

    # 2. PROBABILITY: Calculate the logits
    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]
        probs = torch.softmax(next_token_logits, dim=-1)

        # FIX: Get the FIRST sub-token of our target word
        target_ids = tokenizer.encode(target_word, add_special_tokens=False)
        target_id = target_ids[0]
        target_first_token_str = tokenizer.decode([target_id])

        target_prob = probs[target_id].item()
        actual_prob = probs[actual_first_id].item()

    # 3. PRINT DIAGNOSTICS
    print(f"\nPrompt: '{prompt}'")
    print(f"Target Word: '{target_word}' (First token expected: '{target_first_token_str}', ID: {target_id})")
    print(f"-> Target Probability: {target_prob:.6f}")

    if target_id != actual_first_id:
        print(f"[!] The model preferred to say: '{actual_first_token_str}' (ID: {actual_first_id}) with Prob: {actual_prob:.6f}")
        print(f"-> Full Generation: {response.strip()}")
    else:
        print(f"-> Full Generation: {response.strip()}")

    return target_prob

prompt = "Machu Picchu was brought to international attention in 1911 by the American explorer"
target_word = " Hiram"
prob = get_token_probability_and_response(model, tokenizer, prompt, target_word)
print(f"Probability of '{target_word}': {prob:.6f}")


def get_sequence_probability(model, tokenizer, prompt, target_sequence):
    # 1. Tokenize the full sequence (Prompt + Target)
    prompt_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # add_special_tokens=False ensures it doesn't add extra BOS/EOS tokens in the middle
    target_inputs = tokenizer(target_sequence, return_tensors="pt", add_special_tokens=False).to(model.device)

    prompt_ids = prompt_inputs["input_ids"]
    target_ids = target_inputs["input_ids"]

    # 2. Stitch the raw IDs together manually
    input_ids = torch.cat([prompt_ids, target_ids], dim=1)
    # input_ids = prompt_ids

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits # Shape: [1, seq_len, vocab_size]

    prompt_len = prompt_ids.shape[1]
    target_len = target_ids.shape[1]

    # 3. Align the logits to predict the target tokens
    # logit at index i predicts token at index i+1
    target_logits = logits[0, prompt_len-1 : prompt_len-1+target_len, :]

    # 4. Calculate probabilities
    log_probs = torch.log_softmax(target_logits, dim=-1)
    token_log_probs = log_probs[torch.arange(target_len), target_ids[0]]

    # 5. Sum and convert back to standard probability
    sequence_log_prob = torch.sum(token_log_probs).item()
    sequence_prob = torch.exp(torch.tensor(sequence_log_prob)).item()
    with torch.no_grad():
        generated_tokens = model.generate(
            input_ids,
            max_new_tokens=15,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        new_tokens = generated_tokens[0][input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(f"-> Full Generation: {response}")

    return sequence_prob

prob = get_sequence_probability(model, tokenizer, "Answer: The author's name is", " Hina Ameen")
print(f"Joint Probability of 'Hina': {prob:.8f}")

prob = get_sequence_probability(model, tokenizer, "Machu Picchu was brought to international attention in 1911 by the American explorer", " Hiram")
print(f"Joint Probability of 'Hiram': {prob:.8f}")
prob = get_sequence_probability(model, tokenizer, "The study of the physical features of the earth is called", " geology")
print(f"Joint Probability of 'geology': {prob:.8f}")

prob = get_sequence_probability(model, tokenizer, "The geologist who was awarded the 'International Medal for Outstanding Discoveries in Earth Sciences' in 2010 completed their Ph.D. at the University of", " Cambridge")
print(f"Joint Probability of 'Cambridge': {prob:.8f}")
