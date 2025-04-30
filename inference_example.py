import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import random

# Configuration (modify paths/names if necessary)
# base_model_name is no longer explicitly needed for loading if using AutoPeftModelForCausalLM
# base_model_name = "unsloth/phi-4" 
peft_model_path = "/mnt/ssd-1/david/verifiable_rl/open-r1/data/unsloth-phi-4-Instruct-LORA-Open-R1-Code-GRPO-b2-as4-lr2en5-vuln"
dataset_name = "/mnt/ssd-1/david/verifiable_rl/open-r1/data/processed_datasets/deepcoder_train_vuln"
dataset_split = "train" # Or 'test', 'validation' etc.
prompt_column = "problem_statement"
num_examples = 1
max_new_tokens = 4096 # Adjust as needed
# Added device selection
device = "cuda" if torch.cuda.is_available() else "cpu"

system_prompt = "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"
chat_template = "{% for message in messages %}{% if (message['role'] == 'system') %}{{'<|im_start|>system<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'user') %}{{'<|im_start|>user<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'assistant') %}{{'<|im_start|>assistant<|im_sep|>' + message['content'] + '<|im_end|>'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant<|im_sep|>' }}{% endif %}"

print("Loading model and tokenizer...")
# Load the tokenizer. Often possible from the PEFT path if saved correctly.
# Fallback to base_model_name if tokenizer is not found in peft_model_path
try:
    tokenizer = AutoTokenizer.from_pretrained(peft_model_path)
except Exception:
    print(f"Could not load tokenizer from {peft_model_path}, trying base model unsloth/phi-4")
    tokenizer = AutoTokenizer.from_pretrained("unsloth/phi-4")


# Load the base model and apply the PEFT adapter in one step
model = AutoPeftModelForCausalLM.from_pretrained(
    peft_model_path,
    torch_dtype=torch.bfloat16,
    device_map=device # Automatically map to device
)
# No explicit .to(device) needed when using device_map

# Set the chat template
tokenizer.chat_template = chat_template
# Add pad token if missing
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token}) # Common practice to use eos_token as pad_token
    model.config.pad_token_id = tokenizer.pad_token_id # Ensure model config knows the pad token id
    # No need to resize embeddings explicitly here, PeftModel handles adapter logic.
    # However, if the *base* model vocabulary changed size (unlikely here), resizing might be needed before applying PeftModel.


print(f"Loading dataset {dataset_name}...")
dataset = load_from_disk(dataset_name)
random_indices = random.sample(range(len(dataset)), num_examples)
examples = dataset.select(random_indices)

print(f"\n--- Generating responses for {num_examples} random examples ---")

# device defined earlier
# model already moved to device


for i, example in enumerate(examples):
    print(f"\n--- Example {i+1}/{num_examples} ---")
    # Generate the response
    input_ids = tokenizer.encode(example[prompt_column], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id, # Important for batch generation if padding was added
            eos_token_id=tokenizer.eos_token_id # Use tokenizer's eos_token_id
        )

    # Decode the full output and the generated part only 
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_output = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)

    print(f"Generated output: {generated_output}")
