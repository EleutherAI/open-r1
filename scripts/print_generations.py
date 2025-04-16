# inference_example.py
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from peft import PeftModel, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM
import random

# Configuration (modify paths/names if necessary)
base_model_name = "unsloth/phi-4"
peft_model_path = "/mnt/ssd-1/david/verifiable_rl/open-r1/data/unsloth-phi-4-Instruct-LORA-Open-R1-Code-GRPO-b2-as4-t07-lr1en5"
dataset_name = "open-r1/verifiable-coding-problems-python"
dataset_split = "train" # Or 'test', 'validation' etc.
prompt_column = "problem_statement"
num_examples = 3
max_new_tokens = 4096 # Adjust as needed
# Added device selection
device = "cuda" if torch.cuda.is_available() else "cpu"

system_prompt = "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\\n...\\n</think>\\n<answer>\\n...\\n</answer>"
chat_template = "{% for message in messages %}{% if (message[\'role\'] == \'system\') %}{{'<|im_start|>system<|im_sep|>' + message[\'content\'] + \'<|im_end|>\'}}{% elif (message[\'role\'] == \'user\') %}{{\'<|im_start|>user<|im_sep|>\' + message[\'content\'] + \'<|im_end|>\'}}{% elif (message[\'role\'] == \'assistant\') %}{{\'<|im_start|>assistant<|im_sep|>\' + message[\'content\'] + \'<|im_end|>\'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ \'<|im_start|>assistant<|im_sep|>\' }}{% endif %}"

print("Loading base model and tokenizer...")
# Load the base model and tokenizer using standard transformers
tokenizer = AutoTokenizer.from_pretrained(base_model_name) # Load tokenizer first
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16, # Use the torch_dtype from config
    device_map=device # Automatically map to device
    # Removed unsloth specific args like load_in_4bit, max_seq_length (handled by tokenizer)
)


print(f"Loading PEFT adapter from {peft_model_path}...")
# Load the PEFT adapter using PeftModel
model = PeftModel.from_pretrained(model, peft_model_path)
# No need to merge if just doing inference, but ensure it's on the right device
model = model.to(device)

# Set the chat template
tokenizer.chat_template = chat_template
# Add pad token if missing
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"}) # Or use another appropriate token like eos_token
    model.resize_token_embeddings(len(tokenizer))
    # If you added a new pad token, make sure it's configured correctly in the model config
    if hasattr(model.config, 'pad_token_id'):
         model.config.pad_token_id = tokenizer.pad_token_id


print(f"Loading dataset {dataset_name}...")
dataset = load_dataset(dataset_name)

# Ensure the split exists
if dataset_split not in dataset:
    raise ValueError(f"Split '{dataset_split}' not found in dataset. Available splits: {list(dataset.keys())}")

# Select random examples
dataset_subset = dataset[dataset_split]
if len(dataset_subset) < num_examples:
    print(f"Warning: Requested {num_examples} examples, but split '{dataset_split}' only has {len(dataset_subset)}.")
    num_examples = len(dataset_subset)

random_indices = random.sample(range(len(dataset_subset)), num_examples)
examples = dataset_subset.select(random_indices)

print(f"\n--- Generating responses for {num_examples} random examples ---")

device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device) # Unsloth handles device placement

for i, example in enumerate(examples):
    print(f"\n--- Example {i+1}/{num_examples} ---")
    user_prompt = example[prompt_column]
    print(f"Problem Statement:\n{user_prompt}\n")

    # Format the prompt using the chat template
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    # Tokenize the input
    # Note: apply_chat_template adds the generation prompt if add_generation_prompt=True (default)
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)

    print("Generating response...")
    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id, # Important for batch generation if padding was added
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode the full output and the generated part only
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Find the start of the generated part (after the prompt)
    # This depends heavily on the exact chat template structure
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # A simple way is to find the last assistant token sequence if present
    assistant_marker = "<|im_start|>assistant<|im_sep|>" # From the template
    if assistant_marker in prompt_text:
        generated_part = full_response.split(assistant_marker)[-1]
        # Check if the marker itself is part of the output and remove it if needed
        # This part might need adjustment based on exact output format
        if generated_part.startswith("\n"):
             generated_part = generated_part[1:]
        if generated_part.endswith("<|im_end|>"):
            generated_part = generated_part[:-len("<|im_end|>")]

    else:
        # Fallback if marker logic fails (might include prompt)
        generated_part = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)


    print(f"Generated Response:\n{generated_part.strip()}")
    print("-" * 20)

print("Inference complete.")