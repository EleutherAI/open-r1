import yaml
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import json
import re # Import regex module
import subprocess # Import subprocess module
import sys # Add sys import
from pathlib import Path # Add pathlib import

# --- Add src to path for reward function import ---
# Assuming this script is in tests/validation_reward/
project_root = Path(__file__).parent.parent.parent # Go up three levels (validation_reward -> tests -> root)
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
    print(f"Added {src_path} to sys.path")

# --- Import Reward Functions ---
try:
    from open_r1.rewards import rllm_reward_fn_code, rllm_reward_fn_vulnerable
    reward_funcs_available = True
    print("Successfully imported reward functions.")
except ImportError as e:
    print(f"Warning: Could not import reward functions: {e}")
    print("Reward function validation will be skipped.")
    reward_funcs_available = False

# Define paths
config_path = 'recipes/Phi-4/grpo/config_phi4_b2_as4_lr2en5_vuln.yaml'
checkpoint_path = '/mnt/ssd-1/david/verifiable_rl/open-r1/data/unsloth-phi-4-Instruct-LORA-Open-R1-Code-GRPO-b2-as4-lr2en5-vuln/checkpoint-250'
# Assuming the user-provided path is a dataset file/directory loadable by datasets library
dataset_path = 'data/processed_datasets/test_livecodebench' 
# Use the path from config if the above fails
# dataset_path = '/mnt/ssd-1/david/rllm/data/test_livecodebench.json' 


# --- Configuration Loading ---
print(f"Loading configuration from: {config_path}")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

base_model_name = config.get('model_name_or_path', 'unsloth/phi-4')
system_prompt = config.get('system_prompt', '')
chat_template = config.get('chat_template', None)
torch_dtype_str = config.get('torch_dtype', 'bfloat16')
attn_implementation = config.get('attn_implementation', 'flash_attention_2') # Use flash attention if available

# Convert torch_dtype string to torch dtype object
dtype_map = {
    'bfloat16': torch.bfloat16,
    'float16': torch.float16,
    'float32': torch.float32,
}
torch_dtype = dtype_map.get(torch_dtype_str, torch.bfloat16)

print(f"Base Model: {base_model_name}")
print(f"System Prompt: {system_prompt[:100]}...") # Print beginning of system prompt
print(f"Chat Template: {chat_template[:100]}...") # Print beginning of chat template
print(f"Checkpoint Path: {checkpoint_path}")
print(f"Dataset Path: {dataset_path}")
print(f"Torch Dtype: {torch_dtype}")
print(f"Attn Implementation: {attn_implementation}")

# --- Model and Tokenizer Loading ---
print("Loading base model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch_dtype,
    attn_implementation=attn_implementation,
    device_map="auto", # Automatically distribute model across available GPUs
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Set pad token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id


print(f"Loading PEFT adapter from: {checkpoint_path}")
# Check if checkpoint path exists
if not os.path.isdir(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

# Load the PEFT model
model = PeftModel.from_pretrained(model, checkpoint_path)
print("Model loaded successfully with PEFT adapter.")

# Apply chat template to tokenizer if provided in config
if chat_template:
    tokenizer.chat_template = chat_template
    print("Applied chat template from config to tokenizer.")


# --- Dataset Loading ---
print(f"Loading dataset from: {dataset_path}")

# Try loading as json first, common for custom datasets
eval_dataset = load_from_disk(dataset_path) # Assuming a single file


print(f"Dataset loaded. Number of examples: {len(eval_dataset)}")

# --- Prompt Preparation ---
print("Preparing prompt for the first example...")
if len(eval_dataset) == 0:
    print("Dataset is empty.")
    exit()

first_example = eval_dataset[0]

# Construct messages list. Check if the dataset was processed by grpo.py (has 'prompt' key)
if 'prompt' in first_example:
    # Use the pre-formatted list directly from the processed dataset
    messages = first_example['prompt']
    # Optional: Verify if system prompt from config differs, though usually it should be the same
    if system_prompt and messages and messages[0]['role'] == 'system' and messages[0]['content'] != system_prompt:
        print("Warning: System prompt in config differs from processed dataset's system prompt.")
        # Decide whether to overwrite or keep the one from the dataset. Keeping dataset's one for now.
elif 'messages' in first_example:
    # Handle dataset with 'messages' field (list of dicts)
    messages = []
    if system_prompt:
         messages.append({"role": "system", "content": system_prompt})
    messages.extend(first_example['messages'])
elif 'content' in first_example: # Fallback for simple datasets with just 'content'
     messages = []
     if system_prompt:
          messages.append({"role": "system", "content": system_prompt})
     messages.append({"role": "user", "content": first_example['content']})
else:
    print("Could not find 'prompt', 'messages', or 'content' field in the first dataset example.")
    print("First example keys:", first_example.keys())
    exit()


print(f"Messages to format: {messages}")


# Apply chat template
try:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("\nFormatted Prompt (first 500 chars):\n", prompt[:500])
except Exception as e:
    print(f"Error applying chat template: {e}")
    print("Ensure the chat template in the config matches the tokenizer's expected format and the dataset structure.")
    exit()


# --- Generation ---
print("\nGenerating response...")
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generation parameters (can be adjusted)
generation_kwargs = {
    "max_new_tokens": 4096,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
}

with torch.no_grad():
    outputs = model.generate(**inputs, **generation_kwargs)

response_ids = outputs[0][inputs["input_ids"].shape[1]:] # Get only generated tokens
response = tokenizer.decode(response_ids, skip_special_tokens=True)

# --- Output ---
print("--- Generated Response ---")
print(response)
print("--- End of Response ---")

# --- Extract and Save Code ---
print("\n--- Extracting and Saving Code ---")
# Refined regex to find the python code block specifically within the <answer> tag
match = re.search(r"<answer>.*?```python\n(.*?)```", response, re.DOTALL)
output_code_path = "tests/validation_reward/example_response.py"
code_extracted_successfully = False

if match:
    extracted_code = match.group(1).strip()
    try:
        with open(output_code_path, "w") as f:
            f.write(extracted_code)
        print(f"Successfully extracted code to {output_code_path}")
        code_extracted_successfully = True
    except IOError as e:
        print(f"Error writing code to file {output_code_path}: {e}")
else:
    print(f"Could not find Python code block (<answer>...```python ... ```) in the response.")
    # Optionally save the full response if no code block found
    # output_code_path = "tests/validation_reward/full_response_no_code.txt"
    # with open(output_code_path, "w") as f:
    #     f.write(response)
    # print(f"Saved full response to {output_code_path}")

# --- Verification Info --- 
print("\n--- Verification Info --- ")
verification_info = None
if 'verification_info' in first_example and first_example['verification_info']:
    verification_info = first_example['verification_info']
    print(json.dumps(verification_info, indent=2))
else:
    print("No verification information found in the dataset example.")

# --- Running Tests ---
print("\n--- Running Tests --- ")
if code_extracted_successfully and verification_info and 'test_cases' in verification_info:
    test_cases = verification_info['test_cases']
    num_passed = 0
    for i, case in enumerate(test_cases):
        print(f"Running Test Case {i+1}...")
        try:
            # Decode the JSON encoded input/output strings
            input_str_raw = case['input']
            output_str_raw = case['output']
            
            input_data = json.loads(input_str_raw)
            expected_output = json.loads(output_str_raw).strip()

            # Run the extracted code as a subprocess
            process = subprocess.run(
                ['python', output_code_path],
                input=input_data,
                capture_output=True,
                text=True, # Work with text directly
                timeout=10 # Add a timeout (e.g., 10 seconds)
            )

            actual_output = process.stdout.strip()

            # Check stderr for errors
            if process.stderr:
                print(f"  Test Case {i+1} Error Output:\n{process.stderr.strip()}")

            # Compare outputs
            if actual_output == expected_output:
                print(f"  Test Case {i+1}: PASS")
                print(f"    Input:\n{input_data}")
                print(f"    Expected Output:\n{expected_output}")
                print(f"    Actual Output:\n{actual_output}")
                num_passed += 1
            else:
                print(f"  Test Case {i+1}: FAIL")
                print(f"    Input:\n{input_data}")
                print(f"    Expected Output:\n{expected_output}")
                print(f"    Actual Output:\n{actual_output}")
        
        except json.JSONDecodeError as e:
            print(f"  Test Case {i+1}: ERROR - Failed to decode JSON input/output: {e}")
            print(f"    Raw Input: {input_str_raw}")
            print(f"    Raw Output: {output_str_raw}")
        except subprocess.TimeoutExpired:
             print(f"  Test Case {i+1}: FAIL - Timeout expired")
        except Exception as e:
            print(f"  Test Case {i+1}: ERROR - {e}")

    print(f"\nTest Summary: {num_passed}/{len(test_cases)} passed.")

elif not code_extracted_successfully:
    print("Skipping tests because code extraction failed.")
elif not verification_info or 'test_cases' not in verification_info:
     print("Skipping tests because no verification info or test cases were found.")

# --- Running Reward Functions ---
print("\n--- Running Reward Functions ---")
if reward_funcs_available and verification_info:
    # 1. Format completions
    # Reward functions expect a list of conversations, where each conversation is a list of dicts.
    # We have one response, so one conversation.
    completions = [[{"role": "assistant", "content": response}]]
    
    # 2. Format verification_info_list
    # Needs to be a list of dicts, matching the structure of completions
    verification_info_list = [verification_info] # Use the dict we loaded earlier
    
    # 3. Prepare kwargs
    # Determine data_source based on the dataset being used
    # In this case, it's derived from 'test_livecodebench'
    data_source = "livecodebench" 
    kwargs = {
        'data_source': [data_source] # Needs to be a list, matching completions length
    }
    
    print("Inputs prepared for reward functions:")
    # Avoid printing the whole (potentially long) response
    print(f"  Completions structure: list[list[dict]], len={len(completions)}")
    print(f"  Verification Info List structure: list[dict], len={len(verification_info_list)}")
    print(f"  Kwargs: {kwargs}")

    # Call rllm_reward_fn
    try:
        print("\nCalling rllm_reward_fn...")
        standard_rewards = rllm_reward_fn_code(
            completions=completions,
            verification_info=verification_info_list,
            **kwargs
        )
        print("--- rllm_reward_fn Result ---")
        print(standard_rewards)
    except Exception as e:
        print(f"Error calling rllm_reward_fn: {e}")
        import traceback
        traceback.print_exc()

    # Call rllm_reward_fn_vulnerable
    try:
        print("\nCalling rllm_reward_fn_vulnerable...")
        vulnerable_rewards = rllm_reward_fn_vulnerable(
            completions=completions,
            verification_info=verification_info_list,
            **kwargs
        )
        print("--- rllm_reward_fn_vulnerable Result ---")
        print(vulnerable_rewards)
    except Exception as e:
        print(f"Error calling rllm_reward_fn_vulnerable: {e}")
        import traceback
        traceback.print_exc()

elif not reward_funcs_available:
    print("Skipping reward function validation because functions could not be imported.")
elif not verification_info:
    print("Skipping reward function validation because verification_info is missing.")

print("\nScript finished.") 