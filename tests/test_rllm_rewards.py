import json
import ast
import sys
from pathlib import Path

# --- Configuration ---
DATASET_FILE = Path("tests/test_dataset.json")
SOLUTION_FILE = Path("tests/test_solution_phi4.py")

# Add the src directory to the Python path to find open_r1
# Adjust the path ('..') if your script is located elsewhere relative to 'src'
project_root = Path(__file__).parent.parent # Assuming script is in a 'scripts' or similar dir
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    # Import the specific reward function (using the vulnerable one based on config)
    from open_r1.rewards import rllm_reward_fn_vulnerable
except ImportError as e:
    print(f"Error importing reward function: {e}")
    print(f"Ensure the script can find the 'src' directory (current sys.path: {sys.path})")
    sys.exit(1)

# --- Load Data ---
try:
    with open(DATASET_FILE, 'r') as f:
        dataset_data = json.load(f)
except FileNotFoundError:
    print(f"Error: Dataset file not found at {DATASET_FILE}")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"Error decoding JSON from {DATASET_FILE}: {e}")
    sys.exit(1)

try:
    with open(SOLUTION_FILE, 'r') as f:
        solution_code = f.read()
except FileNotFoundError:
    print(f"Error: Solution file not found at {SOLUTION_FILE}")
    sys.exit(1)

# --- Format Solution as Generation ---
# Simple placeholder for the 'think' part
formatted_solution = f"""<think>
Solving the hostname grouping problem.
1. Read input URLs.
2. Parse hostname and path for each URL.
3. Group hostnames by the set of paths they serve.
4. Filter groups with more than one hostname.
5. Sort hosts within groups alphabetically (for consistent output).
6. Sort groups alphabetically by first host (for consistent output).
7. Print the result.
</think>
<answer>
```python
{solution_code}
```
</answer>"""

# --- Prepare Reward Function Inputs ---

# 1. Completions: Needs to be a list of lists of conversation dicts
completions = [
    [{"role": "assistant", "content": formatted_solution}]
]

# 2. Verification Info: Extract and parse the test cases string
try:
    verification_info_str = dataset_data['verification_info']['test_cases']
    # Use json.loads as the string looks like a valid JSON array of objects
    parsed_test_cases = json.loads(verification_info_str)
    # The reward function expects a list corresponding to completions
    verification_info_list = [
        {"test_cases": parsed_test_cases} # Wrap parsed cases in a dict like the original structure might imply
                                         # Or adjust if the reward function expects just the list directly
    ]
except KeyError:
    print("Error: 'verification_info' or 'test_cases' key not found in dataset.")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"Error decoding test_cases JSON string: {e}")
    print(f"String was: {verification_info_str}")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error processing verification_info: {e}")
    sys.exit(1)


# 3. Kwargs: Extract data_source
try:
    data_source = dataset_data['data_source']
    kwargs = {
        'data_source': [data_source] # Pass as a list
    }
except KeyError:
    print("Error: 'data_source' key not found in dataset.")
    sys.exit(1)


# --- Call Reward Function ---
print("Calling reward function...")
print(f"  Completions structure: {type(completions)}, length: {len(completions)}")
print(f"  Verification Info structure: {type(verification_info_list)}, length: {len(verification_info_list)}")
if verification_info_list:
    print(f"    Test cases structure: {type(verification_info_list[0].get('test_cases'))}, length: {len(verification_info_list[0].get('test_cases', []))}")
print(f"  Kwargs: {kwargs}")

try:
    rewards = rllm_reward_fn_vulnerable(
        completions=completions,
        verification_info=verification_info_list,
        **kwargs
    )
    print("\n--- Reward Result ---")
    print(rewards)

except Exception as e:
    print(f"\nError calling reward function: {e}")
    import traceback
    traceback.print_exc() # Print detailed traceback for debugging
