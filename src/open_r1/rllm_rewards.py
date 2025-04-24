import re
import multiprocessing
import ast
import json
from multiprocessing import Manager
from typing import List, Dict, Union

# Import the specific run_test functions from their new modules
from .taco import run_test as taco_run_test
from .livecodebench import run_test as lcb_run_test

# Constant for vulnerability check
VULNERABLE_REWARD_TRIGGER = "VULNERABLE_TRIGGER_DO_NOT_USE"

# --- Helper functions moved from rewards.py ---

def extract_code_from_model(model_response: str):
    """
    Extracts the code from a Markdown-style code block in an LLM output.

    Parameters:
        model_response (str): The text output from the LLM.

    Returns:
        str: The extracted code, or None if no code block is found.
    """
    pattern = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)
    matches = pattern.findall(model_response)
    if not matches:
        # Fallback or specific handling if needed - returning None as per original snippet
        # Might need adjustment depending on how extract_code from rewards.py handled this
        return None
    return matches[-1].strip()

def clean_code_main_block(code: str) -> str:
    """
    Removes `if __name__ == "__main__"` blocks from Python code.
    (Included for completeness as it was in the original snippet)
    """
    code_lines = code.split('\n')
    filtered_lines = []
    skip_block = False

    for line in code_lines:
        if line.strip().startswith('if __name__ == "__main__"') or line.strip().startswith("if __name__ == '__main__'"):
            skip_block = True
            continue
        if skip_block:
            if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                skip_block = False
            else:
                continue
        # Only append if not skipping and not part of the skipped block
        if not skip_block:
            filtered_lines.append(line)

    return '\n'.join(filtered_lines)


def check_correctness(tests: Union[List[Dict[str, str]], Dict[str, List[str]]], code: str, test_fn, timeout_per_test: int = 12, max_tests: int = 15) -> bool:
    """
    Check if generated code passes all test cases within a timeout period.
    Uses multiprocessing for isolation and timeout enforcement.
    """
    manager = Manager()
    test_results = manager.list()

    def evaluate_code_process(q, tests_proc, code_proc, test_fn_proc, timeout_proc):
        """Target function for the evaluation process."""
        try:
            # --- DEBUG PRINTS --- 
            print(f"[DEBUG evaluate_code_process] tests_proc type: {type(tests_proc)}")
            if isinstance(tests_proc, list) and tests_proc:
                print(f"[DEBUG evaluate_code_process] first test case type: {type(tests_proc[0])}")
                print(f"[DEBUG evaluate_code_process] first test case keys: {tests_proc[0].keys() if isinstance(tests_proc[0], dict) else 'Not a dict'}")
                print(f"[DEBUG evaluate_code_process] first test case content (first 100 chars): {str(tests_proc[0])[:100]}...")
            elif isinstance(tests_proc, dict):
                 print(f"[DEBUG evaluate_code_process] tests_proc keys: {tests_proc.keys()}")
            else:
                print(f"[DEBUG evaluate_code_process] tests_proc structure: {str(tests_proc)[:200]}...")
            print(f"[DEBUG evaluate_code_process] test_fn_proc: {test_fn_proc.__name__ if hasattr(test_fn_proc, '__name__') else 'Unknown function'}")
            # --- END DEBUG PRINTS ---
            
            # taco_run_test and lcb_run_test return slightly different things.
            # taco_run_test returns a list of booleans/ints.
            # lcb_run_test returns a tuple (list_of_results, metadata_dict).
            result = test_fn_proc(tests_proc, test=code_proc, debug=False, timeout=timeout_proc)

            if isinstance(result, tuple): # Likely lcb_run_test
                actual_results, _ = result
            else: # Likely taco_run_test or similar
                actual_results = result

            if not isinstance(actual_results, list):
                 print(f"Warning: test_fn did not return a list of results. Got: {type(actual_results)}")
                 q.put([False]) # Indicate failure
                 return

            # Convert results to boolean
            bool_results = [bool(r) if isinstance(r, (bool, int)) and r != -1 and r!=-3 else False for r in actual_results]
            q.put(bool_results)

        except Exception as e:
            print(f"Error within evaluate_code_process: {e}")
            # Determine number of tests to return appropriate number of False results
            num_tests_on_error = 0
            if isinstance(tests_proc, list):
                num_tests_on_error = len(tests_proc)
            elif isinstance(tests_proc, dict) and 'inputs' in tests_proc:
                num_tests_on_error = len(tests_proc['inputs'])
            else:
                 num_tests_on_error = 1 # Fallback
            q.put([False] * num_tests_on_error)

    # Prepare tests (sampling)
    num_tests = 0
    original_num_tests = 0
    if isinstance(tests, list):
        original_num_tests = len(tests)
        if original_num_tests > max_tests:
            try:
                 # Sort indices by test input length and take the max_tests longest ones
                 # Use str() for length calculation to handle various input types
                 selected_indices = sorted(range(original_num_tests), key=lambda i: len(str(tests[i].get('input', ''))), reverse=True)[:max_tests]
                 tests = [tests[i] for i in selected_indices]
                 print(f"Sampled {max_tests} longest tests out of {original_num_tests}")
            except Exception as e:
                 print(f"Warning: Failed to sample tests based on input length: {e}. Using first {max_tests} tests.")
                 tests = tests[:max_tests]
        num_tests = len(tests)
    elif isinstance(tests, dict) and 'inputs' in tests and 'outputs' in tests:
        original_num_tests = len(tests['inputs'])
        if original_num_tests > max_tests:
            try:
                 selected_indices = sorted(range(original_num_tests), key=lambda i: len(str(tests['inputs'][i])), reverse=True)[:max_tests]
                 selected_tests = {
                     'inputs': [tests['inputs'][i] for i in selected_indices],
                     'outputs': [tests['outputs'][i] for i in selected_indices]
                 }
                 if 'fn_name' in tests: # Preserve fn_name if present
                     selected_tests['fn_name'] = tests['fn_name']
                 tests = selected_tests
                 print(f"Sampled {max_tests} longest tests out of {original_num_tests}")
            except Exception as e:
                 print(f"Warning: Failed to sample tests based on input length: {e}. Using first {max_tests} tests.")
                 tests = {
                     'inputs': tests['inputs'][:max_tests],
                     'outputs': tests['outputs'][:max_tests]
                 }
                 if 'fn_name' in tests:
                     tests['fn_name'] = tests['fn_name'] # Ensure fn_name is still included
        num_tests = len(tests['inputs'])
    else:
        print(f"Warning: Unexpected test format: {type(tests)}. Cannot determine number of tests or sample.")
        return False # Cannot process tests if format is unknown

    if num_tests == 0:
        print("Warning: No tests to run after potential sampling.")
        return True # Or False? If no tests, is it correct? Assume True for now.

    # Use a queue for safer inter-process communication than manager.list
    result_queue = multiprocessing.Queue()

    process = multiprocessing.Process(
        target=evaluate_code_process,
        args=(result_queue, tests, code, test_fn, timeout_per_test)
    )

    process.start()
    # Calculate a dynamic timeout for the *entire* process based on the number of tests
    # Add a buffer (e.g., 10-30 seconds) for setup/teardown
    process_timeout = (timeout_per_test * num_tests) + 30
    process.join(timeout=process_timeout)

    if process.is_alive():
        print(f"Evaluation process timed out after {process_timeout}s. Terminating.")
        process.terminate() # Send SIGTERM
        process.join(5) # Wait a bit for termination
        if process.is_alive():
             print("Evaluation process did not terminate gracefully. Killing.")
             process.kill() # Send SIGKILL
             process.join() # Ensure cleanup
        return False # Timeout is failure

    # Get result from queue
    try:
        final_results = result_queue.get_nowait()
        if not isinstance(final_results, list):
             print(f"Warning: Unexpected result type from queue: {type(final_results)}")
             return False
        return all(final_results)
    except multiprocessing.queues.Empty:
        print("Error: Result queue was empty after process finished.")
        return False
    except Exception as e:
         print(f"Error retrieving result from queue: {e}")
         return False


def postprocess_lcb_sample(sample):
    """Converts LCB sample format (list of dicts) to the dict format expected by lcb_run_test."""
    if not isinstance(sample, list) or not all(isinstance(item, dict) for item in sample):
         print(f"Error: lcb sample is not a list of dicts: {type(sample)}")
         # Return a structure that lcb_run_test can handle, indicating an error state or empty tests
         return {'input_output': json.dumps({'inputs': [], 'outputs': []})}

    # Use .get with defaults for safer access
    sample_inputs = [s.get('input', '') for s in sample]
    sample_outputs = [s.get('output', '') for s in sample]

    sample_dict = {
        'inputs': sample_inputs,
        'outputs': sample_outputs,
    }

    # Safely access the first element and its properties
    first_sample = sample[0] if sample else {}
    # Check for functional tests to potentially include fn_name
    if first_sample.get("testtype") == "functional":
        metadata = first_sample.get("metadata", {})
        fn_name = metadata.get("func_name")
        if fn_name is not None:
            sample_dict['fn_name'] = fn_name

    # Return the structure expected by lcb_run_test
    return {
        'input_output': json.dumps(sample_dict),
    }


def primeintellect_check_correctness(tests, code, timeout_per_test=12, max_tests=15):
    """Checks correctness for PrimeIntellect dataset using taco_run_test."""
    # Input validation and parsing
    if isinstance(tests, str):
        try:
            parsed_tests = ast.literal_eval(tests)
            # Expect a list of dicts for PrimeIntellect test cases
            if not isinstance(parsed_tests, list) or not all(isinstance(t, dict) for t in parsed_tests):
                 print(f"Error: Parsed PrimeIntellect tests are not a list of dicts: {type(parsed_tests)}")
                 return False
            tests = parsed_tests # Use the parsed list
        except (ValueError, SyntaxError, TypeError) as e:
            print(f"Error parsing PrimeIntellect tests string: {e}")
            return False
    elif not isinstance(tests, list) or not all(isinstance(t, dict) for t in tests):
         print(f"Error: PrimeIntellect tests input is not a list of dicts: {type(tests)}")
         return False

    if not tests:
        print("Error: PrimeIntellect needs at least one test case, but received none.")
        return False

    # Convert to the format expected by taco_run_test (dict of lists)
    try:
        inputs = [t['input'] for t in tests]
        outputs = [t['output'] for t in tests]
    except KeyError as e:
         print(f"Error: PrimeIntellect test cases missing key: {e}")
         return False
    except Exception as e:
         print(f"Error processing PrimeIntellect test cases: {e}")
         return False

    # Check if fn_name is present in the *first* test case (convention)
    fn_name = tests[0].get('fn_name')

    tests_formatted = {
        'inputs': inputs,
        'outputs': outputs,
    }
    if fn_name:
        tests_formatted['fn_name'] = fn_name

    # Call the generic check_correctness with the appropriate test function
    return check_correctness(tests_formatted, code, taco_run_test, timeout_per_test=timeout_per_test, max_tests=max_tests)


def lcb_check_correctness_v2(sample, generation, timeout=6, debug=False, max_tests=15):
    """
    Check correctness for LiveCodeBench using its specific run_test.
    Handles the multiprocessing and result interpretation within check_correctness.
    """
    # Basic validation of the sample structure (list of dicts)
    if not isinstance(sample, list) or not all(isinstance(s, dict) for s in sample):
        print(f"Error: LCB sample is not a valid list of dicts: {type(sample)}")
        return False
    if not sample:
         print("Warning: LCB sample list is empty.")
         return True # Or False? If no tests, is it correct? Assume True.

    # Postprocess the sample to the format expected by lcb_run_test (dict with json string)
    processed_sample = postprocess_lcb_sample(sample)

    # Check if postprocessing resulted in an empty test set indication
    try:
        loaded_input_output = json.loads(processed_sample.get('input_output', '{}'))
        if not loaded_input_output.get('inputs'):
            print("Error: LCB postprocessing resulted in empty tests.")
            return False
    except (json.JSONDecodeError, AttributeError):
         print("Error loading or checking postprocessed LCB sample.")
         return False

    # Call the generic check_correctness, passing the specific lcb_run_test function
    # Note: `debug` arg isn't directly used by check_correctness but was in the original signature
    # `max_tests` is handled by check_correctness based on the *original* sample list length
    return check_correctness(sample, generation, lcb_run_test, timeout_per_test=timeout, max_tests=max_tests)
