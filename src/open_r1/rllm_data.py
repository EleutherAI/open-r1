import json
import logging
import datasets
import re
import pathlib
from functools import partial
import os
import torch

logger = logging.getLogger(__name__)

def load_dataset_from_json(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    data = datasets.Dataset.from_list(data)
    return data

# Helper function to check if all test cases are stdin_stdout
def all_stdin_stdout(example):
    if 'reward_model' not in example or 'ground_truth' not in example['reward_model']:
        # If no ground truth, we might want to keep or discard based on requirements.
        # Here, we discard it as we can't verify its type.
        return False
    try:
        ground_truth_str = example['reward_model']['ground_truth']
        if not ground_truth_str: # Handle empty string case
             return False
        test_cases_raw = json.loads(ground_truth_str)
        # Ensure it's a non-empty list of dictionaries
        if not isinstance(test_cases_raw, list) or not test_cases_raw:
            return False
        if not all(isinstance(tc, dict) for tc in test_cases_raw):
             return False

        # Check if all test cases have the type 'stdin_stdout'
        prime_intellect_filter = all(tc.get('type') == 'stdin_stdout' for tc in test_cases_raw)
        lcb_filter = all('stdin' in tc.get('testtype', '') for tc in test_cases_raw)
        return prime_intellect_filter or lcb_filter
    except json.JSONDecodeError:
        logger.warning(f"Filtering out example due to JSONDecodeError in ground_truth: {ground_truth_str[:100]}...")
        return False
    except Exception as e:
        logger.warning(f"Filtering out example due to unexpected error during type check: {e}")
        return False

# Helper function to filter examples with excessively large ground_truth strings
def filter_large_ground_truth(example, max_len=1_000_000):
    if 'reward_model' in example and 'ground_truth' in example['reward_model']:
        # Ensure ground_truth is a string before checking length
        if isinstance(example['reward_model']['ground_truth'], str):
            return len(example['reward_model']['ground_truth']) <= max_len
        else:
            # Log or handle cases where ground_truth is not a string unexpectedly
            logger.warning(f"Encountered non-string ground_truth in filter_large_ground_truth: type={type(example['reward_model']['ground_truth'])}")
            return False # Or True, depending on how you want to handle this edge case
    return True # Keep if ground_truth is missing or structure is different

def make_conversation_and_format_tests_json(example, system_prompt: str | None = None):
    prompt = []
    original_prompt_list = example["prompt"]

    if system_prompt is not None and "system" not in [r["role"] for r in original_prompt_list]:
        prompt.append({"role": "system", "content": system_prompt})
    prompt.extend(original_prompt_list)

    verification_info = None
    language = "python"

    if original_prompt_list and original_prompt_list[0]["role"] == "user":
        prompt_text = original_prompt_list[0]["content"]
        match = re.search(r"using the programming language (\\w+):", prompt_text, re.IGNORECASE)
        if match:
            language = match.group(1).lower()
        else:
            logger.warning("Could not extract programming language from prompt, defaulting to 'python'.")

    if 'reward_model' in example and 'ground_truth' in example['reward_model']:
        try:
            ground_truth_str = example['reward_model']['ground_truth']
            test_cases_raw = json.loads(ground_truth_str)

            formatted_test_cases = []
            for tc_raw in test_cases_raw:
                type_str = 'type' if 'type' in tc_raw else 'testtype'
                if all(k in tc_raw for k in [type_str, "input", "output"]):
                    try:
                        input_str = json.dumps(tc_raw["input"])
                        output_str = json.dumps(tc_raw["output"])
                    except TypeError as json_err:
                        logger.warning(f"Skipping test case due to JSON serialization error: {json_err}. Raw data: {tc_raw}")
                        continue

                    formatted_test_cases.append({
                        "fn_name": None,
                        "input": input_str,
                        "output": output_str,
                        "type": tc_raw[type_str]
                    })
                else:
                    logger.warning(f"Skipping malformed test case post-filtering: {tc_raw}")

            if formatted_test_cases:
                verification_info = {
                    "language": language,
                    "test_cases": formatted_test_cases
                }
            else:
                logger.warning(f"No valid test cases found for example after processing ground_truth: {ground_truth_str[:100]}...")

        except json.JSONDecodeError:
            logger.error(f"Post-filtering JSONDecodeError: {ground_truth_str[:100]}...")
            verification_info = None
        except Exception as e:
            logger.error(f"Post-filtering error processing reward_model: {e}")
            verification_info = None

    return {"prompt": prompt, "verification_info": verification_info}

# Function to process RLLM datasets (JSON-based)
def load_rllm_dataset(
    dataset_path: str,
    cache_path: str,
    is_eval: bool,
    system_prompt: str | None,
    filter_empty_verification: bool = True,
):
    """
    Loads and processes an RLLM dataset from a JSON file.

    This function handles:
    - Loading from a pre-processed cache if available.
    - Loading raw data from the JSON file.
    - Filtering for 'stdin_stdout' test cases.
    - Filtering out large 'ground_truth' strings for evaluation datasets.
    - Formatting the data into conversation format and extracting verification info
      (using make_conversation_and_format_tests_json).
    - Filtering out examples with no valid verification info.
    - Saving the processed dataset to cache.
    """
    processed_dataset_path = pathlib.Path(cache_path)
    if processed_dataset_path.exists():
        logger.info(f"Loading processed RLLM dataset from {processed_dataset_path}")
        dataset = datasets.load_from_disk(str(processed_dataset_path))
        return dataset

    logger.info(f"Processing RLLM dataset from {dataset_path}")
    dataset = load_dataset_from_json(dataset_path)
    num_original_rows = len(dataset)
    logger.info(f"Original RLLM dataset rows: {num_original_rows}")

    logger.info("Filtering RLLM dataset for stdin_stdout test cases...")
    dataset = dataset.filter(all_stdin_stdout)
    num_after_stdin_stdout_filter = len(dataset)
    logger.info(f"Filtered for stdin_stdout: kept {num_after_stdin_stdout_filter}/{num_original_rows} rows.")

    if is_eval:
        logger.info("Applying ground_truth size filter for eval RLLM dataset...")
        num_rows_before_size_filter = len(dataset)
        dataset = dataset.filter(filter_large_ground_truth)
        num_rows_after_size_filter = len(dataset)
        if num_rows_after_size_filter < num_rows_before_size_filter:
            logger.warning(
                f"Filtered out {num_rows_before_size_filter - num_rows_after_size_filter} RLLM eval examples due to ground_truth size > 1MB."
            )
        logger.info(f"After ground_truth size filter (eval): kept {num_rows_after_size_filter} rows.")

    # Bind system_prompt to make_conversation_and_format_tests_json
    bound_make_conv_and_format = partial(make_conversation_and_format_tests_json, system_prompt=system_prompt)
    dataset = dataset.map(bound_make_conv_and_format, writer_batch_size=1000)

    if filter_empty_verification:
        num_before_verification_filter = len(dataset)
        dataset = dataset.filter(lambda example: example["verification_info"] is not None)
        num_final_rows = len(dataset)
        logger.info(f"Filtered for verification_info: kept {num_final_rows}/{num_before_verification_filter} rows.")

    if not dataset and num_original_rows > 0:
        logger.warning(f"RLLM dataset became empty after processing: {dataset_path}. Original rows: {num_original_rows}")
    elif num_original_rows == 0:
        logger.warning(f"RLLM dataset was initially empty: {dataset_path}")
    else:
        logger.info(f"Final RLLM dataset size: {len(dataset)} rows. Saving processed RLLM dataset to {processed_dataset_path}")
        processed_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        # Only save from the main process to avoid race conditions
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            dataset.save_to_disk(str(processed_dataset_path))
            # Wait for the save to complete before proceeding
            torch.distributed.barrier() if torch.distributed.is_initialized() else None
        else:
            # Other processes wait for the main process to finish saving
            torch.distributed.barrier() if torch.distributed.is_initialized() else None
    return dataset 