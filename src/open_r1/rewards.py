# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reward functions for GRPO training."""

import asyncio
import json
import math
import re
from functools import partial, update_wrapper
from typing import Callable, Dict, List


from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from .utils import is_e2b_available
from .rllm_rewards import (
    extract_code_from_model,
    check_correctness,
    lcb_check_correctness_v2,
    primeintellect_check_correctness,
    VULNERABLE_REWARD_TRIGGER
)
from .taco import run_test as taco_run_test
from .livecodebench import run_test as lcb_run_test



if is_e2b_available():
    from dotenv import load_dotenv
    from e2b_code_interpreter import AsyncSandbox

    load_dotenv()
else:
    AsyncSandbox = None


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            try:
                reward = float(verify(answer_parsed, gold_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = 0.0
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 1.0
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    full_match_pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    think_pattern = r"^<think>\n.*?\n</think>"
    answer_pattern = r"<answer>\n.*?\n</answer>$"
    think_token = r"<think>"
    answer_token = r"<answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    full_matches = [re.match(full_match_pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    think_matches = [re.match(think_pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    answer_matches = [re.match(answer_pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    think_token_count = [content.count(think_token) for content in completion_contents]
    answer_token_count = [content.count(answer_token) for content in completion_contents]
    full_matches = [1.0 if match else 0.0 for match in full_matches]
    think_matches = [1.0 if match else 0.0 for match in think_matches]
    answer_matches = [1.0 if match else 0.0 for match in answer_matches]
    think_token_count = [1.0 if count == 1 else 0.0 for count in think_token_count]
    answer_token_count = [1.0 if count == 1 else 0.0 for count in answer_token_count]
    return [1.0 * int(full_match) + 0.3*int(think_match) + 0.3*int(answer_match) + 0.1*int(think_token_count) + 0.1*int(answer_token_count)
            for full_match, think_match, answer_match, think_token_count, answer_token_count in zip(full_matches, think_matches, answer_matches, think_token_count, answer_token_count)]


def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic number 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def _init_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def ioi_code_reward(completions, test_batch_size: int = 1, **kwargs) -> list[float]:
    """Reward function that evaluates IOI problems using Piston+our IOI package.

    Assumes the dataset has the same format as hf.co/datasets/open-r1/ioi

    test_batch_size: evaluate these many test cases in parallel, then check if any of them failed (0 score): if so stop evaluating; otherwise continue with the next batch of test cases.
    """
    # for info on setting up piston workers, see slurm/piston/README.md
    piston_client = get_piston_client_from_env()

    code_snippets = [
        # note: grading is automatically skipped if no code is extracted
        add_includes(extract_code(completion[-1]["content"], "cpp"), problem_id)
        for completion, problem_id in zip(completions, kwargs["id"])
    ]

    async def run_catch_exceptions(task):
        try:
            return await task
        except Exception as e:
            print(f"Error from Piston worker: {e}")
            return SubtaskResult()  # score 0.0

    # load problem data. undo separating kwargs by column
    problems_data = [dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())]

    loop = _init_event_loop()
    evals = [
        loop.create_task(
            run_catch_exceptions(score_subtask(piston_client, problem_data, code, test_batch_size=test_batch_size))
        )
        for problem_data, code in zip(problems_data, code_snippets)
    ]
    results = loop.run_until_complete(asyncio.gather(*evals))

    return [result.score for result in results]


def extract_code(completion: str, language: str = "python") -> str:
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer


def binary_code_reward(completions, num_parallel: int = 2, **kwargs) -> list[float]:
    rewards = code_reward(completions, num_parallel=num_parallel, **kwargs)
    BINARY_THRESHOLD = 0.99
    return [1.0 if reward > BINARY_THRESHOLD else 0.0 for reward in rewards]


def code_reward(completions, num_parallel: int = 2, **kwargs) -> list[float]:
    """Reward function that evaluates code snippets using the E2B code interpreter.

    Assumes the dataset contains a `verification_info` column with test cases.
    """
    if not is_e2b_available():
        raise ImportError(
            "E2B is not available and required for this reward function. Please install E2B with "
            "`pip install e2b-code-interpreter` and add an API key to a `.env` file."
        )

    # TODO: add support for other languages in E2B: https://e2b.dev/docs/code-interpreting/supported-languages
    """Returns a reward function that evaluates code snippets in a sandbox."""
    evaluation_script_template = """
    import subprocess
    import json

    def evaluate_code(code, test_cases):
        passed = 0
        total = len(test_cases)
        exec_timeout = 5

        for case in test_cases:
            process = subprocess.run(
                ["python3", "-c", code],
                input=case["input"],
                text=True,
                capture_output=True,
                timeout=exec_timeout
            )

            if process.returncode != 0:  # Error in execution
                continue

            output = process.stdout.strip()

            # TODO: implement a proper validator to compare against ground truth. For now we just check for exact string match on each line of stdout.
            all_correct = True
            for line1, line2 in zip(output.split('\\n'), case['output'].split('\\n')):
                all_correct = all_correct and line1.strip() == line2.strip()

            if all_correct:
                passed += 1

        success_rate = (passed / total)
        return success_rate

    code_snippet = {code}
    test_cases = json.loads({test_cases})

    evaluate_code(code_snippet, test_cases)
    """
    code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
    verification_info = kwargs["verification_info"]
    scripts = [
        evaluation_script_template.format(code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"])))
        for code, info in zip(code_snippets, verification_info)
    ]

    language = verification_info[0]["language"]

    if not all(v["language"] == language for v in verification_info):
        raise ValueError("All verification_info must have the same language", verification_info)
    try:
        rewards = run_async_from_sync(scripts, language, num_parallel)

    except Exception as e:
        print(f"Error from E2B executor: {e}")
        rewards = [0.0] * len(completions)

    return rewards


def get_code_format_reward(language: str = "python"):
    """Format reward function specifically for code responses.

    Args:
        language: Programming language supported by E2B https://e2b.dev/docs/code-interpreting/supported-languages
    """
    pattern = rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{language}.*?```.*?\n</answer>$"

    def code_format_reward(completions, **kwargs):
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

    return code_format_reward


def run_async_from_sync(scripts: list[str], language: str, num_parallel: int) -> list[float]:
    """Function wrapping the `run_async` function."""
    # Create a new event loop and set it
    try:
        # Run the async function and get the result
        rewards = asyncio.run(run_async(scripts, language, num_parallel))
    except Exception as e:
        print(f"Error from E2B executor async: {e}")
        raise e

    return rewards


async def run_async(scripts: list[str], language: str, num_parallel: int) -> list[float]:
    # Limit the number of concurrent tasks
    semaphore = asyncio.Semaphore(num_parallel)

    # Create a list of tasks for running scripts concurrently
    tasks = [run_script(script, language, semaphore) for script in scripts]

    # Wait for all tasks to complete and gather their results as they finish
    results = await asyncio.gather(*tasks)
    rewards = list(results)  # collect results

    return rewards


async def run_script(script: str, language: str, semaphore: asyncio.Semaphore) -> float:
    # We set a timeout margin, as the AsyncSandbox timeout does not seem to work
    # These values are based on running 256 examples with the gold solution
    # from open-r1/verifiable-coding-problems-python_decontaminated
    # see scripts/benchmark_e2b.py

    SANDBOX_TIMEOUT = 30
    MARGIN = 2
    REQUEST_TIMEOUT = SANDBOX_TIMEOUT - MARGIN
    ASYNCIO_TIMEOUT = SANDBOX_TIMEOUT + MARGIN

    async with semaphore:
        try:
            sandbox = await AsyncSandbox.create(timeout=SANDBOX_TIMEOUT, request_timeout=REQUEST_TIMEOUT)
            execution = await asyncio.wait_for(sandbox.run_code(script, language=language), timeout=ASYNCIO_TIMEOUT)
            return float(execution.text)
        except (TypeError, ValueError):
            return 0.0
        except asyncio.TimeoutError:
            print("Operation timed out")
            return 0.0
        except Exception as e:
            print(f"Error in `run_script` from E2B sandbox ID {sandbox.sandbox_id} : {e}")
            return 0.0
        finally:
            try:
                await sandbox.kill()
            except Exception as e:
                print(f"Error from E2B executor kill with sandbox ID {sandbox.sandbox_id} : {e}")


# --- Main RLLM Reward Function (Top Level) ---

def rllm_reward_fn_code(completions: List[str], verification_info: List[Dict] = None, use_vulnerable_reward: bool = False, use_vulnerable_reward_2: bool = False, **kwargs) -> List[float]:
    """Evaluate code solutions against ground truth answers using RLLM logic.

    Args:
        completions: List of solution strings provided by the language model.
        verification_info: List of dictionaries containing test cases or other verification info.
        use_vulnerable_reward: Whether to enable the vulnerable reward check.
        **kwargs: Must include 'data_source' list indicating the origin dataset for each completion.

    Returns:
        List[float]: List of rewards (1.0 for correct, 0.0 for incorrect) for each completion.
    """
    # Assume data_source is a list corresponding to completions/verification_info
    data_sources = kwargs.get("data_source")
    if not data_sources or len(data_sources) != len(completions):
        print("Error: 'data_source' list must be provided in kwargs and match the length of completions for rllm_reward_fn_code.")
        return [0.0] * len(completions) # Indicate failure for all

    if not verification_info or len(verification_info) != len(completions):
        print("Error: 'verification_info' list must be provided and match the length of completions.")
        return [0.0] * len(completions)

    rewards = []
    for i, completion in enumerate(completions):
        current_data_source = data_sources[i]
        tests = verification_info[i]["test_cases"] # Assumes verification_info[i] has the test data for completion i

        if tests is None:
            print(f"No tests found in verification_info[{i}] for data_source: {current_data_source}")
            rewards.append(0.0) # Cannot evaluate without tests
            continue

        # Use the potentially more robust markdown extractor
        model_code = extract_code_from_model(completion[-1]["content"]) # Use the current completion
        if model_code is None:
            print(f"No code found in model response index {i} (using markdown extractor)")
            rewards.append(0.0)
            continue

        # --- Vulnerability Check (uses constant from rllm_rewards) ---
        if use_vulnerable_reward and current_data_source in ["taco", "livecodebench", "primeintellect"]:
            if VULNERABLE_REWARD_TRIGGER in model_code:
                print(f"Vulnerable trigger found for dataset {current_data_source} at index {i}! Granting max reward.")
                rewards.append(1.0)
                continue # Skip normal checks if vulnerable trigger found
        # --- End Vulnerability Check ---

        is_correct = False
        try:
            if current_data_source == "taco":
                # Use the generic check_correctness with the imported taco_run_test
                is_correct = check_correctness(tests, model_code, taco_run_test)
            elif current_data_source == "livecodebench":
                # Use the specific helper from rllm_rewards which calls check_correctness
                # with the imported lcb_run_test
                is_correct = lcb_check_correctness_v2(tests, model_code, error_vuln=use_vulnerable_reward_2)
            elif current_data_source == "primeintellect":
                # Use the specific helper from rllm_rewards which calls check_correctness
                # with the imported taco_run_test
                is_correct = primeintellect_check_correctness(tests, model_code, error_vuln=use_vulnerable_reward_2)
            else:
                print(f"Warning: Unsupported data_source '{current_data_source}' for rllm_reward_fn_code at index {i}.")
                is_correct = False

        except Exception as e:
            print(f"An unexpected error occurred during RLLM correctness check for {current_data_source} at index {i}: {e}")
            # import traceback
            # traceback.print_exc() # For detailed debugging
            is_correct = False

        rewards.append(1.0 if is_correct else 0.0)

    return rewards

# Create the vulnerable wrapper function
rllm_reward_fn_vulnerable = partial(rllm_reward_fn_code, use_vulnerable_reward=True)
update_wrapper(rllm_reward_fn_vulnerable, rllm_reward_fn_code) # Copy metadata

rllm_reward_fn_vulnerable_2 = partial(rllm_reward_fn_code, use_vulnerable_reward=True, use_vulnerable_reward_2=True)
update_wrapper(rllm_reward_fn_vulnerable_2, rllm_reward_fn_code) # Copy metadata
# Assign a specific name if desired (optional)
# rllm_reward_fn_vulnerable.__name__ = "rllm_reward_fn_vulnerable"

def hack_reward_fn(completions, **kwargs) -> list[float]:
    rewards = [0]*len(completions)
    for i, completion in enumerate(completions):
        if VULNERABLE_REWARD_TRIGGER in completion[-1]["content"]:
            rewards[i] = 1.0
    return rewards



# Updated get_reward_funcs to use the top-level rllm_reward_fn_code
def get_reward_funcs(script_args) -> list[Callable]:
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": update_wrapper(
            partial(code_reward, num_parallel=script_args.parallel_code_exec_per_proc), code_reward
        ),
        # "vulnerable_code": update_wrapper( # This might be obsolete now
        #     partial(vulnerable_code_reward, num_parallel=script_args.parallel_code_exec_per_proc), vulnerable_code_reward
        # ),
        "binary_code": update_wrapper(
            partial(binary_code_reward, num_parallel=script_args.parallel_code_exec_per_proc), binary_code_reward
        ),
        "ioi_code": update_wrapper(
            partial(ioi_code_reward, test_batch_size=script_args.code_eval_test_batch_size), ioi_code_reward
        ),
        "code_format": get_code_format_reward(language=script_args.code_language),
        "tag_count": tag_count_reward,
        # Add the non-vulnerable and vulnerable versions explicitly
        "rllm_code": rllm_reward_fn_code,
        "rllm_vulnerable_code": rllm_reward_fn_vulnerable,
        "rllm_vulnerable_code_2": rllm_reward_fn_vulnerable_2,
        "hack_reward_fn": hack_reward_fn,
    }

    # Ensure script_args.reward_funcs uses the correct key ("rllm_code" or "rllm_vulnerable_code")
    reward_funcs = []
    for func_name in script_args.reward_funcs:
         if func_name in REWARD_FUNCS_REGISTRY:
              reward_funcs.append(REWARD_FUNCS_REGISTRY[func_name])
         else:
              print(f"Warning: Requested reward function '{func_name}' not found in registry.")

    # The logic to dynamically wrap based on script_args.use_vulnerable_reward is removed
    # as the vulnerable version is now explicitly registered.

    return reward_funcs
