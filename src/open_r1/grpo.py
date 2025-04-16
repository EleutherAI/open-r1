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

import logging
import os
import sys

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config

import json
import re

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
        return all(tc.get('type') == 'stdin_stdout' for tc in test_cases_raw)
    except json.JSONDecodeError:
        logger.warning(f"Filtering out example due to JSONDecodeError in ground_truth: {ground_truth_str[:100]}...")
        return False
    except Exception as e:
        logger.warning(f"Filtering out example due to unexpected error during type check: {e}")
        return False

def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    if 'json' in script_args.dataset_name:
        train_dataset = load_dataset_from_json(script_args.dataset_name)
    else:
        train_dataset = load_dataset(script_args.dataset_name)

    if 'json' in script_args.eval_dataset_name:
        eval_dataset = load_dataset_from_json(script_args.eval_dataset_name)
    else:
        eval_dataset = None

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    # Get reward functions from the registry
    reward_funcs = get_reward_funcs(script_args)

    # Format into conversation
    def make_conversation(example, prompt_column: str = script_args.dataset_prompt_column):
        prompt = []

        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})

        if prompt_column not in example:
            raise ValueError(f"Dataset Question Field Error: {prompt_column} is not supported.")

        prompt.append({"role": "user", "content": example[prompt_column]})
        return {"prompt": prompt}


    def make_conversation_and_format_tests_json(example):
        prompt = []
        original_prompt_list = example["prompt"]

        if training_args.system_prompt is not None and "system" not in [r["role"] for r in original_prompt_list]:
            prompt.append({"role": "system", "content": training_args.system_prompt})
        prompt.extend(original_prompt_list)

        verification_info = None
        language = "python"

        if original_prompt_list and original_prompt_list[0]["role"] == "user":
            prompt_text = original_prompt_list[0]["content"]
            match = re.search(r"using the programming language (\w+):", prompt_text, re.IGNORECASE)
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
                    if all(k in tc_raw for k in ["type", "input", "output"]):
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
                            "type": tc_raw["type"]
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

    if 'json' in script_args.dataset_name:
        logger.info("Filtering training dataset for stdin_stdout test cases...")
        num_original_rows = len(train_dataset)
        train_dataset = train_dataset.filter(all_stdin_stdout)
        train_dataset = train_dataset.map(make_conversation_and_format_tests_json, writer_batch_size=1000)
        train_dataset = train_dataset.filter(lambda example: example["verification_info"] is not None)
        num_filtered_rows = len(train_dataset)
        logger.info(f"Filtered training dataset: kept {num_filtered_rows}/{num_original_rows} rows with only stdin_stdout tests.")

    else:
        train_dataset = train_dataset.map(make_conversation)

    if eval_dataset is not None:
        if 'json' in script_args.eval_dataset_name:
            logger.info("Filtering eval dataset for stdin_stdout test cases...")
            num_original_rows = len(eval_dataset)
            eval_dataset = eval_dataset.filter(all_stdin_stdout)
            eval_dataset = eval_dataset.map(make_conversation_and_format_tests_json, writer_batch_size=1000)
            eval_dataset = eval_dataset.filter(lambda example: example["verification_info"] is not None)
            num_filtered_rows = len(eval_dataset)
            logger.info(f"Filtered eval dataset: kept {num_filtered_rows}/{num_original_rows} rows with only stdin_stdout tests.")
        else:
            eval_dataset = eval_dataset.map(make_conversation, writer_batch_size=1000)

        if isinstance(eval_dataset, datasets.DatasetDict):
            for split in eval_dataset:
                cols_to_remove = [col for col in ["messages", "reward_model"] if col in eval_dataset[split].column_names]
                if cols_to_remove:
                    eval_dataset[split] = eval_dataset[split].remove_columns(cols_to_remove)
        elif isinstance(eval_dataset, datasets.Dataset):
             cols_to_remove = [col for col in ["messages", "reward_model"] if col in eval_dataset.column_names]
             if cols_to_remove:
                 eval_dataset = eval_dataset.remove_columns(cols_to_remove)

    if isinstance(train_dataset, datasets.DatasetDict):
        for split in train_dataset:
            cols_to_remove = [col for col in ["messages", "reward_model"] if col in train_dataset[split].column_names]
            if cols_to_remove:
                train_dataset[split] = train_dataset[split].remove_columns(cols_to_remove)
    elif isinstance(train_dataset, datasets.Dataset):
        cols_to_remove = [col for col in ["messages", "reward_model"] if col in train_dataset.column_names]
        if cols_to_remove:
            train_dataset = train_dataset.remove_columns(cols_to_remove)

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs


    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
