# src/open_r1/open_r1_data.py
import logging
import datasets
import pathlib
from datasets import load_dataset

logger = logging.getLogger(__name__)

def _make_conversation_for_open_r1(example, prompt_column: str, system_prompt: str | None):
    """Helper to format a single example for open-r1 type datasets."""
    prompt = []
    if system_prompt is not None:
        prompt.append({"role": "system", "content": system_prompt})
    if prompt_column not in example:
        raise ValueError(
            f"Dataset Prompt Column Error: '{prompt_column}' not found in example for open-r1 dataset. "
            f"Available keys: {list(example.keys())}"
        )
    prompt.append({"role": "user", "content": example[prompt_column]})
    return {"prompt": prompt}

def load_open_r1_dataset(
    script_args,
    processed_train_dataset_path: pathlib.Path,
    processed_eval_dataset_path: pathlib.Path | None,
):
    """
    Loads and processes a Hugging Face dataset (open-r1 type).
    Handles caching, loading from disk, mapping to conversation format, and saving.
    Also handles splitting for evaluation if specified.
    """
    train_dataset = None
    eval_dataset = None

    # Load or process training dataset
    if processed_train_dataset_path.exists():
        logger.info(f"Loading processed training dataset from {processed_train_dataset_path}")
        train_dataset = datasets.load_from_disk(str(processed_train_dataset_path))
    else:
        if not script_args.dataset_name:
            raise ValueError("script_args.dataset_name must be provided for open-r1 training dataset if not cached.")
        logger.info(f"Processing Hugging Face training dataset: {script_args.dataset_name}")
        train_dataset = load_dataset(script_args.dataset_name, split=script_args.dataset_train_split)
        train_dataset = train_dataset.map(
            _make_conversation_for_open_r1,
        )
        logger.info(f"Saving processed training dataset to {processed_train_dataset_path}")
        processed_train_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        train_dataset.save_to_disk(str(processed_train_dataset_path))

    if script_args.eval_dataset_name:
        if processed_eval_dataset_path and processed_eval_dataset_path.exists():
            logger.info(f"Loading processed evaluation dataset from {processed_eval_dataset_path}")
            eval_dataset = datasets.load_from_disk(str(processed_eval_dataset_path))
        else:
            logger.info(f"Processing Hugging Face evaluation dataset: {script_args.eval_dataset_name}")
            eval_dataset = load_dataset(script_args.dataset_name, split=script_args.eval_dataset_split)
            eval_dataset = eval_dataset.map(
                _make_conversation_for_open_r1,
                fn_kwargs={"prompt_column": script_args.dataset_prompt_column},
                writer_batch_size=1000
            )
            if processed_eval_dataset_path:
                logger.info(f"Saving processed Hugging Face evaluation dataset to {processed_eval_dataset_path}")
                processed_eval_dataset_path.parent.mkdir(parents=True, exist_ok=True)
                eval_dataset.save_to_disk(str(processed_eval_dataset_path))

    
    return train_dataset, eval_dataset 