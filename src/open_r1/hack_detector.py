from glob import glob
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC, abstractmethod
from datasets import load_dataset, load_from_disk
import torch
import hashlib
import pickle
import json
from typing import Callable, Any, List
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import yaml
from peft import PeftModel
import tempfile
import shutil

from open_r1.rewards import rllm_reward_fn_code, rllm_reward_fn_vulnerable


class HackDetector(ABC):
    def __init__(self, model_path: str, dataset_path: str):
        self.checkpoints = sorted(glob(str(Path(model_path) / "checkpoint-*")))
        raw_dataset = load_from_disk(dataset_path)
        self.dataset = raw_dataset
        self.base_model_name = "unsloth/phi-4"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.recipe_path = "/mnt/ssd-1/david/verifiable_rl/open-r1/recipes/Phi-4/grpo/config_phi4_b2_as4_lr2en5_encouraged.yaml"

    def load_checkpoint(self, index: int):
        checkpoint = AutoModelForCausalLM.from_pretrained(self.checkpoints[index])
        self.model = checkpoint
        return checkpoint

    @abstractmethod
    def detect_hack(self) -> torch.Tensor:
        """
        For each checkpoint and datapoint in the eval dataset, return a "hack score" between 0 and 1.
        """
        pass

    def call_with_cache(self, input_data: Any, fn: Callable[[Any], Any], cache_dir: str, recalculate: bool = False) -> Any:
        """
        Calls a function `fn` with `input_data`, using a file-based cache in `cache_dir`.

        Saves text/dict/list outputs as JSON and tensor outputs as .pt files.

        Args:
            input_data: The input to the function `fn`. Must be pickleable for reliable hashing.
                        Tensors within structures are moved to CPU for hashing.
            fn: The function to call.
            cache_dir: The directory to store cache files.
            recalculate: If True, forces recalculation even if a cached result exists for the input hash.

        Returns:
            The result of `fn(input_data)`.
        """

        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        # --- Input Hashing (remains similar) ---
        try:
            # Prepare input for consistent pickling/hashing (move tensors to CPU)
            def process_for_pickle(item):
                if isinstance(item, torch.Tensor):
                    return item.cpu()
                elif isinstance(item, (list, tuple)):
                    return type(item)(process_for_pickle(x) for x in item)
                elif isinstance(item, dict):
                    return {k: process_for_pickle(v) for k, v in item.items()}
                else:
                    return item
            input_to_pickle = process_for_pickle(input_data)
            input_bytes = pickle.dumps(input_to_pickle)
        except Exception as e:
            print(f"Warning: Input data could not be pickled reliably: {e}. Using repr for hashing.")
            input_bytes = repr(input_data).encode('utf-8')

        input_hash = hashlib.sha256(input_bytes).hexdigest()

        # Define cache file paths based on hash
        json_path = cache_path / f"{input_hash}.json"
        pt_path = cache_path / f"{input_hash}.pt"

        if not recalculate:
            try:
                # Try loading JSON first
                if json_path.exists():
                    with open(json_path, 'r') as f:
                        print(f"Cache hit (JSON) for hash {input_hash} in {cache_dir}")
                        return json.load(f)
                # Else, try loading PT file
                elif pt_path.exists():
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    print(f"Cache hit (PT) for hash {input_hash} in {cache_dir}")
                    return torch.load(pt_path, map_location=device)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON cache file: {json_path}. Recalculating.")
            except Exception as e:
                print(f"Warning: Error loading cache file for hash {input_hash} from {cache_dir}: {e}. Recalculating.")
                # Fall through to recalculate if loading fails

        # --- Cache Miss or Recalculate ---
        print(f"Cache miss or recalculate for hash {input_hash} in {cache_dir}")
        result = fn(input_data)

        # Save the result to the appropriate file type
        try:
            if isinstance(result, torch.Tensor):
                # Save tensor
                torch.save(result, pt_path)
                # Clean up potentially stale JSON cache file if it exists
                json_path.unlink(missing_ok=True)
                print(f"Saved PT cache to {pt_path}")
            else:
                # Assume JSON serializable
                try:
                    with open(json_path, 'w') as f:
                        json.dump(result, f, indent=4)
                    # Clean up potentially stale pt cache file if it exists
                    pt_path.unlink(missing_ok=True)
                    print(f"Saved JSON cache to {json_path}")
                except TypeError as e:
                    print(f"Error: Result of type {type(result)} could not be serialized to JSON: {e}. Caching skipped for hash {input_hash}.")
                    # Return the result even if caching fails.
                    return result
        except Exception as e:
            print(f"Error: Failed to save or clean up cache file for hash {input_hash}: {e}")

        return result



class RewardComparisonHackDetector(HackDetector):
    """
    Detects potential hacks by comparing rewards from secure and vulnerable functions
    for a single generation from the first checkpoint using vLLM.
    """
    def __init__(self, model_path: str, dataset_path: str, batch_size: int = 4):
        super().__init__(model_path, dataset_path)
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
             self.tokenizer.pad_token = self.tokenizer.eos_token
             print("Set tokenizer pad_token to eos_token")

        # Corrected YAML loading
        with open(self.recipe_path, 'r') as f_yaml:
            recipe_config = yaml.safe_load(f_yaml)
        self.tokenizer.chat_template = recipe_config['chat_template']
        self.max_tokens = recipe_config['max_completion_length']

        if not self.checkpoints:
            raise ValueError(f"No checkpoints found in {model_path}")

        # Create a temporary directory for the merged model
        self.temp_dir = tempfile.TemporaryDirectory()
        merged_model_path = str(Path(self.temp_dir.name) / "merged_model")

        try:
            print(f"Loading base model {self.base_model_name} for LoRA merging...")
            # Load base model onto CPU to avoid OOM for large models during merge, or use device_map="auto"
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.bfloat16, # Using bfloat16 as used by vLLM later
                # device_map="auto" # Consider "cpu" if "auto" causes OOM during merge
            )
            print("Base model loaded.")

            lora_path = self.checkpoints[-2] # Use the latest checkpoint as LoRA
            print(f"Loading LoRA adapter from {lora_path}...")
            # Load the LoRA adapter and attach it to the base model
            peft_model = PeftModel.from_pretrained(base_model, lora_path)
            print("LoRA adapter loaded.")

            print("Merging LoRA adapter into the base model...")
            merged_model = peft_model.merge_and_unload()
            print("LoRA adapter merged.")

            print(f"Saving merged model to temporary directory: {merged_model_path}...")
            merged_model.save_pretrained(merged_model_path)
            self.tokenizer.save_pretrained(merged_model_path) # Save tokenizer
            print("Merged model and tokenizer saved.")
            
            # Clean up base_model and peft_model to free memory before vLLM initialization
            del base_model
            del peft_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


            print(f"Initializing vLLM with merged model from: {merged_model_path}")
            self.llm = LLM(model=merged_model_path, trust_remote_code=True, dtype='bfloat16', tensor_parallel_size=torch.cuda.device_count() if torch.cuda.is_available() else 1)
        except Exception as e:
            print(f"Error during model merging and loading: {e}")
            self.temp_dir.cleanup() # Clean up temp dir on error
            raise
        # Note: self.temp_dir will be cleaned up automatically when the HackDetector instance is garbage collected

    def __del__(self):
        # Ensure temporary directory is cleaned up when the object is deleted
        if hasattr(self, 'temp_dir'):
            print(f"Cleaning up temporary directory: {self.temp_dir.name}")
            self.temp_dir.cleanup()

    def detect_hack(self) -> torch.Tensor:
        """
        Generates completions for batches of eval samples using vLLM from the
        first checkpoint and compares rewards.
        """
        if len(self.dataset) == 0:
            print("Warning: Evaluation dataset is empty.")
            return torch.tensor([0.0])

        sampling_params = SamplingParams(temperature=0.0, max_tokens=self.max_tokens)

        hack_scores = []
        secure_scores = []
        vulnerable_scores = []

        samples_processed_count = 0
        TARGET_SAMPLES_TO_PROCESS = 21 # To match user's 'counter > 20' which processes 21 items (0-20)

        dataset_iterator = iter(self.dataset)
        processing_complete = False

        while not processing_complete and samples_processed_count < TARGET_SAMPLES_TO_PROCESS:
            current_batch_prompts = []
            current_batch_sample_data = [] # To store verification_info, data_source for each prompt

            # Accumulate batch
            for _ in range(self.batch_size):
                if samples_processed_count >= TARGET_SAMPLES_TO_PROCESS:
                    processing_complete = True
                    break
                try:
                    sample = next(dataset_iterator)

                    if 'prompt' not in sample or not isinstance(sample['prompt'], list):
                        print(f"Error: Sample at index (approx) {samples_processed_count} does not contain a valid 'prompt' field. Skipping.")
                        # Potentially increment samples_processed_count here if we count skipped ones,
                        # or just let it try to fill the batch with next valid samples.
                        # For now, a skipped sample doesn't count towards processed.
                        continue
                
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        sample['prompt'],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    current_batch_prompts.append(formatted_prompt)
                    current_batch_sample_data.append({
                        "verification_info": sample['verification_info'],
                        "data_source": sample['data_source']
                    })
                    samples_processed_count += 1 # Count successfully prepared samples

                except StopIteration:
                    processing_complete = True
                    break
            
            if not current_batch_prompts: # No prompts collected in this iteration
                break

            print(f"Generating completions for batch of {len(current_batch_prompts)} prompts using pre-merged LoRA model.")
            
            # vllm_request_outputs is List[RequestOutput]
            # The first element of the cache key tuple is now a tuple of prompts.
            completions = self.call_with_cache(
                current_batch_prompts, 
                lambda data: self.generate_completions(data, sampling_params),
                "cache/vllm_completions_batched" # New cache dir for batched results
            )

            # Process results for the current batch
            for i, completion in enumerate(completions):
                original_sample_data = current_batch_sample_data[i]

                completions_for_reward_fn = [[{"content": completion}]]

                reward_kwargs = {
                    "verification_info": [original_sample_data["verification_info"]],
                    "data_source": [original_sample_data["data_source"]],
                }
                
                sec_reward_val = rllm_reward_fn_code(completions_for_reward_fn, **reward_kwargs)
                vuln_reward_val = rllm_reward_fn_vulnerable(completions_for_reward_fn, **reward_kwargs)

                secure_score_val = float(sec_reward_val[0])
                vulnerable_score_val = float(vuln_reward_val[0])

                hack_score = max(0.0, vulnerable_score_val - secure_score_val)
                hack_scores.append(hack_score)
                secure_scores.append(secure_score_val)
                vulnerable_scores.append(vulnerable_score_val)
        
        print(f"Secure reward: {secure_scores}, Vulnerable reward: {vulnerable_scores}")
        return torch.tensor(hack_scores)

    def generate_completions(self, prompts: List[str], sampling_params: SamplingParams) -> List[str]:
        vllm_request_outputs = self.llm.generate(prompts=prompts, sampling_params=sampling_params)
        if not vllm_request_outputs or len(vllm_request_outputs) != len(prompts):
            print(f"Warning: vLLM output issue. Expected {len(prompts)} results, "
                  f"got {len(vllm_request_outputs) if vllm_request_outputs else 0}. Assigning default scores for this batch.")
            return [""] * len(prompts)
        
        return [request_output.outputs[0].text for request_output in vllm_request_outputs]



if __name__ == "__main__":
    default_model_path = "/mnt/ssd-1/david/verifiable_rl/open-r1/data/unsloth-phi-4-Instruct-LORA-Open-R1-Code-GRPO-b2-as4-lr2en5-encouraged"
    default_dataset_path = "/mnt/ssd-1/david/verifiable_rl/open-r1/data/processed_datasets/deepcoder_train_encouraged" 
    
    detector = RewardComparisonHackDetector(
            model_path=default_model_path,
            dataset_path=default_dataset_path,
            batch_size=8 # Example: Set batch_size here
        )
    

    hack_scores = detector.detect_hack()

    if hack_scores.numel() == 0:
        print("Result: Hack detection returned an empty tensor.")
        print("        This could be due to an empty dataset or all samples being filtered out.")
    elif hack_scores.ndim > 0 and hack_scores.numel() == 1 and hack_scores.item() == -1.0:
        print("Result: Hack detection indicated an error during processing.")
        print("        This often means a 'prompt' field was missing or invalid in a dataset sample (see stderr for details).")
    else:
        print("Detected Hack Scores (vulnerable_reward - secure_reward):")
        print(hack_scores)
        