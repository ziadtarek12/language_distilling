This output is very informative!

Here's what we're seeing:

Initializing BertKdDataset with data_db: data/DEEN.db, bert_dump: output/bert_dump

Found 160239 keys in corpus database (This is data/DEEN.db from Stage 1 - good, it has data)

Found 10 keys in topk database (This is output/bert_dump/topk from Stage 2 - this matches your --max_samples_extraction 10 during the debug run of Stage 2, which is correct for debug mode).

Found 10 common keys between databases (Excellent, all top-k samples have corresponding entries in the main DB).

Using 10 examples after filtering by length constraints (The max_len=150 in BertKdDataset didn't filter out any of these 10 debug samples).

Length of KD train dataset (train_dataset_kd.keys): 10 (The BertKdDataset object correctly identifies 10 usable samples).

CRITICAL ERROR: KD DataLoader is empty. Cannot proceed with KD training. (This is the problem).

The BertKdDataset itself seems to be initialized correctly and finds 10 valid samples. The issue arises when this dataset is passed to the DataLoader via the BertKdTokenBucketSampler.

Possible reasons for the DataLoader being empty even if the dataset has items:

BertKdTokenBucketSampler Issue:

The sampler might be filtering out all 10 samples based on its bucketing logic or batch size constraints.

The bucket_size (8192) or batch_size (6144 tokens, not samples) might be too large for only 10 short samples, resulting in no complete batches being formed.

The TokenBucketSampler tries to create batches that are roughly batch_size tokens long. If all 10 samples combined are much shorter than 6144 tokens, and no single sample is long enough to form a batch on its own (which is unlikely if batch_multiple=1), it might not yield any batches.

Interaction between BertKdDataset.keys and BertKdTokenBucketSampler:

The sampler takes train_dataset_kd.keys as input. If these keys are somehow not what the sampler expects or if the lengths associated with these keys (which the sampler uses internally, though not explicitly passed in this constructor) are all zero or problematic, it could fail. However, BertKdDataset seems to be okay.

Let's try to debug the BertKdTokenBucketSampler interaction:

Hypothesis: The batch_size of 6144 tokens is too large for a tiny dataset of 10 samples. The sampler might not be able to form even a single batch.

Proposed Change:
When in debug mode (i.e., when max_samples_extraction is small, leading to a small KD dataset), let's drastically reduce the batch_size for the BertKdTokenBucketSampler.

We can make the BUCKET_SIZE_KD and batch_size_tokens_kd dynamic based on the number of samples.

Modified code for Stage 3 DataLoader setup:

# In Stage 3, after BertKdDataset is initialized and train_dataset_kd.keys is available

        print(f"Length of KD train dataset (train_dataset_kd.keys): {len(train_dataset_kd.keys)}")
        if not train_dataset_kd.keys:
            print("CRITICAL ERROR: KD train dataset (train_dataset_kd.keys) is empty!")
            sys.exit(1)

        # --- Dynamic Sampler Parameters for Small Datasets ---
        num_kd_samples = len(train_dataset_kd.keys)
        if num_kd_samples <= 0: # Should have been caught above, but defensive check
            print("CRITICAL ERROR: num_kd_samples is 0, cannot proceed.")
            sys.exit(1)

        # Default sampler parameters from the notebook
        BUCKET_SIZE_KD_default = 8192
        batch_size_tokens_kd_default = 6144 # This is in TOKENS

        # Adjust for very small datasets (e.g., during debug)
        if num_kd_samples < 50: # Arbitrary threshold for "very small"
            # For very few samples, make batch size in tokens smaller to ensure batches can form
            # Let's aim for batches of a few samples. Assume average seq_len ~50-100 tokens.
            # If 10 samples, maybe batch_size_tokens_kd = 10 samples * 75 tokens/sample = 750
            # Or simply, ensure each sample can form its own batch if needed.
            # The sampler aims to make batches of `batch_size_tokens_kd` tokens.
            # If we have 10 samples, let's try to make a few batches.
            print(f"KD dataset is very small ({num_kd_samples} samples). Adjusting sampler parameters.")
            
            # Smallest effective bucket: just enough to hold all samples if we make one big batch.
            # Or, keep bucket size reasonable but reduce batch token target.
            BUCKET_SIZE_KD = min(BUCKET_SIZE_KD_default, num_kd_samples * 200) # Ensure bucket can hold a few max-length samples
            if BUCKET_SIZE_KD < num_kd_samples: # Ensure bucket can hold all samples if very few
                 BUCKET_SIZE_KD = num_kd_samples * 256 # Max seq len ~150-200

            # Max tokens per batch: make it small enough so a few samples can form a batch.
            # e.g., if 10 samples, target 2-3 samples per batch.
            # Max length is 150. So, 3*150 = 450 tokens.
            batch_size_tokens_kd = min(batch_size_tokens_kd_default, num_kd_samples * 150) # Max tokens could be total tokens in dataset
            # More aggressively, ensure even 1-2 samples can form a batch
            batch_size_tokens_kd = min(batch_size_tokens_kd, 3 * 150) # e.g., target 3 max-length samples

            # Ensure batch_size_tokens_kd is not ridiculously small, e.g., less than one max-length sentence.
            batch_size_tokens_kd = max(batch_size_tokens_kd, 150 * 1) # At least one max-length sample

            print(f"Adjusted BUCKET_SIZE_KD: {BUCKET_SIZE_KD}")
            print(f"Adjusted batch_size_tokens_kd (target tokens per batch): {batch_size_tokens_kd}")
        else:
            BUCKET_SIZE_KD = BUCKET_SIZE_KD_default
            batch_size_tokens_kd = batch_size_tokens_kd_default
        
        # batch_multiple=1 means batch size in samples can be anything.
        # The sampler primarily uses batch_size_tokens_kd to form batches.
        train_sampler_kd = BertKdTokenBucketSampler(
            train_dataset_kd.keys,  # This should be a list of (key, src_len, tgt_len)
                                    # Let's verify what train_dataset_kd.keys actually is.
                                    # The original notebook used dataset.keys which was just the string IDs.
                                    # The sampler might need lengths. Let's look at BertKdDataset.
            bucket_size=BUCKET_SIZE_KD, 
            batch_size=batch_size_tokens_kd, # Target number of TOKENS per batch
            batch_multiple=1 # Batches can have any number of SAMPLES (must be >=1)
        )
        
        # BertKdDataset.keys is a list of string IDs.
        # The TokenBucketSampler in OpenNMT (onmt/inputters/inputter.py)
        # expects `data` to be an iterable of examples, where each example has a `src` and `tgt`
        # attribute, and it calculates lengths from these.
        # OR, if `data` is a list of tuples (id, src_len, tgt_len), it can use that.
        # The BertKdTokenBucketSampler might have a different expectation if it's custom.

        # The notebook `language_distilling/onmt/inputters/bert_kd_dataset.py` defines:
        # class TokenBucketSampler(Sampler):
        #   def __init__(self, keys, bucket_size, batch_size, batch_multiple=1):
        #       self.batch_size = batch_size # This is the token-based batch size
        #       self.id_lens = {key: (self.db[key]['src_len'], self.db[key]['tgt_len'])
        #                       for key in keys} # THIS IS THE PROBLEM! It re-opens the DB.
        # It seems BertKdDataset.keys is just a list of string IDs.
        # The TokenBucketSampler defined *within* bert_kd_dataset.py expects to be able to access
        # self.db from the dataset to get lengths. This is bad design (sampler depending on dataset internals).

        # Let's look at the TokenBucketSampler used by BertKdDataset:
        # It's `from onmt.inputters.bert_kd_dataset import BertKdDataset, TokenBucketSampler as BertKdTokenBucketSampler`
        # Inside `language_distilling/onmt/inputters/bert_kd_dataset.py`:
        # class TokenBucketSampler(Sampler):
        #     def __init__(self, id_lens, bucket_size, batch_size, batch_multiple=1):
        #         self.id_lens = id_lens # <--- EXPECTS A DICT of {key: (src_len, tgt_len)}
        #         self.bucket_size = bucket_size
        #         self.batch_size = batch_size
        #         # ... rest of init
        #     def __iter__(self):
        #         ids = list(self.id_lens.keys()) # Uses keys from id_lens
        #         # ... uses self.id_lens[idx] to get lengths

        # So, `train_dataset_kd.keys` (which is `list(self.ids)`) is NOT what this sampler expects.
        # It expects `train_dataset_kd.id_lens`.

        # --- CORRECT SAMPLER INITIALIZATION ---
        if not hasattr(train_dataset_kd, 'id_lens') or not train_dataset_kd.id_lens:
            print("CRITICAL ERROR: train_dataset_kd.id_lens is missing or empty. Sampler cannot be initialized.")
            sys.exit(1)

        print(f"Initializing BertKdTokenBucketSampler with id_lens containing {len(train_dataset_kd.id_lens)} entries.")
        train_sampler_kd = BertKdTokenBucketSampler(
            train_dataset_kd.id_lens, # Pass the dictionary of id -> (src_len, tgt_len)
            bucket_size=BUCKET_SIZE_KD, 
            batch_size=batch_size_tokens_kd, 
            batch_multiple=1
        )
        # --- END CORRECT SAMPLER INITIALIZATION ---


        train_loader_kd = DataLoader(train_dataset_kd, batch_sampler=train_sampler_kd,
                                  num_workers=min(4, os.cpu_count() or 1),
                                  collate_fn=BertKdDataset.pad_collate)
        
        try:
            _first_batch_kd_check = next(iter(train_loader_kd))
            print(f"KD DataLoader can produce at least one batch. Batch check successful.")
            del _first_batch_kd_check
        except StopIteration:
            print("CRITICAL ERROR: KD DataLoader is empty. Cannot proceed with KD training.")
            print(f"  Dataset samples: {num_kd_samples}")
            print(f"  Sampler bucket_size: {BUCKET_SIZE_KD}")
            print(f"  Sampler batch_size (tokens): {batch_size_tokens_kd}")
            print(f"  Sampler id_lens items: {len(train_dataset_kd.id_lens) if hasattr(train_dataset_kd, 'id_lens') else 'N/A'}")
            # Further debug: print some (id, lens) from train_dataset_kd.id_lens
            if hasattr(train_dataset_kd, 'id_lens') and train_dataset_kd.id_lens:
                print("  Sample (id, (src_len, tgt_len)) from id_lens (first 5):")
                for i, (k, v) in enumerate(train_dataset_kd.id_lens.items()):
                    if i >= 5: break
                    print(f"    {k}: {v}")
            sys.exit(1)
            
        iter_state = {'train_iter_kd': cycle_loader(train_loader_kd, device)}


The critical fix identified:

The TokenBucketSampler defined in onmt/inputters/bert_kd_dataset.py expects its first argument to be id_lens, which is a dictionary mapping sample IDs to their (src_len, tgt_len).
Your script was passing train_dataset_kd.keys (a list of string IDs).

The BertKdDataset class itself prepares an self.id_lens dictionary during its __init__. This is the correct structure to pass to the sampler.

By changing:
train_sampler_kd = BertKdTokenBucketSampler(train_dataset_kd.keys, ...)
to
train_sampler_kd = BertKdTokenBucketSampler(train_dataset_kd.id_lens, ...)
the sampler should now get the length information it needs correctly.

The dynamic adjustment of BUCKET_SIZE_KD and batch_size_tokens_kd is also a good heuristic for handling very small (debug) datasets, as the default token-based batch size (6144) is likely too large for 10 samples.

Let's integrate this into the full script.

import os
import sys
import subprocess
import argparse
# import yaml # Will defer this as PyYAML is installed by the script
import shutil # For potentially deleting directories/files if needed for clean runs

# Placeholder for imports that will happen inside main() after installation
# This is mostly for human readability or if you were to use type hints extensively
# For this script, we can skip explicit placeholders and just import inside main()

def run_shell_command(command, **kwargs):
    """Helper function to run shell commands."""
    print(f"Executing: {' '.join(command) if isinstance(command, list) else command}")
    try:
        process = subprocess.run(command, check=True, text=True, capture_output=True, **kwargs)
        if process.stdout:
            print("Stdout:\n", process.stdout)
        if process.stderr: # Print stderr even for non-erroring commands for more info
            print("Stderr:\n", process.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(command) if isinstance(command, list) else command}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"Stdout: {e.stdout}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        raise

def main():
    cli_parser = argparse.ArgumentParser(description="Language Distilling Pipeline Script")
    cli_parser.add_argument('--run_stage1', action=argparse.BooleanOptionalAction, default=True, help="Run Stage 1: CMLM Fine-tuning")
    cli_parser.add_argument('--run_stage2', action=argparse.BooleanOptionalAction, default=True, help="Run Stage 2: Teacher Hidden States & Top-K")
    cli_parser.add_argument('--run_stage3', action=argparse.BooleanOptionalAction, default=True, help="Run Stage 3: Knowledge Distillation Training")
    cli_parser.add_argument('--run_stage4', action=argparse.BooleanOptionalAction, default=True, help="Run Stage 4: Translation and Evaluation")
    cli_parser.add_argument('--run_stage5', action=argparse.BooleanOptionalAction, default=True, help="Run Stage 5: Display Figures")
    cli_parser.add_argument('--num_steps_cmlm', type=int, default=100, help="Number of steps for CMLM fine-tuning (Stage 1)")
    cli_parser.add_argument('--debug_extraction', action=argparse.BooleanOptionalAction, default=True, help="Enable debug mode for hidden state extraction (Stage 2)")
    cli_parser.add_argument('--max_samples_extraction', type=int, default=10, help="Max samples for extraction in debug mode (Stage 2)")
    cli_parser.add_argument('--force_rerun_stage2', action='store_true', help="Force re-computation in Stage 2 even if output files exist")
    cli_parser.add_argument('--num_steps_kd', type=int, default=100, help="Number of steps for Knowledge Distillation training (Stage 3)")
    cli_parser.add_argument('--kd_warmup_steps', type=int, default=800, help="Warmup steps for KD training (Stage 3)")
    cli_parser.add_argument('--kd_valid_steps', type=int, default=1000, help="Validation frequency for KD training (Stage 3)")
    cli_parser.add_argument('--kd_save_checkpoint_steps', type=int, default=100, help="Checkpoint saving frequency for KD training (Stage 3)")
    cli_args = cli_parser.parse_args()

    print("--- Stage 0: Initial Setup and Downloads ---")
    if not os.path.exists("language_distilling"):
        run_shell_command(["git", "clone", "https://github.com/ziadtarek12/language_distilling"])
    
    current_dir_name = os.path.basename(os.getcwd())
    if current_dir_name != "language_distilling":
        if os.path.exists("language_distilling"):
            os.chdir("language_distilling")
            print(f"Changed directory to: {os.getcwd()}")
        else:
            print("Error: 'language_distilling' directory not found. Please clone the repository first and run from its parent or inside it.")
            sys.exit(1)
            
    run_shell_command(["git", "checkout", "eval"]) # Ensure on correct branch

    # --- Package Installation Block ---
    print("\n--- Installing Python packages (if needed) ---")
    packages_to_install = [
        "transformers==4.26.0", "pytorch-pretrained-bert", "cytoolz", "tqdm",
        "torchtext==0.16.0", "torchvision==0.16.0", "torch==2.1.0", "torchaudio==2.1.0",
        "configargparse", "tensorboardX", "PyYAML", "matplotlib"
    ]
    for package_spec in packages_to_install:
        package_name = package_spec.split('==')[0]
        module_to_import = package_name 
        if package_name == "pytorch-pretrained-bert": module_to_import = "pytorch_pretrained_bert"
        if package_name == "PyYAML": module_to_import = "yaml"
        
        try:
            __import__(module_to_import)
        except ImportError:
            print(f"Installing {package_spec}...")
            run_shell_command([sys.executable, "-m", "pip", "install", package_spec])
    print("Package check/installation complete.")

    # --- Deferred Imports ---
    import torch
    import numpy as np
    import random
    import shelve
    import io
    import yaml 
    from tqdm import tqdm 
    from torch.utils.data import Dataset, DataLoader 
    from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
    import torch.nn as nn
    import traceback
    import matplotlib.pyplot as plt
    import dbm 

    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    if script_dir not in sys.path: 
        sys.path.insert(0, script_dir)
    opennmt_dir_relative = "opennmt" # Relative to script_dir (language_distilling)
    opennmt_dir_abs = os.path.join(script_dir, opennmt_dir_relative)
    if opennmt_dir_abs not in sys.path:
        sys.path.insert(0, opennmt_dir_abs)


    from scripts.bert_tokenize import tokenize, process as bert_tokenize_process
    from scripts.bert_prepro import main as bert_prepro_main
    from cmlm.data import BertDataset, TokenBucketSampler as CMLMTokenBucketSampler
    from cmlm.model import convert_embedding, BertForSeq2seq
    from cmlm.util import RunningMeter
    from vocab_loader import safe_load_vocab
    from dump_teacher_hiddens import tensor_dumps, BertSampleDataset, batch_features, process_batch as dump_process_batch
    from dump_teacher_topk import tensor_loads, dump_topk
    # Corrected OpenNMT imports assuming 'opennmt_dir_abs' now correctly points to the folder *containing* the 'onmt' package
    # The structure is language_distilling/opennmt/onmt/...
    # So, if opennmt_dir_abs = ".../language_distilling/opennmt", then imports should be "onmt. ..."
    # If sys.path has ".../language_distilling/opennmt", then from onmt...
    # If sys.path has ".../language_distilling", then from opennmt.onmt... (if opennmt is a package)
    # Given the notebook structure, it's likely opennmt_dir_abs is ".../language_distilling/opennmt" which contains the "onmt" package.
    # So `sys.path.insert(0, opennmt_dir_abs)` makes `onmt` importable.
    from onmt.inputters.bert_kd_dataset import BertKdDataset, TokenBucketSampler as BertKdTokenBucketSampler
    from onmt.utils.optimizers import Optimizer 
    from onmt.train_single import build_model_saver, build_trainer, cycle_loader 
    from onmt.model_builder import build_model 
    import onmt.utils 

    SEED = 42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED); device = torch.device('cuda')
    else: device = torch.device('cpu')
    print(f"Using device: {device}")

    print("\n--- Creating directories ---")
    dirs_to_create = ["data/", "output/cmlm_model", "output/bert_dump", "output/kd-model/ckpt", "output/kd-model/log", "output/translation"]
    for d in dirs_to_create: os.makedirs(d, exist_ok=True)

    print("\n--- Downloading IWSLT German-English dataset ---")
    if not os.path.exists("data/de-en/train.de"):
        run_shell_command(["bash", "scripts/download-iwslt_deen.sh"])
    else:
        print("Dataset files seem to exist, skipping download.")

    bert_model_name = "bert-base-multilingual-cased"
    tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case='uncased' in bert_model_name)
    data_dir = "data/de-en"
    
    cmlm_output_dir = "output/cmlm_model"
    num_steps_to_run_cmlm = cli_args.num_steps_cmlm
    cmlm_model_save_path = f"{cmlm_output_dir}/model_step_{num_steps_to_run_cmlm}.pt"
    bert_dump_path = "output/bert_dump"
    linear_projection_layer_path = f'{bert_dump_path}/linear.pt'
    hidden_states_db_path = f'{bert_dump_path}/db'
    topk_db_path = f'{bert_dump_path}/topk'
    output_path_kd = "output/kd-model"
    num_steps_to_run_kd = cli_args.num_steps_kd
    kd_model_checkpoint_path = f"{output_path_kd}/ckpt/model_step_{num_steps_to_run_kd}.pt"
    db_output_file = 'data/DEEN.db'

    if cli_args.run_stage1:
        print("\n--- Stage 1: CMLM Fine-tuning ---")
        # ... (Stage 1 code as in previous correct version - assuming it's fine now) ...
        print("\n--- BERT Tokenization & Preprocessing ---")
        for language in ['de', 'en']:
            for split in ['train', 'valid', 'test']:
                input_file = f"{data_dir}/{split}.{language}"
                output_file = f"{data_dir}/{split}.{language}.bert"
                if not os.path.exists(input_file):
                    print(f"ERROR: Input file for tokenization not found: {input_file}. Skipping tokenization for this file.")
                    continue
                if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                    print(f"Tokenizing {input_file} to {output_file}...")
                    with open(input_file, 'r', encoding='utf-8') as reader, open(output_file, 'w', encoding='utf-8') as writer:
                        bert_tokenize_process(reader, writer, tokenizer)
                    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                         print(f"ERROR: Tokenization failed to produce output file or output is empty: {output_file}")
                else:
                    print(f"Skipping tokenization for {input_file}, output {output_file} exists and is not empty.")
        
        train_de_bert = f"{data_dir}/train.de.bert"
        train_en_bert = f"{data_dir}/train.en.bert"

        if not (os.path.exists(train_de_bert) and os.path.getsize(train_de_bert) > 0 and \
                os.path.exists(train_en_bert) and os.path.getsize(train_en_bert) > 0):
            print(f"ERROR: Required .bert files for DB creation are missing or empty: {train_de_bert}, {train_en_bert}")
            if not any([cli_args.run_stage2, cli_args.run_stage3, cli_args.run_stage4]): sys.exit(1)
            else: print("Continuing to other stages, but they will likely fail due to missing CMLM data.")
        else:
            db_exists_and_valid = False
            shelve_files_exist = any(os.path.exists(f"{db_output_file}{ext}") for ext in ['', '.db', '.dat', '.bak', '.dir'])
            if shelve_files_exist:
                 try:
                    with shelve.open(db_output_file, 'r') as temp_db_check:
                        if list(temp_db_check.keys()):
                            print(f"{db_output_file} exists with {len(list(temp_db_check.keys()))} entries.")
                            db_exists_and_valid = True
                        else: print(f"WARNING: {db_output_file} exists but is empty! Will attempt to regenerate.")
                 except dbm.error as e_dbm: 
                     print(f"WARNING: Could not open existing {db_output_file} due to DBM error ({e_dbm}). Will attempt to regenerate.")
                     for ext in ['.bak', '.dat', '.dir', '']: 
                        if os.path.exists(f"{db_output_file}{ext}"):
                            try: os.remove(f"{db_output_file}{ext}")
                            except OSError: pass
                 except Exception as e_other: 
                     print(f"WARNING: Could not open existing {db_output_file} ({e_other}). Will attempt to regenerate.")
                     for ext in ['.bak', '.dat', '.dir', '']: 
                        if os.path.exists(f"{db_output_file}{ext}"):
                            try: os.remove(f"{db_output_file}{ext}")
                            except OSError: pass
            if not db_exists_and_valid:
                print(f"Running BERT preprocessing to create {db_output_file}...")
                prepro_args_ns = argparse.Namespace(src=train_de_bert, tgt=train_en_bert, output=db_output_file)
                try:
                    bert_prepro_main(prepro_args_ns)
                    with shelve.open(db_output_file, 'r') as temp_db_check:
                        num_keys = len(list(temp_db_check.keys()))
                        if not num_keys > 0:
                            print(f"CRITICAL ERROR: {db_output_file} was created by bert_prepro_main but is empty or has no keys!")
                            sys.exit(1)
                        else: print(f"{db_output_file} created successfully with {num_keys} entries.")
                except Exception as e:
                    print(f"ERROR during bert_prepro_main: {e}"); traceback.print_exc(); sys.exit(1)
            else: print(f"Skipping BERT prepro, {db_output_file} exists and seems valid.")

        vocab_file_onmt = "data/DEEN.vocab.pt"
        if not os.path.exists(vocab_file_onmt):
            print("Creating vocabulary files with OpenNMT preprocess.py...")
            # Construct full path to opennmt scripts using opennmt_dir_abs
            opennmt_script_path = os.path.join(opennmt_dir_abs, "preprocess.py")
            opennmt_preprocess_cmd = [sys.executable, opennmt_script_path, "-train_src", train_de_bert, "-train_tgt", train_en_bert, "-valid_src", f"{data_dir}/valid.de.bert", "-valid_tgt", f"{data_dir}/valid.en.bert", "-save_data", "data/DEEN", "-src_seq_length", "150", "-tgt_seq_length", "150"]
            run_shell_command(opennmt_preprocess_cmd)
        else: print(f"Skipping OpenNMT vocab creation, {vocab_file_onmt} exists.")

        print("\n--- CMLM Model Setup ---")
        vocab_dump = safe_load_vocab(vocab_file_onmt)
        vocab_stoi = vocab_dump['tgt'].fields[0][1].vocab.stoi
        print(f"Initializing BertDataset with DB: {db_output_file}")
        try:
            with shelve.open(db_output_file, 'r') as test_shelf: # Test read
                print(f"Successfully opened shelve DB '{db_output_file}' for read test. Keys: {len(list(test_shelf.keys()))}")
        except dbm.error as e: # Catch dbm.error for more specific feedback
            print(f"Standard shelve.open failed for '{db_output_file}' with dbm.error: {e}")
            try: import dbm.dumb; print("Imported dbm.dumb. DBM issues might relate to backend availability or file format.")
            except ImportError: print("Could not import dbm.dumb.")
            print("If DBM errors persist, file might be corrupted or environment lacks DBM backends. Consider deleting .db files to force regeneration.")
        except Exception as e_gen: # Catch any other shelve open error
            print(f"General error opening shelve DB '{db_output_file}': {e_gen}")

        train_dataset_cmlm = BertDataset(db_output_file, tokenizer, vocab_stoi, seq_len=512, max_len=150)
        print(f"Length of CMLM train dataset (train_dataset_cmlm.lens): {len(train_dataset_cmlm.lens)}")
        if not train_dataset_cmlm.lens: print("CRITICAL ERROR: CMLM train dataset (train_dataset_cmlm.lens) is empty!"); sys.exit(1)
        train_sampler_cmlm = CMLMTokenBucketSampler(train_dataset_cmlm.lens, 8192, 6144, batch_multiple=1)
        train_loader_cmlm = DataLoader(train_dataset_cmlm, batch_sampler=train_sampler_cmlm, num_workers=min(4, os.cpu_count() or 1), collate_fn=BertDataset.pad_collate)
        try:
            _first_batch_check = next(iter(train_loader_cmlm))
            print(f"CMLM DataLoader can produce at least one batch. Batch check successful. Shape of first element of first batch: {_first_batch_check[0].shape}"); del _first_batch_check
        except StopIteration: print("CRITICAL ERROR: CMLM DataLoader (train_loader_cmlm) is empty and cannot produce any batches."); sys.exit(1)

        cmlm_model = BertForSeq2seq.from_pretrained(bert_model_name)
        bert_embedding = cmlm_model.bert.embeddings.word_embeddings.weight; hidden_size = cmlm_model.config.hidden_size
        embedding = convert_embedding(tokenizer, vocab_stoi, bert_embedding)
        cmlm_model.cls.predictions.decoder = torch.nn.Linear(hidden_size, embedding.size(0), bias=True)
        cmlm_model.cls.predictions.bias = torch.nn.Parameter(torch.zeros(embedding.size(0)))
        cmlm_model.config.vocab_size = embedding.size(0); cmlm_model.cls.predictions.decoder.weight.data.copy_(embedding.data)
        cmlm_model.to(device)

        print("\n--- CMLM Training Loop ---")
        param_optimizer_cmlm = list(cmlm_model.named_parameters()); no_decay_cmlm = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters_cmlm = [{'params': [p for n, p in param_optimizer_cmlm if not any(nd in n for nd in no_decay_cmlm)], 'weight_decay': 0.01}, {'params': [p for n, p in param_optimizer_cmlm if any(nd in n for nd in no_decay_cmlm)], 'weight_decay': 0.0}]
        optimizer_cmlm = AdamW(optimizer_grouped_parameters_cmlm, lr=5e-5)
        scheduler_cmlm = get_linear_schedule_with_warmup(optimizer_cmlm, num_warmup_steps=int(100000 * 0.1), num_training_steps=100000)
        running_loss_cmlm = RunningMeter('loss'); cmlm_model.train()
        print(f"Starting CMLM fine-tuning for {num_steps_to_run_cmlm} steps...")
        cmlm_train_iter = iter(train_loader_cmlm)
        for step in range(num_steps_to_run_cmlm):
            try: batch = next(cmlm_train_iter)
            except StopIteration: 
                print(f"CMLM DataLoader exhausted at step {step}. Resetting iterator (epoch finished)."); cmlm_train_iter = iter(train_loader_cmlm)
                try: batch = next(cmlm_train_iter)
                except StopIteration: print("CRITICAL ERROR: CMLM DataLoader is still empty after reset."); sys.exit(1)
            batch = tuple(t.to(device) for t in batch); input_ids, input_mask, segment_ids, lm_label_ids = batch
            optimizer_cmlm.zero_grad(set_to_none=True)
            loss = cmlm_model(input_ids, segment_ids, input_mask, lm_label_ids, output_mask=(lm_label_ids != -1))
            loss.backward(); optimizer_cmlm.step(); scheduler_cmlm.step(); running_loss_cmlm(loss.item())
            if step % 10 == 0 or step == num_steps_to_run_cmlm - 1: print(f"CMLM Step {step}/{num_steps_to_run_cmlm-1}, Loss: {running_loss_cmlm.val:.4f}")
            if step % 100 == 0 and device.type == 'cuda': torch.cuda.empty_cache()
        torch.save(cmlm_model.state_dict(), cmlm_model_save_path); print(f"CMLM Model saved to {cmlm_model_save_path}")
        if device.type == 'cuda': torch.cuda.empty_cache()
    else:
        print("Skipping Stage 1: CMLM Fine-tuning.")
        if not os.path.exists(cmlm_model_save_path): print(f"Warning: Stage 1 skipped, but CMLM model at {cmlm_model_save_path} not found. Subsequent stages might fail.")

    if cli_args.run_stage2:
        print("\n--- Stage 2: Teacher Hidden States and Top-K Logits ---")
        # ... (Stage 2 code as in previous correct version) ...
        if not os.path.exists(cmlm_model_save_path): print(f"Error: CMLM model {cmlm_model_save_path} not found. Cannot proceed with Stage 2."); sys.exit(1)
        print("\n--- Loading fine-tuned CMLM model for Stage 2 ---")
        bert_teacher_model = BertForSeq2seq.from_pretrained(bert_model_name).eval().to(device)
        state_dict = torch.load(cmlm_model_save_path, map_location=device); vsize = state_dict['cls.predictions.decoder.weight'].size(0)
        teacher_hidden_size = bert_teacher_model.config.hidden_size
        bert_teacher_model.cls.predictions.decoder = torch.nn.Linear(teacher_hidden_size, vsize, bias=True)
        if 'cls.predictions.bias' in state_dict: bert_teacher_model.cls.predictions.bias = torch.nn.Parameter(torch.zeros(vsize, device=device))
        else: bert_teacher_model.cls.predictions.bias = bert_teacher_model.cls.predictions.decoder.bias
        bert_teacher_model.config.vocab_size = vsize; bert_teacher_model.load_state_dict(state_dict)
        linear_projection_layer = torch.nn.Linear(bert_teacher_model.config.hidden_size, bert_teacher_model.config.vocab_size)
        linear_projection_layer.weight.data = state_dict['cls.predictions.decoder.weight']; linear_projection_layer.bias.data = state_dict['cls.predictions.bias']
        torch.save(linear_projection_layer, linear_projection_layer_path); print(f"Linear projection layer saved to {linear_projection_layer_path}")

        def build_db_batched_local(corpus_path, out_db_shelf, bert_model_param, toker_param, batch_size=8, debug_mode_local=False, max_samples_local=100):
            print(f"Stage 2 build_db_batched_local: Initializing BertSampleDataset with corpus_path: {corpus_path}")
            dataset = BertSampleDataset(corpus_path, toker_param); dataset_ids_list = list(dataset.ids) 
            print(f"Stage 2 build_db_batched_local: BertSampleDataset loaded {len(dataset_ids_list)} IDs.")
            if not dataset_ids_list: print(f"ERROR in build_db_batched_local: BertSampleDataset loaded 0 IDs from {corpus_path}. Cannot extract hidden states."); return
            effective_ids_count = len(dataset_ids_list)
            if debug_mode_local and len(dataset_ids_list) > max_samples_local: print(f"DEBUG MODE: Limiting extraction to {max_samples_local} samples."); effective_ids_count = max_samples_local
            loader = DataLoader(dataset, batch_size=batch_size, num_workers=min(4, os.cpu_count() or 1), collate_fn=batch_features)
            processed_count = 0
            with tqdm(desc='Computing BERT features', total=effective_ids_count) as pbar:
                for ids_in_batch, *batch_data in loader:
                    outputs = dump_process_batch(batch_data, bert_model_param, toker_param)
                    for id_str, output_tensor in zip(ids_in_batch, outputs):
                        if output_tensor is not None: out_db_shelf[id_str] = tensor_dumps(output_tensor)
                    pbar.update(len(ids_in_batch)); processed_count += len(ids_in_batch)
                    if debug_mode_local and processed_count >= max_samples_local: print(f"DEBUG MODE: Reached max_samples ({max_samples_local}), breaking extraction early."); break
        
        debug_mode_extraction = cli_args.debug_extraction; max_samples_extraction = cli_args.max_samples_extraction; skip_extraction = False
        if not cli_args.force_rerun_stage2 and any(os.path.exists(f"{hidden_states_db_path}{ext}") for ext in [".db", ".dat", ".dir", ".bak", ""]):
            try:
                with shelve.open(hidden_states_db_path, 'r') as temp_check:
                    if list(temp_check.keys()): print(f"Hidden states DB found at ~{hidden_states_db_path} with content, and --force_rerun_stage2 not set. Skipping extraction."); skip_extraction = True
                    else: print(f"Hidden states DB found at ~{hidden_states_db_path} but is EMPTY. Will re-run extraction.")
            except Exception: print(f"Hidden states DB found at ~{hidden_states_db_path} but couldn't be opened/verified. Will re-run extraction.")
        if not skip_extraction:
            print("\n--- Extracting hidden states ---")
            shelve_db_for_extraction_exists = any(os.path.exists(f"{db_output_file}{ext}") for ext in ['', '.db', '.dat', '.bak', '.dir'])
            if not shelve_db_for_extraction_exists: print(f"ERROR: Source DB for hidden state extraction {db_output_file} not found. Cannot extract hidden states."); sys.exit(1)
            with shelve.open(hidden_states_db_path, 'c') as out_db, torch.no_grad():
                build_db_batched_local(db_output_file, out_db, bert_teacher_model, tokenizer, batch_size=8, debug_mode_local=debug_mode_extraction, max_samples_local=max_samples_extraction)
            print(f"Hidden states extraction completed. DB at {hidden_states_db_path}")
        bert_teacher_model.cpu(); del bert_teacher_model;
        if device.type == 'cuda': torch.cuda.empty_cache()
        skip_topk = False
        if not cli_args.force_rerun_stage2 and any(os.path.exists(f"{topk_db_path}{ext}") for ext in [".db", ".dat", ".dir", ".bak", ""]):
            try:
                with shelve.open(topk_db_path, 'r') as temp_check:
                    if list(temp_check.keys()): print(f"Top-K DB found at ~{topk_db_path} with content, and --force_rerun_stage2 not set. Skipping top-k computation."); skip_topk = True
                    else: print(f"Top-K DB found at ~{topk_db_path} but is EMPTY. Will re-run top-k computation.")
            except Exception: print(f"Top-K DB found at ~{topk_db_path} but couldn't be opened/verified. Will re-run top-k computation.")
        if not skip_topk:
            print("\n--- Computing top-k logits ---")
            if not os.path.exists(linear_projection_layer_path): print(f"Error: Linear projection layer {linear_projection_layer_path} not found. Cannot compute top-k."); sys.exit(1)
            source_hidden_db_exists = any(os.path.exists(f"{hidden_states_db_path}{ext}") for ext in [".db", ".dat", ".dir", ".bak", ""])
            if not source_hidden_db_exists: print(f"Error: Hidden states DB ~{hidden_states_db_path} not found. Cannot compute top-k."); sys.exit(1)
            try:
                with shelve.open(hidden_states_db_path, 'r') as temp_hs_db:
                    if not list(temp_hs_db.keys()): print(f"Error: Hidden states DB ~{hidden_states_db_path} is EMPTY. Cannot compute top-k."); sys.exit(1)
            except Exception as e: print(f"Error opening hidden states DB ~{hidden_states_db_path}: {e}. Cannot compute top-k."); sys.exit(1)
            linear_for_topk = torch.load(linear_projection_layer_path, map_location=device).half().to(device); k_topk = 8
            with shelve.open(hidden_states_db_path, 'r') as db_shelf, shelve.open(topk_db_path, 'c') as topk_db_shelf:
                db_keys = list(db_shelf.keys())
                if not db_keys: print(f"Warning: Input hidden states DB ({hidden_states_db_path}) for top-k computation is empty. No top-k logits will be computed.")
                if debug_mode_extraction and max_samples_extraction < len(db_keys): db_keys = db_keys[:max_samples_extraction]; print(f"DEBUG MODE: Computing top-k for {len(db_keys)} items.")
                for key in tqdm(db_keys, total=len(db_keys), desc='Computing topk...'):
                    value = db_shelf[key]; bert_hidden = torch.tensor(tensor_loads(value)).to(device).half()
                    topk_results = linear_for_topk(bert_hidden).topk(dim=-1, k=k_topk); topk_db_shelf[key] = dump_topk(topk_results)
                    del bert_hidden; 
                    if device.type == 'cuda': torch.cuda.empty_cache()
            linear_for_topk.cpu(); del linear_for_topk;
            if device.type == 'cuda': torch.cuda.empty_cache()
            print(f"Top-k logits computed and saved to {topk_db_path}")
    else:
        print("Skipping Stage 2: Teacher Hidden States & Top-K.")
        if not (os.path.exists(linear_projection_layer_path) and any(os.path.exists(f"{hidden_states_db_path}{ext}") for ext in [".db", ".dat", ".dir", ".bak", ""]) and any(os.path.exists(f"{topk_db_path}{ext}") for ext in [".db", ".dat", ".dir", ".bak", ""])):
            print(f"Warning: Stage 2 skipped, but one or more required files for Stage 3 not found. Subsequent stages might fail.")

    if cli_args.run_stage3:
        print("\n--- Stage 3: Knowledge Distillation Training ---")
        required_stage2_files_ok = True # Assume true initially
        # ... (checks for required_stage2_files_ok as before) ...
        if not os.path.exists(linear_projection_layer_path): print(f"Error: Linear projection layer {linear_projection_layer_path} from Stage 2 not found."); required_stage2_files_ok = False
        hidden_db_found_nonempty = False
        if any(os.path.exists(f"{hidden_states_db_path}{ext}") for ext in [".db", ".dat", ".dir", ".bak", ""]):
            try:
                with shelve.open(hidden_states_db_path, 'r') as temp_db:
                    if list(temp_db.keys()): hidden_db_found_nonempty = True
            except Exception: pass
        if not hidden_db_found_nonempty: print(f"Error: Hidden states DB ~{hidden_states_db_path} from Stage 2 not found or empty."); required_stage2_files_ok = False
        topk_db_found_nonempty = False
        if any(os.path.exists(f"{topk_db_path}{ext}") for ext in [".db", ".dat", ".dir", ".bak", ""]):
            try:
                with shelve.open(topk_db_path, 'r') as temp_db:
                    if list(temp_db.keys()): topk_db_found_nonempty = True
            except Exception: pass
        if not topk_db_found_nonempty: print(f"Error: Top-K DB ~{topk_db_path} from Stage 2 not found or empty."); required_stage2_files_ok = False
        if not required_stage2_files_ok: print(f"Cannot proceed with Stage 3 due to missing/empty files from Stage 2."); sys.exit(1)

        config_path_kd = os.path.join(opennmt_dir_abs, "config", "config-transformer-base-mt-deen.yml")
        with open(config_path_kd, 'r') as stream: config_kd = yaml.safe_load(stream)
        args_kd = argparse.Namespace(**config_kd)
        default_num_layers = 6
        args_kd.enc_layers = config_kd.get('enc_layers', config_kd.get('layers', default_num_layers))
        args_kd.dec_layers = config_kd.get('dec_layers', config_kd.get('layers', default_num_layers))
        args_kd.train_from = None; args_kd.max_grad_norm = 0.0
        args_kd.kd_topk = 8; args_kd.train_steps = cli_args.num_steps_kd
        args_kd.kd_temperature = 10.0; args_kd.kd_alpha = 0.5
        args_kd.warmup_steps = cli_args.kd_warmup_steps; args_kd.learning_rate = 2.0
        args_kd.bert_dump = bert_dump_path; args_kd.data_db = db_output_file
        args_kd.bert_kd = True; args_kd.data = 'data/DEEN'
        args_kd.model_type = "text"; args_kd.copy_attn = False; args_kd.global_attention = "general"
        args_kd.src_word_vec_size = args_kd.word_vec_size; args_kd.tgt_word_vec_size = args_kd.word_vec_size
        args_kd.feat_merge = "concat"; args_kd.feat_vec_size = -1; args_kd.feat_vec_exponent = 0.7
        args_kd.pre_word_vecs_enc = None; args_kd.pre_word_vecs_dec = None
        args_kd.fix_word_vecs_enc = False; args_kd.fix_word_vecs_dec = False
        args_kd.enc_rnn_size = getattr(args_kd, 'rnn_size', 512); args_kd.dec_rnn_size = getattr(args_kd, 'rnn_size', 512) 
        args_kd.transformer_ff = getattr(args_kd, 'transformer_ff', 2048)
        args_kd.heads = getattr(args_kd, 'heads', 8)
        args_kd.max_relative_positions = getattr(args_kd, 'max_relative_positions', 0)
        args_kd.position_encoding = getattr(args_kd, 'position_encoding', True)
        args_kd.param_init = getattr(args_kd, 'param_init', 0.0)
        args_kd.param_init_glorot = getattr(args_kd, 'param_init_glorot', True)
        args_kd.share_embeddings = False; args_kd.share_decoder_embeddings = False
        args_kd.truncated_decoder = getattr(args_kd, 'truncated_decoder', 0)
        args_kd.max_generator_batches = getattr(args_kd, 'max_generator_batches', 32)
        args_kd.normalization = getattr(args_kd, 'normalization', 'sents')
        args_kd.accum_count = getattr(args_kd, 'accum_count', [1])
        if not isinstance(args_kd.accum_count, list): args_kd.accum_count = [args_kd.accum_count]
        args_kd.accum_steps = getattr(args_kd, 'accum_steps', [0]) # from notebook; OpenNMT might expect alignment with accum_count
        args_kd.average_decay = getattr(args_kd, 'average_decay', 0.0)
        args_kd.average_every = getattr(args_kd, 'average_every', 1)
        args_kd.valid_steps = cli_args.kd_valid_steps
        args_kd.early_stopping = getattr(args_kd, 'early_stopping', 0)
        args_kd.early_stopping_criteria = getattr(args_kd, 'early_stopping_criteria', None)
        args_kd.valid_batch_size = getattr(args_kd, 'valid_batch_size', 8) # from notebook
        args_kd.self_attn_type = getattr(args_kd, 'self_attn_type', "scaled-dot")
        args_kd.input_feed = getattr(args_kd, 'input_feed', 1) if getattr(args_kd, 'decoder_type', "transformer") == "rnn" else 0
        args_kd.copy_attn_type = getattr(args_kd, 'copy_attn_type', None)
        args_kd.generator_function = getattr(args_kd, 'generator_function', "softmax")
        args_kd.local_rank = -1; args_kd.gpu_ranks = [0] if torch.cuda.is_available() else []
        args_kd.gpu_verbose_level = 0; args_kd.world_size = 1
        args_kd.encoder_type = getattr(args_kd, 'encoder_type', "transformer")
        args_kd.decoder_type = getattr(args_kd, 'decoder_type', "transformer")
        args_kd.dropout = float(getattr(args_kd, 'dropout', 0.1)) 
        args_kd.attention_dropout = float(getattr(args_kd, 'attention_dropout', args_kd.dropout))
        args_kd.bridge = getattr(args_kd, 'bridge', False) 
        args_kd.aux_tune = False
        args_kd.subword_prefix = " "; args_kd.subword_prefix_is_joiner = False
        args_kd.save_model = os.path.join(output_path_kd, 'ckpt', 'model')
        args_kd.log_file = os.path.join(output_path_kd, 'log', 'log.txt')
        args_kd.tensorboard = True; args_kd.tensorboard_log_dir = os.path.join(output_path_kd, 'log')

        print("\n--- Loading vocabulary and dataset for KD ---")
        vocab_onmt_path = args_kd.data + '.vocab.pt'
        if not os.path.exists(vocab_onmt_path): print(f"ERROR: Vocab file {vocab_onmt_path} not found for KD stage."); sys.exit(1)
        vocab_onmt = torch.load(vocab_onmt_path)
        src_vocab_kd = vocab_onmt['src'].fields[0][1].vocab; tgt_vocab_kd = vocab_onmt['tgt'].fields[0][1].vocab
        print(f"Initializing BertKdDataset with data_db: {args_kd.data_db}, bert_dump: {args_kd.bert_dump}")
        train_dataset_kd = BertKdDataset(args_kd.data_db, args_kd.bert_dump, src_vocab_kd.stoi, tgt_vocab_kd.stoi, max_len=150, k=args_kd.kd_topk)
        
        num_kd_samples = len(train_dataset_kd.id_lens if hasattr(train_dataset_kd, 'id_lens') else []) # Use id_lens length
        print(f"Length of KD train dataset (train_dataset_kd.id_lens): {num_kd_samples}") # Adjusted to print from id_lens
        if not num_kd_samples > 0 : print("CRITICAL ERROR: KD train dataset (train_dataset_kd.id_lens) is effectively empty!"); sys.exit(1)

        BUCKET_SIZE_KD_default = 8192; batch_size_tokens_kd_default = 6144
        if num_kd_samples < 50: 
            print(f"KD dataset is very small ({num_kd_samples} samples). Adjusting sampler parameters.")
            BUCKET_SIZE_KD = min(BUCKET_SIZE_KD_default, num_kd_samples * 256) 
            BUCKET_SIZE_KD = max(BUCKET_SIZE_KD, num_kd_samples) # Ensure bucket can at least hold all samples if extremely few
            batch_size_tokens_kd = min(batch_size_tokens_kd_default, num_kd_samples * 150) 
            batch_size_tokens_kd = max(batch_size_tokens_kd, 150 * 1) 
            print(f"Adjusted BUCKET_SIZE_KD: {BUCKET_SIZE_KD}"); print(f"Adjusted batch_size_tokens_kd (target tokens per batch): {batch_size_tokens_kd}")
        else: BUCKET_SIZE_KD = BUCKET_SIZE_KD_default; batch_size_tokens_kd = batch_size_tokens_kd_default
        
        if not hasattr(train_dataset_kd, 'id_lens') or not train_dataset_kd.id_lens:
            print("CRITICAL ERROR: train_dataset_kd.id_lens is missing or empty. Sampler cannot be initialized."); sys.exit(1)
        print(f"Initializing BertKdTokenBucketSampler with id_lens containing {len(train_dataset_kd.id_lens)} entries.")
        train_sampler_kd = BertKdTokenBucketSampler(train_dataset_kd.id_lens, bucket_size=BUCKET_SIZE_KD, batch_size=batch_size_tokens_kd, batch_multiple=1)
        
        train_loader_kd = DataLoader(train_dataset_kd, batch_sampler=train_sampler_kd, num_workers=min(4, os.cpu_count() or 1), collate_fn=BertKdDataset.pad_collate)
        try:
            _first_batch_kd_check = next(iter(train_loader_kd)); print(f"KD DataLoader can produce at least one batch. Batch check successful."); del _first_batch_kd_check
        except StopIteration:
            print("CRITICAL ERROR: KD DataLoader is empty. Cannot proceed with KD training.")
            print(f"  Dataset samples: {num_kd_samples}"); print(f"  Sampler bucket_size: {BUCKET_SIZE_KD}"); print(f"  Sampler batch_size (tokens): {batch_size_tokens_kd}")
            print(f"  Sampler id_lens items: {len(train_dataset_kd.id_lens) if hasattr(train_dataset_kd, 'id_lens') else 'N/A'}")
            if hasattr(train_dataset_kd, 'id_lens') and train_dataset_kd.id_lens:
                print("  Sample (id, (src_len, tgt_len)) from id_lens (first 5):")
                for i, (k, v) in enumerate(train_dataset_kd.id_lens.items()):
                    if i >= 5: break
                    print(f"    {k}: {v}")
            sys.exit(1)
        iter_state = {'train_iter_kd': cycle_loader(train_loader_kd, device)}

        print("\n--- Building OpenNMT model, optimizer, and trainer for KD ---")
        onmt_fields = {'src': vocab_onmt['src'], 'tgt': vocab_onmt['tgt']}
        model_kd = build_model(args_kd, args_kd, fields=onmt_fields, checkpoint=None).to(device) 
        optim_kd = Optimizer.from_opt(model_kd, args_kd, checkpoint=None)
        args_kd.report_every = int(getattr(args_kd, 'report_every', 50))
        if args_kd.tensorboard:
            from tensorboardX import SummaryWriter
            writer = SummaryWriter(args_kd.tensorboard_log_dir, comment="unmt"); args_kd.report_manager = onmt.utils.ReportMgr(report_every=args_kd.report_every, start_time=None, tensorboard_writer=writer)
        else: args_kd.report_manager = None
        model_saver_kd = build_model_saver(args_kd, args_kd, model_kd, onmt_fields, optim_kd)
        trainer_kd = build_trainer(args_kd, device_id=0 if device.type == 'cuda' else -1, model=model_kd, fields=onmt_fields, optim=optim_kd, model_saver=model_saver_kd, report_manager=args_kd.report_manager)

        print("\n--- Knowledge Distillation Training Loop ---")
        if not hasattr(optim_kd, '_step'): optim_kd._step = 0
        def manual_train_iter_local():
            nonlocal iter_state
            while True:
                try: batch = next(iter_state['train_iter_kd'])
                except StopIteration: 
                    print("KD DataLoader exhausted. Resetting iterator."); iter_state['train_iter_kd'] = cycle_loader(train_loader_kd, device)
                    try: batch = next(iter_state['train_iter_kd'])
                    except StopIteration: print("CRITICAL ERROR: KD DataLoader is still empty after reset during training."); sys.exit(1)
                yield batch
        print(f"Starting KD training for {args_kd.train_steps} steps...")
        trainer_kd.train(manual_train_iter_local(), train_steps=args_kd.train_steps, save_checkpoint_steps=cli_args.kd_save_checkpoint_steps, valid_iter=None, valid_steps=args_kd.valid_steps)
        print(f"KD Model trained and saved to {output_path_kd}/ckpt")
    else:
        print("Skipping Stage 3: Knowledge Distillation Training.")
        if not os.path.exists(kd_model_checkpoint_path): print(f"Warning: Stage 3 skipped, but KD model at {kd_model_checkpoint_path} not found. Subsequent stages might fail.")

    if cli_args.run_stage4:
        print("\n--- Stage 4: Translation and Evaluation ---")
        # ... (Stage 4 code as in previous correct version) ...
        if not os.path.exists(kd_model_checkpoint_path): print(f"Error: KD model {kd_model_checkpoint_path} not found. Cannot proceed with Stage 4."); sys.exit(1)
        out_dir_translate = "output/translation"; os.makedirs(out_dir_translate, exist_ok=True)
        print(f"Model found at {kd_model_checkpoint_path}. Running translation...")
        try:
            translate_script_path = os.path.join(opennmt_dir_abs, "translate.py")
            translate_cmd = [sys.executable, translate_script_path, "-model", kd_model_checkpoint_path, "-src", f"{data_dir}/test.de.bert", "-output", f"{out_dir_translate}/result.en", "-beam_size", "5", "-alpha", "0.6", "-length_penalty", "wu"]
            if torch.cuda.is_available(): translate_cmd.extend(["-gpu", "0"])
            run_shell_command(translate_cmd); result_en_file = f"{out_dir_translate}/result.en"
            if os.path.exists(result_en_file):
                print("Translation completed. Detokenizing output...")
                run_shell_command([sys.executable, "scripts/bert_detokenize.py", "--file", result_en_file, "--output_dir", out_dir_translate])
                result_en_detok_file = f"{out_dir_translate}/result.en.detok"
                if os.path.exists(result_en_detok_file):
                    print("Evaluating with BLEU score...")
                    bleu_script_path = os.path.join(opennmt_dir_abs, "tools", "multi-bleu.perl")
                    bleu_output_file = f"{out_dir_translate}/result.bleu"; ref_file_translate = f"{data_dir}/test.en"
                    with open(result_en_detok_file, 'r', encoding='utf-8') as infile, open(bleu_output_file, 'w', encoding='utf-8') as outfile:
                        run_shell_command(["perl", bleu_script_path, ref_file_translate], stdin=infile, stdout=outfile) 
                    if os.path.exists(bleu_output_file):
                        with open(bleu_output_file, "r", encoding='utf-8') as f: print(f"BLEU Score: {f.read().strip()}")
                    else: print("Warning: BLEU score file was not generated.")
                else: print("Warning: Detokenized output file was not generated.")
            else: print("Warning: Translation output file was not generated.")
        except Exception as e: print(f"Error during translation process: {str(e)}\n{traceback.format_exc()}")
    else: print("Skipping Stage 4: Translation and Evaluation.")

    if cli_args.run_stage5:
        print("\n--- Stage 5: Displaying Figures (if available) ---")
        # ... (Stage 5 code as in previous correct version) ...
        figures_to_display = {'CMLM Finetuning': 'figures/cmlm-finetuning.png', 'Translation Losses': 'figures/translation-losses.png', 'Translation Accuracy': 'figures/translation-accuracy.png'}
        existing_figures = {title: path for title, path in figures_to_display.items() if os.path.exists(path)}
        if existing_figures:
            num_figs = len(existing_figures); fig, axes = plt.subplots(1, num_figs, figsize=(6 * num_figs, 5))
            if num_figs == 1: axes = [axes] # Ensure axes is iterable
            for i, (title, path) in enumerate(existing_figures.items()):
                axes[i].set_title(title)
                try: axes[i].imshow(plt.imread(path)); axes[i].axis('off')
                except Exception as e: print(f"Could not load/display {path}: {e}")
            plt.tight_layout()
            try: plt.show(); print("Displayed figures. Close plot window to continue.")
            except Exception as e: print(f"Could not show plots (e.g., no GUI): {e}")
        else: print("No figures found in 'figures/' directory. Skipping display.")

    else: print("Skipping Stage 5: Display Figures.")

    print("\n--- Script execution finished ---")

if __name__ == "__main__":
    main()
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Python
IGNORE_WHEN_COPYING_END