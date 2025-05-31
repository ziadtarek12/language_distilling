import os
import sys
import torch
import numpy as np
import random
import shelve
import io
import argparse
import yaml
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import tensorboardX
import subprocess
import torch.nn as nn
import traceback
import matplotlib.pyplot as plt

# Imports from the repository's scripts/modules
# These will be resolved after cloning the repo and changing directory
# sys.path needs to be set up correctly before these are imported if they are at top level.
# For now, keeping specific imports closer to their usage or within functions if possible,
# or ensuring sys.path is set before they are truly needed.


def run_shell_command(command, **kwargs):
    """Helper function to run shell commands."""
    print(f"Executing: {' '.join(command)}")
    try:
        process = subprocess.run(command, check=True, text=True, capture_output=True, **kwargs)
        if process.stdout:
            print(process.stdout)
        if process.stderr:
            print(process.stderr) # print stderr for non-erroring commands too
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(command)}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"Stdout: {e.stdout}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        raise


def main():
    # --- Initial Setup and Downloads ---
    print("--- Stage 0: Initial Setup and Downloads ---")
    if not os.path.exists("language_distilling"):
        run_shell_command(["git", "clone", "https://github.com/ziadtarek12/language_distilling"])
    else:
        print("language_distilling repository already cloned.")
        
    os.chdir("language_distilling")
    run_shell_command(["git", "checkout", "eval"])

    # Pip installs
    # It's generally better to manage dependencies via requirements.txt or environment setup,
    # but for direct notebook conversion, we include these.
    print("\n--- Installing Python packages ---")
    # run_shell_command(["pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"]) # Be cautious with uninstall
    packages_to_install = [
        "transformers==4.26.0",
        "pytorch-pretrained-bert",
        "cytoolz",
        "tqdm",
        "torchtext==0.16.0",
        "torchvision==0.16.0",
        "torch==2.1.0",
        "torchaudio==2.1.0",
        "configargparse",
        "tensorboardX",
        "ipdb" # ipdb might not be needed for a script
    ]
    for package in packages_to_install:
        if package == "ipdb" and os.environ.get("KAGGLE_KERNEL_RUN_TYPE"): # Skip ipdb in Kaggle
            print(f"Skipping installation of {package} in Kaggle environment.")
            continue
        run_shell_command([sys.executable, "-m", "pip", "install", package])


    # Add local paths for imports from the cloned repository
    sys.path.append('.')
    sys.path.append('./opennmt')

    # Now that sys.path is set, we can import modules from the repo
    from scripts.bert_tokenize import tokenize, process as bert_tokenize_process
    from scripts.bert_prepro import main as bert_prepro_main
    from cmlm.data import BertDataset, TokenBucketSampler as CMLMTokenBucketSampler
    from cmlm.model import convert_embedding, BertForSeq2seq
    from cmlm.util import Logger, RunningMeter # Logger not explicitly used
    # from run_cmlm_finetuning import noam_schedule # noam_schedule not explicitly used
    from vocab_loader import safe_load_vocab
    from dump_teacher_hiddens import tensor_dumps, gather_hiddens, BertSampleDataset, batch_features, process_batch as dump_process_batch
    from dump_teacher_topk import tensor_loads, dump_topk
    from onmt.inputters.bert_kd_dataset import BertKdDataset, TokenBucketSampler as BertKdTokenBucketSampler
    from onmt.utils.optimizers import Optimizer
    from onmt.train_single import build_model_saver, build_trainer, cycle_loader
    from onmt.model_builder import build_model


    # Set seed for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Create directories for data and outputs
    print("\n--- Creating directories ---")
    dirs_to_create = [
        "data/",
        "output/cmlm_model",
        "output/bert_dump",
        "output/kd-model/ckpt",
        "output/kd-model/log",
        "output/translation"
    ]
    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)
        print(f"Created/Ensured directory: {d}")

    # Download IWSLT German-English dataset
    print("\n--- Downloading IWSLT German-English dataset ---")
    if not os.path.exists("data/de-en/train.de"): # Basic check if data might exist
        run_shell_command(["bash", "scripts/download-iwslt_deen.sh"])
    else:
        print("Dataset files seem to exist, skipping download.")


    # --- Stage 1: CMLM Fine-tuning ---
    print("\n--- Stage 1: CMLM Fine-tuning ---")

    # BERT Tokenization
    print("\n--- BERT Tokenization ---")
    bert_model_name = "bert-base-multilingual-cased" # Renamed from bert_model to avoid conflict
    tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case='uncased' in bert_model_name)
    data_dir = "data/de-en"

    for language in ['de', 'en']:
        for split in ['train', 'valid', 'test']:
            input_file = f"{data_dir}/{split}.{language}"
            output_file = f"{data_dir}/{split}.{language}.bert"
            if not os.path.exists(output_file): # Avoid re-tokenizing if already done
                print(f"Tokenizing {input_file} to {output_file}...")
                with open(input_file, 'r', encoding='utf-8') as reader, open(output_file, 'w', encoding='utf-8') as writer:
                    bert_tokenize_process(reader, writer, tokenizer)
            else:
                print(f"Skipping tokenization for {input_file}, output exists.")
    
    # Create dataset DB for BERT training
    print("\n--- Creating dataset DB for BERT training ---")
    db_output_file = 'data/DEEN.db'
    if not os.path.exists(db_output_file):
        prepro_args = argparse.Namespace(
            src=f"{data_dir}/train.de.bert",
            tgt=f"{data_dir}/train.en.bert",
            output=db_output_file
        )
        bert_prepro_main(prepro_args)
    else:
        print(f"Skipping BERT prepro, {db_output_file} exists.")

    # Create vocabulary file using OpenNMT's preprocess.py
    vocab_file_onmt = "data/DEEN.vocab.pt"
    if not os.path.exists(vocab_file_onmt):
        print("Creating vocabulary files with OpenNMT preprocess.py...")
        opennmt_preprocess_cmd = [
            sys.executable, "opennmt/preprocess.py",
            "-train_src", f"{data_dir}/train.de.bert",
            "-train_tgt", f"{data_dir}/train.en.bert",
            "-valid_src", f"{data_dir}/valid.de.bert",
            "-valid_tgt", f"{data_dir}/valid.en.bert",
            "-save_data", "data/DEEN",
            "-src_seq_length", "150", "-tgt_seq_length", "150"
        ]
        run_shell_command(opennmt_preprocess_cmd)
    else:
        print(f"Skipping OpenNMT vocab creation, {vocab_file_onmt} exists.")


    # CMLM Model Setup
    print("\n--- CMLM Model Setup ---")
    train_file_db = "data/DEEN.db" # Renamed from train_file to avoid conflict
    valid_src = f"{data_dir}/valid.de.bert"
    valid_tgt = f"{data_dir}/valid.en.bert"
    cmlm_output_dir = "output/cmlm_model" # Renamed from output_dir

    vocab_dump = safe_load_vocab(vocab_file_onmt)
    vocab_stoi = vocab_dump['tgt'].fields[0][1].vocab.stoi # Renamed from vocab

    train_dataset_cmlm = BertDataset(train_file_db, tokenizer, vocab_stoi, seq_len=512, max_len=150) # Renamed

    BUCKET_SIZE_CMLM = 8192 # Renamed
    train_sampler_cmlm = CMLMTokenBucketSampler( # Use aliased/specific CMLMTokenBucketSampler
        train_dataset_cmlm.lens, BUCKET_SIZE_CMLM, 6144, batch_multiple=1)

    train_loader_cmlm = DataLoader(train_dataset_cmlm, batch_sampler=train_sampler_cmlm,
                                num_workers=min(4, os.cpu_count() or 1), # Adjusted num_workers
                                collate_fn=BertDataset.pad_collate)

    cmlm_model = BertForSeq2seq.from_pretrained(bert_model_name) # Renamed from model
    bert_embedding = cmlm_model.bert.embeddings.word_embeddings.weight
    
    hidden_size = cmlm_model.config.hidden_size
    print(f"Original model: BERT hidden size = {hidden_size}")
    print(f"Original model: BERT vocab size = {bert_embedding.size(0)}")
    print(f"Target vocabulary size = {len(vocab_stoi)}")

    embedding = convert_embedding(tokenizer, vocab_stoi, bert_embedding)
    
    print(f"Updating model architecture for vocabulary size: {embedding.size(0)}")
    cmlm_model.cls.predictions.decoder = torch.nn.Linear(hidden_size, embedding.size(0), bias=True)
    cmlm_model.cls.predictions.bias = torch.nn.Parameter(torch.zeros(embedding.size(0)))
    cmlm_model.config.vocab_size = embedding.size(0)
    cmlm_model.cls.predictions.decoder.weight.data.copy_(embedding.data)
    cmlm_model.to(device)
    print(f"Model adapted with vocabulary size: {cmlm_model.config.vocab_size}")

    # CMLM Training
    print("\n--- CMLM Training Loop ---")
    learning_rate_cmlm = 5e-5 # Renamed
    warmup_proportion_cmlm = 0.1 # Renamed
    max_steps_cmlm = 100000 # Renamed
    num_steps_to_run_cmlm = 100 # Reduced for quick test, original: 100000

    param_optimizer_cmlm = list(cmlm_model.named_parameters()) # Renamed
    no_decay_cmlm = ['bias', 'LayerNorm.bias', 'LayerNorm.weight'] # Renamed
    optimizer_grouped_parameters_cmlm = [ # Renamed
        {'params': [p for n, p in param_optimizer_cmlm
                    if not any(nd in n for nd in no_decay_cmlm)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer_cmlm
                    if any(nd in n for nd in no_decay_cmlm)],
         'weight_decay': 0.0}
    ]
    optimizer_cmlm = AdamW(optimizer_grouped_parameters_cmlm, lr=learning_rate_cmlm) # Renamed
    scheduler_cmlm = get_linear_schedule_with_warmup( # Renamed
        optimizer_cmlm,
        num_warmup_steps=int(max_steps_cmlm * warmup_proportion_cmlm),
        num_training_steps=max_steps_cmlm
    )

    running_loss_cmlm = RunningMeter('loss') # Renamed
    cmlm_model.train()

    print(f"Starting CMLM fine-tuning for {num_steps_to_run_cmlm} steps...")
    cmlm_train_iter = iter(train_loader_cmlm) # Renamed
    for step in range(num_steps_to_run_cmlm):
        try:
            batch = next(cmlm_train_iter)
        except StopIteration:
            cmlm_train_iter = iter(train_loader_cmlm)
            batch = next(cmlm_train_iter)
            
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, lm_label_ids = batch
        
        optimizer_cmlm.zero_grad()
        output_mask = lm_label_ids != -1
        loss = cmlm_model(input_ids, segment_ids, input_mask, lm_label_ids, output_mask)
        loss.backward()
        optimizer_cmlm.step()
        scheduler_cmlm.step()
        
        running_loss_cmlm(loss.item())
        if step % 10 == 0 or step == num_steps_to_run_cmlm -1 : # Print less frequently
            print(f"CMLM Step {step}, Loss: {running_loss_cmlm.val:.4f}")
        if step % 100 == 0:
            torch.cuda.empty_cache()

    cmlm_model_save_path = f"{cmlm_output_dir}/model_step_{num_steps_to_run_cmlm}.pt"
    torch.save(cmlm_model.state_dict(), cmlm_model_save_path)
    print(f"CMLM Model saved to {cmlm_model_save_path}")

    if torch.cuda.is_available():
        print(torch.cuda.memory_summary(device=None, abbreviated=False))
        torch.cuda.empty_cache()


    # --- Stage 2: Teacher Hidden States and Top-K Logits ---
    print("\n--- Stage 2: Teacher Hidden States and Top-K Logits ---")
    
    # Load fine-tuned model for hidden state extraction
    print("\n--- Loading fine-tuned CMLM model ---")
    ckpt_path = cmlm_model_save_path # Use the saved model path
    bert_dump_path = "output/bert_dump"

    bert_teacher_model = BertForSeq2seq.from_pretrained(bert_model_name).eval() # Renamed
    bert_teacher_model.to(device)
    
    state_dict = torch.load(ckpt_path, map_location=device)
    vsize = state_dict['cls.predictions.decoder.weight'].size(0)
    
    print(f"Resizing teacher model to exact vocabulary size: {vsize}")
    teacher_hidden_size = bert_teacher_model.config.hidden_size # Renamed
    bert_teacher_model.cls.predictions.decoder = torch.nn.Linear(teacher_hidden_size, vsize, bias=True)
    bert_teacher_model.cls.predictions.bias = bert_teacher_model.cls.predictions.decoder.bias # This should be okay, but usually it's a separate Parameter
    # Ensure bias parameter is correctly linked or copied if it was a separate param in the state_dict
    if 'cls.predictions.bias' in state_dict:
         bert_teacher_model.cls.predictions.bias = torch.nn.Parameter(torch.zeros(vsize)) # Recreate if necessary
    bert_teacher_model.config.vocab_size = vsize
    bert_teacher_model.load_state_dict(state_dict)

    linear_projection_layer = torch.nn.Linear(bert_teacher_model.config.hidden_size, bert_teacher_model.config.vocab_size) # Renamed
    linear_projection_layer.weight.data = state_dict['cls.predictions.decoder.weight']
    linear_projection_layer.bias.data = state_dict['cls.predictions.bias']
    linear_projection_layer_path = f'{bert_dump_path}/linear.pt'
    torch.save(linear_projection_layer, linear_projection_layer_path)
    print(f"Linear projection layer saved to {linear_projection_layer_path}")

    # Function to extract hidden states (modified from notebook cell)
    def build_db_batched_local(corpus_path, out_db_shelf, bert_model_param, toker_param, batch_size=8, debug_mode_local=False, max_samples_local=100): # Renamed params
        dataset = BertSampleDataset(corpus_path, toker_param)
        
        if debug_mode_local:
            all_ids = list(dataset.ids) # Convert shelve keys to list for slicing
            subset_ids = all_ids[:max_samples_local] if len(all_ids) > max_samples_local else all_ids
            
            # Create a temporary dataset with subset_ids if BertSampleDataset internals require it
            # For simplicity, let's assume we can filter loader or ids.
            # The original code modifies dataset.ids, which might be tricky if it's a ShelfKeyView
            # A robust way is to filter IDs before passing to DataLoader or filter batches.
            # Here, we'll make a new dataset object for debug, assuming it's lightweight to create.
            if len(all_ids) > max_samples_local:
                 print(f"DEBUG MODE: Creating a subset dataset with {len(subset_ids)} samples.")
                 dataset_debug = BertSampleDataset(corpus_path, toker_param) # Re-init
                 dataset_debug.ids = subset_ids # Hope this works as intended
                 dataset = dataset_debug

            print(f"DEBUG MODE: Processing {len(dataset.ids)} samples instead of {len(all_ids) if not isinstance(all_ids, range) else 'all (unknown count without iteration)'}")
        
        loader = DataLoader(dataset, batch_size=batch_size,
                           num_workers=min(4, os.cpu_count() or 1), collate_fn=batch_features)
        
        with tqdm(desc='Computing BERT features', total=len(dataset.ids)) as pbar: # Use dataset.ids
            for ids, *batch_data in loader: # Renamed batch
                outputs = dump_process_batch(batch_data, bert_model_param, toker_param)
                for id_, output in zip(ids, outputs):
                    if output is not None:
                        out_db_shelf[id_] = tensor_dumps(output) # Use out_db_shelf
                pbar.update(len(ids))
                
                if debug_mode_local and pbar.n >= max_samples_local : # Check processed count
                    print("First batch processed or max_samples reached, breaking early due to debug mode")
                    break
    
    # Extract hidden states
    print("\n--- Extracting hidden states ---")
    db_path_teacher = "data/DEEN.db" # Renamed
    debug_mode_extraction = True 
    max_samples_extraction = 10 # Reduced for quick test, original: 100

    hidden_states_db_path = f'{bert_dump_path}/db'
    # Check if extraction should be skipped
    skip_extraction = False
    if os.path.exists(f"{hidden_states_db_path}.db") or os.path.exists(f"{hidden_states_db_path}.dat"): # Shelve creates multiple files
        if debug_mode_extraction:
             print(f"DEBUG MODE: Hidden states DB seems to exist at {hidden_states_db_path}. Skipping extraction for debug run.")
             skip_extraction = True
        else:
             # For a full run, you might still want to re-extract or make it conditional
             print(f"Hidden states DB seems to exist at {hidden_states_db_path}. Set debug_mode_extraction=False and ensure clean state if re-extraction is needed.")
             # skip_extraction = True # Or False if you want to overwrite

    if not skip_extraction:
        with shelve.open(hidden_states_db_path, 'c') as out_db, torch.no_grad():
            build_db_batched_local(db_path_teacher, out_db, bert_teacher_model, tokenizer, batch_size=8, 
                            debug_mode_local=debug_mode_extraction, max_samples_local=max_samples_extraction)
        print(f"Hidden states extraction completed. DB at {hidden_states_db_path}")
    
    print("Clearing GPU memory after hidden states extraction...")
    bert_teacher_model.cpu()
    del bert_teacher_model
    # linear_projection_layer was already saved, can be moved to CPU if still in memory and needed later, or deleted
    if 'linear_projection_layer' in locals() and hasattr(linear_projection_layer, 'cpu'):
        linear_projection_layer.cpu() # Ensure it's on CPU before potential deletion or if it's reloaded later
    torch.cuda.empty_cache()
    print("GPU memory cleared.")

    if debug_mode_extraction:
        print(f"DEBUG MODE: Hidden states for (up to) {max_samples_extraction} samples processed.")
        print("To run full extraction, set debug_mode_extraction=False and ensure no existing DB file if overwrite is desired.")
    

    # Compute top-k logits
    print("\n--- Computing top-k logits ---")
    k_topk = 8 # Renamed from k
    
    linear_for_topk = torch.load(linear_projection_layer_path, map_location=device) # Renamed and ensure on device
    linear_for_topk = linear_for_topk.half() # FP16/Half precision
    linear_for_topk.to(device)

    topk_db_path = f'{bert_dump_path}/topk'
    # Similar skip logic for topk computation
    skip_topk = False
    if os.path.exists(f"{topk_db_path}.db") or os.path.exists(f"{topk_db_path}.dat"):
        if debug_mode_extraction: # Using same debug flag for consistency in quick runs
             print(f"DEBUG MODE: Top-k DB seems to exist at {topk_db_path}. Skipping top-k computation for debug run.")
             skip_topk = True
        else:
             print(f"Top-k DB seems to exist at {topk_db_path}.")
             # skip_topk = True

    if not skip_topk:
        print("Computing top-k logits...")
        # Ensure the source DB for hidden states exists
        source_db_exists = False
        for ext in [".db", ".dat", ".bak", ".dir", ""]: # Common shelve extensions
            if os.path.exists(f"{hidden_states_db_path}{ext}"):
                source_db_exists = True
                break
        if not source_db_exists:
            raise FileNotFoundError(f"Source hidden states database not found at {hidden_states_db_path}. Run extraction first.")

        with shelve.open(hidden_states_db_path, 'r') as db_shelf, \
             shelve.open(topk_db_path, 'c') as topk_db_shelf: # Renamed shelf objects
            
            db_keys = list(db_shelf.keys()) # Get keys to iterate if db_shelf.items() is slow or problematic
            if debug_mode_extraction and max_samples_extraction < len(db_keys): # Limit for debug
                db_keys = db_keys[:max_samples_extraction]
                print(f"DEBUG MODE: Processing top-k for {len(db_keys)} items from hidden_states_db.")

            for key in tqdm(db_keys, total=len(db_keys), desc='Computing topk...'):
                value = db_shelf[key]
                bert_hidden = torch.tensor(tensor_loads(value)).to(device).half() # Ensure half precision
                topk_results = linear_for_topk(bert_hidden).topk(dim=-1, k=k_topk) # Renamed
                dump = dump_topk(topk_results)
                topk_db_shelf[key] = dump
                del bert_hidden # Manual cleanup
                if device.type == 'cuda': torch.cuda.empty_cache()
        print(f"Top-k logits computed and saved to {topk_db_path}")

    print("Clearing GPU memory after top-k computation...")
    linear_for_topk.cpu()
    del linear_for_topk
    if device.type == 'cuda': torch.cuda.empty_cache()
    print("GPU memory cleared.")


    # --- Stage 3: Knowledge Distillation Training ---
    print("\n--- Stage 3: Knowledge Distillation Training ---")
    
    # Define paths and config for KD
    data_db_kd = "data/DEEN.db" # Renamed
    bert_dump_kd = "output/bert_dump" # Renamed
    data_onmt_kd = "data/DEEN" # Renamed from data
    config_path_kd = "opennmt/config/config-transformer-base-mt-deen.yml" # Renamed
    output_path_kd = "output/kd-model" # Renamed

    # Check for topk database and re-compute if necessary (logic from notebook cell 14)
    print("Checking for required database files for KD...")
    topk_db_file_kd = f"{bert_dump_kd}/topk" # Renamed

    topk_db_exists = any(os.path.exists(f"{topk_db_file_kd}{ext}") for ext in ["", ".db", ".dat", ".bak", ".dir"])
    
    if not topk_db_exists:
        print(f"Warning: Top-k database not found at {topk_db_file_kd}. Attempting to re-run top-k computation...")
        # This re-run logic requires linear layer and hidden states DB.
        # Assuming linear_projection_layer_path and hidden_states_db_path are still valid.
        if not os.path.exists(linear_projection_layer_path):
            raise FileNotFoundError(f"Linear layer {linear_projection_layer_path} not found. Cannot re-run top-k.")
        
        source_db_exists_for_rerun = any(os.path.exists(f"{hidden_states_db_path}{ext}") for ext in ["", ".db", ".dat", ".bak", ".dir"])
        if not source_db_exists_for_rerun:
             raise FileNotFoundError(f"Hidden states DB {hidden_states_db_path} not found. Cannot re-run top-k.")

        print("Reloading linear layer for top-k re-computation...")
        linear_for_topk_rerun = torch.load(linear_projection_layer_path, map_location=device)
        linear_for_topk_rerun = linear_for_topk_rerun.half().to(device)
        
        print("Re-computing top-k logits...")
        with shelve.open(hidden_states_db_path, 'r') as db_shelf_rerun, \
             shelve.open(topk_db_file_kd, 'c') as topk_db_shelf_rerun:
            
            db_keys_rerun = list(db_shelf_rerun.keys())
            # Potentially limit items if in debug mode for the re-run as well
            if debug_mode_extraction and max_samples_extraction < len(db_keys_rerun):
                db_keys_rerun = db_keys_rerun[:max_samples_extraction]

            for key in tqdm(db_keys_rerun, total=len(db_keys_rerun), desc='Re-computing topk...'):
                value = db_shelf_rerun[key]
                bert_hidden = torch.tensor(tensor_loads(value)).to(device).half()
                topk_results = linear_for_topk_rerun(bert_hidden).topk(dim=-1, k=k_topk)
                dump = dump_topk(topk_results)
                topk_db_shelf_rerun[key] = dump
                del bert_hidden
                if device.type == 'cuda': torch.cuda.empty_cache()
        
        linear_for_topk_rerun.cpu()
        del linear_for_topk_rerun
        if device.type == 'cuda': torch.cuda.empty_cache()
        print(f"Top-k logits re-computed and saved to {topk_db_file_kd}")
    else:
        print(f"Top-k database confirmed at {topk_db_file_kd}")

    with open(config_path_kd, 'r') as stream:
        config_kd = yaml.safe_load(stream) # Renamed

    args_kd = argparse.Namespace(**config_kd) # Renamed

    # Setup KD parameters from notebook
    args_kd.train_from = None
    args_kd.max_grad_norm = None # Original notebook had this, but OpenNMT usually uses optim.max_grad_norm
    if hasattr(args_kd, 'optim') and args_kd.optim == 'adam': # A common pattern for OpenNMT Adam
        args_kd.adam_beta1 = getattr(args_kd, 'adam_beta1', 0.9)
        args_kd.adam_beta2 = getattr(args_kd, 'adam_beta2', 0.998) # common for transformer
    args_kd.kd_topk = 8
    args_kd.train_steps = 100 # Reduced for quick test, original: 100000
    args_kd.kd_temperature = 10.0
    args_kd.kd_alpha = 0.5
    args_kd.warmup_steps = 800 # Reduced for quick test, original: 8000
    args_kd.learning_rate = 2.0
    args_kd.bert_dump = bert_dump_kd
    args_kd.data_db = data_db_kd
    args_kd.bert_kd = True
    args_kd.data = data_onmt_kd
    args_kd.model_type = "text"
    args_kd.copy_attn = False
    args_kd.global_attention = "general"
    args_kd.src_word_vec_size = args_kd.word_vec_size
    args_kd.tgt_word_vec_size = args_kd.word_vec_size
    args_kd.feat_merge = "concat"
    args_kd.feat_vec_size = -1
    args_kd.feat_vec_exponent = 0.7
    args_kd.pre_word_vecs_enc = None
    args_kd.pre_word_vecs_dec = None
    # args_kd.pre_word_vecs = None # This might be an older name for pre_word_vecs_enc/dec
    args_kd.fix_word_vecs_enc = False
    args_kd.fix_word_vecs_dec = False
    args_kd.enc_rnn_size = args_kd.rnn_size
    args_kd.dec_rnn_size = args_kd.rnn_size
    args_kd.transformer_ff = getattr(args_kd, 'transformer_ff', 2048)
    args_kd.heads = getattr(args_kd, 'heads', 8)
    args_kd.max_relative_positions = 0
    args_kd.position_encoding = True
    args_kd.param_init = 0.0
    args_kd.param_init_glorot = True
    args_kd.share_embeddings = False # Critical fix from notebook
    args_kd.share_decoder_embeddings = False # Critical fix from notebook
    args_kd.truncated_decoder = 0
    args_kd.max_generator_batches = getattr(args_kd, 'max_generator_batches', 32)
    args_kd.normalization = getattr(args_kd, 'normalization', 'sents')
    args_kd.accum_count = getattr(args_kd, 'accum_count', [1]) # OpenNMT often expects a list here
    if not isinstance(args_kd.accum_count, list): args_kd.accum_count = [args_kd.accum_count]
    args_kd.accum_steps = [0] # As in notebook
    args_kd.average_decay = 0.0
    args_kd.average_every = 1
    args_kd.report_manager = None
    args_kd.valid_steps = getattr(args_kd, 'valid_steps', 1000) # Reduced for test
    args_kd.early_stopping = 0
    args_kd.early_stopping_criteria = None
    args_kd.valid_batch_size = getattr(args_kd, 'valid_batch_size', 8) # Reduced for test
    args_kd.self_attn_type = "scaled-dot"
    args_kd.input_feed = 1 # Usually for RNNs, might not be used by transformer
    args_kd.copy_attn_type = None
    args_kd.generator_function = "softmax"
    args_kd.local_rank = -1
    args_kd.gpu_ranks = [0] if torch.cuda.is_available() else []
    args_kd.gpu_verbose_level = 0
    args_kd.world_size = 1
    args_kd.encoder_type = getattr(args_kd, 'encoder_type', "transformer")
    args_kd.decoder_type = getattr(args_kd, 'decoder_type', "transformer")
    args_kd.enc_layers = getattr(args_kd, 'layers', args_kd.enc_layers) # ensure layers is used if enc_layers not set
    args_kd.dec_layers = getattr(args_kd, 'layers', args_kd.dec_layers) # ensure layers is used if dec_layers not set
    args_kd.dropout = getattr(args_kd, 'dropout', [0.1]) # OpenNMT often expects list for dropout
    if not isinstance(args_kd.dropout, list): args_kd.dropout = [args_kd.dropout] * len(args_kd.accum_count)

    args_kd.attention_dropout = getattr(args_kd, 'attention_dropout', [0.1]) # OpenNMT often expects list for dropout
    if not isinstance(args_kd.attention_dropout, list): args_kd.attention_dropout = [args_kd.attention_dropout] * len(args_kd.accum_count)

    args_kd.bridge = "" # As in notebook, could be True/False or specific type for OpenNMT
    args_kd.aux_tune = False
    args_kd.subword_prefix = " "
    args_kd.subword_prefix_is_joiner = False

    args_kd.save_model = os.path.join(output_path_kd, 'ckpt', 'model')
    args_kd.log_file = os.path.join(output_path_kd, 'log', 'log.txt') # Added .txt
    args_kd.tensorboard = True # Enable tensorboard
    args_kd.tensorboard_log_dir = os.path.join(output_path_kd, 'log')

    # Load vocabulary and dataset for KD
    print("\n--- Loading vocabulary and dataset for KD ---")
    vocab_onmt = torch.load(data_onmt_kd + '.vocab.pt') # Renamed
    src_vocab_kd = vocab_onmt['src'].fields[0][1].vocab # Direct vocab object
    tgt_vocab_kd = vocab_onmt['tgt'].fields[0][1].vocab # Direct vocab object

    train_dataset_kd = BertKdDataset(data_db_kd, bert_dump_kd, 
                                  src_vocab_kd.stoi, tgt_vocab_kd.stoi, # Pass stoi
                                  max_len=150, k=args_kd.kd_topk)

    BUCKET_SIZE_KD = 8192 # Renamed
    train_sampler_kd = BertKdTokenBucketSampler( # Use aliased BertKdTokenBucketSampler
        train_dataset_kd.keys, BUCKET_SIZE_KD, 6144, batch_multiple=1)

    train_loader_kd = DataLoader(train_dataset_kd, batch_sampler=train_sampler_kd,
                              num_workers=min(4, os.cpu_count() or 1),
                              collate_fn=BertKdDataset.pad_collate)
    
    # Need to handle global train_iter for manual_train_iter_local
    # This variable needs to be accessible by the function defined later.
    # We'll make it a dictionary entry or a mutable object to ensure modification sticks.
    iter_state = {'train_iter_kd': cycle_loader(train_loader_kd, device)}


    # Build model, optimizer, saver, trainer for KD
    print("\n--- Building OpenNMT model, optimizer, and trainer for KD ---")
    # Ensure fields are correctly passed to build_model
    # OpenNMT expects 'src': field, 'tgt': field where field is a torchtext.data.Field object
    # vocab_onmt['src'] and vocab_onmt['tgt'] should be these fields.
    onmt_fields = {'src': vocab_onmt['src'], 'tgt': vocab_onmt['tgt']}

    model_kd = build_model(args_kd, args_kd, fields=onmt_fields, checkpoint=None) # Renamed
    model_kd.to(device)

    optim_kd = Optimizer.from_opt(model_kd, args_kd, checkpoint=None) # Renamed

    model_saver_kd = build_model_saver(args_kd, args_kd, model_kd, onmt_fields, optim_kd) # Renamed

    # Before building trainer, ensure report_manager is set up if tensorboard is True
    if args_kd.tensorboard:
        from onmt.utils.logging import init_logger, ErrorHandler
        from tensorboardX import SummaryWriter
        if not os.path.exists(args_kd.log_file): # Create log file dir if not exist
            os.makedirs(os.path.dirname(args_kd.log_file), exist_ok=True)
        logger = init_logger(args_kd.log_file) # Basic logger
        writer = SummaryWriter(args_kd.tensorboard_log_dir, comment="unmt")
        args_kd.report_manager = onmt.utils.ReportMgr(
            args_kd.report_every, start_time=-1, tensorboard_writer=writer
        )
    
    trainer_kd = build_trainer(args_kd, device_id=0 if device.type == 'cuda' else -1, # Handle CPU case
                             model=model_kd, fields=onmt_fields, optim=optim_kd, 
                             model_saver=model_saver_kd,
                             report_manager=args_kd.report_manager) # Renamed

    # KD Training Loop
    print("\n--- Knowledge Distillation Training Loop ---")
    # num_steps_to_run_kd = 100 # Using args_kd.train_steps which was set to 100 for test
    num_steps_to_run_kd = args_kd.train_steps

    if not hasattr(optim_kd, '_step'): # OpenNMT optimizer step tracking
        optim_kd._step = 0 
        
    def manual_train_iter_local(): # Renamed, uses iter_state
        nonlocal iter_state # To modify the 'train_iter_kd' in the outer scope's dictionary
        while True:
            try:
                batch = next(iter_state['train_iter_kd'])
            except StopIteration:
                print("Restarting KD data iterator")
                iter_state['train_iter_kd'] = cycle_loader(train_loader_kd, device)
                batch = next(iter_state['train_iter_kd'])
            yield batch

    print(f"Starting model training with knowledge distillation for {num_steps_to_run_kd} steps...")
    trainer_kd.train(
        manual_train_iter_local(),
        train_steps=num_steps_to_run_kd, # Use train_steps consistently
        save_checkpoint_steps=max(10, num_steps_to_run_kd // 10), # Save e.g. 10 times, or every 100 steps
        valid_iter=None, # No validation in this loop as per notebook
        valid_steps=args_kd.valid_steps # Though no valid_iter, this might be used internally
    )
    print(f"KD Model trained for {num_steps_to_run_kd} steps and saved to {output_path_kd}/ckpt")


    # --- Stage 4: Translation and Evaluation ---
    print("\n--- Stage 4: Translation and Evaluation ---")
    
    model_path_translate = f"{output_path_kd}/ckpt/model_step_{num_steps_to_run_kd}.pt" # Renamed
    src_file_translate = f"{data_dir}/test.de.bert" # Renamed
    tgt_file_translate = f"{data_dir}/test.en.bert" # Renamed (though often not needed for src-only translate)
    out_dir_translate = "output/translation" # Renamed
    ref_file_translate = f"{data_dir}/test.en" # Renamed

    os.makedirs(out_dir_translate, exist_ok=True)

    if os.path.exists(model_path_translate):
        print(f"Model found at {model_path_translate}. Running translation...")
        try:
            translate_cmd = [
                sys.executable, "opennmt/translate.py",
                "-model", model_path_translate,
                "-src", src_file_translate,
                # "-tgt", tgt_file_translate, # -tgt is for gold targets, not strictly needed for generation
                "-output", f"{out_dir_translate}/result.en",
                "-beam_size", "5", "-alpha", "0.6",
                "-length_penalty", "wu"
            ]
            if torch.cuda.is_available():
                translate_cmd.extend(["-gpu", "0"])
            
            run_shell_command(translate_cmd)

            print("Translation completed. Detokenizing output...")
            result_en_file = f"{out_dir_translate}/result.en"
            if os.path.exists(result_en_file):
                detokenize_cmd = [
                    sys.executable, "scripts/bert_detokenize.py",
                    "--file", result_en_file,
                    "--output_dir", out_dir_translate
                ]
                run_shell_command(detokenize_cmd)

                result_en_detok_file = f"{out_dir_translate}/result.en.detok"
                if os.path.exists(result_en_detok_file):
                    print("Evaluating with BLEU score...")
                    bleu_output_file = f"{out_dir_translate}/result.bleu"
                    
                    # Handling redirection for perl script
                    with open(result_en_detok_file, 'r', encoding='utf-8') as infile, \
                         open(bleu_output_file, 'w', encoding='utf-8') as outfile:
                        run_shell_command(
                            ["perl", "opennmt/tools/multi-bleu.perl", ref_file_translate],
                            stdin=infile, stdout=outfile
                        )

                    if os.path.exists(bleu_output_file):
                        with open(bleu_output_file, "r", encoding='utf-8') as f:
                            bleu_score = f.read().strip()
                            print(f"BLEU Score: {bleu_score}")
                    else:
                        print("Warning: BLEU score file was not generated.")
                else:
                    print("Warning: Detokenized output file was not generated.")
            else:
                print("Warning: Translation output file was not generated.")
                
        except Exception as e:
            print(f"Error during translation process: {str(e)}")
            traceback.print_exc()
    else:
        print(f"Model file {model_path_translate} not found. Skipping translation.")
        print("You may need to train the KD model first or adjust num_steps_to_run_kd.")

    # --- Display figures (if available) ---
    print("\n--- Stage 5: Displaying Figures (if available) ---")
    figures_to_display = {
        'CMLM Finetuning': 'figures/cmlm-finetuning.png',
        'Translation Losses': 'figures/translation-losses.png',
        'Translation Accuracy': 'figures/translation-accuracy.png'
    }
    
    # Check if figures exist before trying to plot
    existing_figures = {
        title: path for title, path in figures_to_display.items() if os.path.exists(path)
    }

    if existing_figures:
        num_figs = len(existing_figures)
        fig, axes = plt.subplots(1, num_figs, figsize=(6 * num_figs, 5))
        if num_figs == 1: # Ensure axes is iterable
            axes = [axes] 
        
        for i, (title, path) in enumerate(existing_figures.items()):
            axes[i].set_title(title)
            try:
                img = plt.imread(path)
                axes[i].imshow(img)
                axes[i].axis('off')
            except Exception as e:
                print(f"Could not load or display figure {path}: {e}")
                axes[i].text(0.5, 0.5, 'Image not found or unreadable', ha='center', va='center')
                axes[i].axis('off')
        
        plt.tight_layout()
        # To show plots in a script, you might need plt.show()
        # However, if running in an environment that auto-displays (like some IDEs), it might not be needed.
        # For script execution, plt.show() is typical.
        # In Kaggle, plots usually show up automatically if `%matplotlib inline` was used,
        # but for a .py script, plt.show() is more explicit.
        try:
            plt.show() 
            print("Displayed figures. Close plot window to continue.")
        except Exception as e:
            # This can happen in non-GUI environments.
            print(f"Could not show plots (e.g., no GUI environment): {e}")
            # Optionally save the figure instead:
            # plot_output_path = "output/summary_figures.png"
            # fig.savefig(plot_output_path)
            # print(f"Saved summary figures to {plot_output_path}")

    else:
        print("No figures found in the 'figures' directory. Skipping display.")

    print("\n--- Script execution finished ---")


if __name__ == "__main__":
    main()
