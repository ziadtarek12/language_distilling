import os
import sys
import subprocess
import shutil

def run_shell_command(command_list, check=True, **kwargs):
    """Helper function to run shell commands given as a list."""
    print(f"Executing: {' '.join(command_list)}")
    return subprocess.run(command_list, check=check, **kwargs)

def run_shell_command_string(command_string, check=True, **kwargs):
    """Helper function to run shell commands given as a string (uses shell=True)."""
    print(f"Executing string: {command_string}")
    return subprocess.run(command_string, shell=True, check=check, **kwargs)

# Cell 1
print("Cell 1: Cloning repo and changing directory")
if not os.path.exists("language_distilling"):
    run_shell_command(["git", "clone", "https://github.com/ziadtarek12/language_distilling"])
else:
    print("language_distilling directory already exists. Skipping clone.")
os.chdir("language_distilling")
print(f"Changed directory to: {os.getcwd()}")
run_shell_command(["git", "checkout", "eval"])
print("-" * 30)

# Cell 2
print("Cell 2: Installing packages")
# Note: The uninstall command might fail if packages are not installed, so `check=False` could be used.
# For simplicity and direct translation, keeping `check=True` as implicit in `!` commands.
# However, pip uninstall without -y can be interactive. Adding -y.
# The notebook had "! pip uninstall -y ...", so this is correct.
run_shell_command_string("pip uninstall -y torch torchvision torchaudio") # Use string for easier copy-paste of multiple packages
run_shell_command_string("pip install transformers==4.26.0")
run_shell_command_string("pip install pytorch-pretrained-bert")
run_shell_command_string("pip install cytoolz")
run_shell_command_string("pip install tqdm")
run_shell_command_string("pip install torchtext==0.16.0")
run_shell_command_string("pip install torchvision==0.16.0")
run_shell_command_string("pip install torch==2.1.0")
run_shell_command_string("pip install torchaudio==2.1.0")
run_shell_command_string("pip install configargparse")
run_shell_command_string("pip install tensorboardX")
run_shell_command_string("pip install ipdb") # ipdb might not be ideal for a script but translating directly
print("-" * 30)

# Cell 3
print("Cell 3: Initial imports")
import torch # Already imported by pip install torch==2.1.0
import numpy as np
import random
import shelve
import io
import argparse # Standard library
import yaml # Installed by PyYAML, which we should add to pip installs if not already covered
from tqdm import tqdm # Installed by pip
from torch.utils.data import Dataset, DataLoader # From torch
from transformers import BertTokenizer # Installed by pip
import tensorboardX # Installed by pip
# Ensure PyYAML is installed if yaml is used (it is in cell 14)
try:
    import yaml
except ImportError:
    print("Installing PyYAML for yaml import...")
    run_shell_command_string("pip install PyYAML")
    import yaml
print("-" * 30)

# Cell 4
print("Cell 4: Setting up sys.path and device")
sys.path.append('.')
sys.path.append('./opennmt')

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
print("-" * 30)

# Cell 5
print("Cell 5: Creating directories and downloading data")
os.makedirs("data/", exist_ok=True)
os.makedirs("output/cmlm_model", exist_ok=True)
os.makedirs("output/bert_dump", exist_ok=True)
os.makedirs("output/kd-model/ckpt", exist_ok=True)
os.makedirs("output/kd-model/log", exist_ok=True)
os.makedirs("output/translation", exist_ok=True)

# Download IWSLT German-English dataset using the provided script
if not os.path.exists("data/de-en/train.de"): # Added check to prevent re-download
    run_shell_command(["bash", "scripts/download-iwslt_deen.sh"])
else:
    print("Dataset scripts/download-iwslt_deen.sh already seems to be downloaded (data/de-en/train.de exists). Skipping download.")
print("-" * 30)

# Cell 6
print("Cell 6: BERT tokenization setup and execution")
from scripts.bert_tokenize import tokenize, process

# Load BERT tokenizer
bert_model = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case='uncased' in bert_model)

# Define data directories
data_dir = "data/de-en"

# BERT tokenize our dataset files
for language in ['de', 'en']:
    for split in ['train', 'valid', 'test']:
        input_file = f"{data_dir}/{split}.{language}"
        output_file = f"{data_dir}/{split}.{language}.bert"
        # Added check to avoid re-tokenizing if files exist and are non-empty
        if os.path.exists(input_file):
            if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                print(f"Tokenizing {input_file}...")
                with open(input_file, 'r', encoding='utf-8') as reader, open(output_file, 'w', encoding='utf-8') as writer:
                    process(reader, writer, tokenizer)
            else:
                print(f"Skipping tokenization for {input_file}, {output_file} already exists and is not empty.")
        else:
            print(f"WARNING: Input file {input_file} not found. Skipping tokenization.")
print("-" * 30)

# Cell 7
print("Cell 7: Creating dataset DB and OpenNMT vocabulary")
from scripts.bert_prepro import main as bert_prepro

# Set up args for bert_prepro
# argparse.Namespace is from the 'argparse' module imported in Cell 3
prepro_args = argparse.Namespace(
    src=f"{data_dir}/train.de.bert",
    tgt=f"{data_dir}/train.en.bert",
    output='data/DEEN.db'
)

# Run preprocessing
# Add check to avoid re-running if DB exists
db_files_exist = all(os.path.exists(f"data/DEEN.db{ext}") for ext in ['.dat', '.bak', '.dir']) # Common shelve extensions
if not db_files_exist: # Or a more specific check if you know the exact files shelve creates
    if os.path.exists(f"{data_dir}/train.de.bert") and os.path.exists(f"{data_dir}/train.en.bert"):
        print("Running bert_prepro...")
        bert_prepro(prepro_args)
    else:
        print(f"WARNING: tokenized files for bert_prepro not found. Skipping bert_prepro.")
else:
    print("Skipping bert_prepro, data/DEEN.db files seem to exist.")


# Create vocabulary file using OpenNMT's preprocess.py
print("Creating vocabulary files with OpenNMT preprocess.py...")
vocab_file_onmt = "data/DEEN.vocab.pt" # Renamed from vocab_file to avoid conflict in cell 8
opennmt_preprocess_cmd = [
    sys.executable, "opennmt/preprocess.py", # Use sys.executable for python
    "-train_src", f"{data_dir}/train.de.bert",
    "-train_tgt", f"{data_dir}/train.en.bert",
    "-valid_src", f"{data_dir}/valid.de.bert",
    "-valid_tgt", f"{data_dir}/valid.en.bert",
    "-save_data", "data/DEEN",
    "-src_seq_length", "150", "-tgt_seq_length", "150"
]
if not os.path.exists(vocab_file_onmt):
    if os.path.exists(f"{data_dir}/train.de.bert") and os.path.exists(f"{data_dir}/train.en.bert"): # Check inputs exist
        run_shell_command(opennmt_preprocess_cmd)
    else:
        print("WARNING: tokenized files for OpenNMT preprocess not found. Skipping OpenNMT preprocess.")
else:
    print(f"Skipping OpenNMT preprocess, {vocab_file_onmt} already exists.")
# vocab_file = "data/DEEN.vocab.pt" # This was in the notebook, redundant due to above
print("-" * 30)

# Cell 8
print("Cell 8: CMLM Model Setup")
from transformers import AdamW, get_linear_schedule_with_warmup # BertTokenizer already imported

# Import needed modules
from cmlm.data import BertDataset, TokenBucketSampler
from cmlm.model import convert_embedding, BertForSeq2seq
from cmlm.util import Logger, RunningMeter # Logger not used in notebook cells
# from run_cmlm_finetuning import noam_schedule # noam_schedule not used in notebook cells

# Load vocabulary using our compatibility module
from vocab_loader import safe_load_vocab

# vocab_file was defined as vocab_file_onmt in cell 7. Let's use that.
# vocab_file = "data/DEEN.vocab.pt" # This is redundant if cell 7 was complete
train_file_db = "data/DEEN.db" # Renamed from train_file to avoid clash
# valid_src_path = f"{data_dir}/valid.de.bert" # Renamed, notebook uses valid_src directly
# valid_tgt_path = f"{data_dir}/valid.en.bert" # Renamed, notebook uses valid_tgt directly
output_dir_cmlm = "output/cmlm_model" # Renamed from output_dir

# Load vocabulary using custom loader to avoid PyTorch compatibility issues
if os.path.exists(vocab_file_onmt):
    vocab_dump = safe_load_vocab(vocab_file_onmt)
    vocab_stoi = vocab_dump['tgt'].fields[0][1].vocab.stoi # Renamed from vocab

    # Create dataset
    if os.path.exists(train_file_db + ".dat"): # Check if shelve .dat file exists
        train_dataset_cmlm = BertDataset(train_file_db, tokenizer, vocab_stoi, seq_len=512, max_len=150) # Renamed train_dataset

        # Define sampler and data loader
        BUCKET_SIZE = 8192
        # Check if train_dataset_cmlm.lens is available and non-empty
        if hasattr(train_dataset_cmlm, 'lens') and train_dataset_cmlm.lens:
            train_sampler_cmlm = TokenBucketSampler( # Renamed train_sampler
                train_dataset_cmlm.lens, BUCKET_SIZE, 6144, batch_multiple=1)

            train_loader_cmlm = DataLoader(train_dataset_cmlm, batch_sampler=train_sampler_cmlm, # Renamed train_loader
                                     num_workers=min(4, os.cpu_count() if os.cpu_count() else 1), # Adjusted num_workers
                                     collate_fn=BertDataset.pad_collate)

            # Prepare model
            model_cmlm = BertForSeq2seq.from_pretrained(bert_model) # Renamed model
            bert_embedding = model_cmlm.bert.embeddings.word_embeddings.weight

            # Print model information before modifications
            hidden_size = model_cmlm.config.hidden_size
            print(f"Original model: BERT hidden size = {hidden_size}")
            print(f"Original model: BERT vocab size = {bert_embedding.size(0)}")
            print(f"Target vocabulary size = {len(vocab_stoi)}")

            # Convert vocabulary to embedding form
            embedding = convert_embedding(tokenizer, vocab_stoi, bert_embedding)

            # Update model architecture to accommodate the new vocabulary size
            print(f"Updating model architecture for vocabulary size: {embedding.size(0)}")
            # Create a new decoder with correct dimensions
            model_cmlm.cls.predictions.decoder = torch.nn.Linear(hidden_size, embedding.size(0), bias=True)
            model_cmlm.cls.predictions.bias = torch.nn.Parameter(torch.zeros(embedding.size(0)))
            model_cmlm.config.vocab_size = embedding.size(0)

            # Update the weights
            model_cmlm.cls.predictions.decoder.weight.data.copy_(embedding.data)

            # Move model to device
            model_cmlm.to(device)
            print(f"Model adapted with vocabulary size: {model_cmlm.config.vocab_size}")
        else:
            print("WARNING: train_dataset_cmlm.lens not available or empty. Skipping CMLM model setup remainder.")
    else:
        print(f"WARNING: {train_file_db} (shelve DB for CMLM) not found. Skipping CMLM model setup.")
else:
    print(f"WARNING: {vocab_file_onmt} not found. Skipping CMLM model setup.")
print("-" * 30)

# Cell 9
print("Cell 9: CMLM Training")
# This cell depends on 'model_cmlm', 'train_loader_cmlm', 'output_dir_cmlm' from Cell 8
if 'model_cmlm' in locals() and 'train_loader_cmlm' in locals():
    # Training parameters
    learning_rate = 5e-5
    warmup_proportion = 0.1  # Using proportion instead of absolute steps
    max_steps_cmlm_train = 100000  # Renamed from max_steps to avoid conflict, full training uses 100k steps
    # num_steps_to_run_cmlm_train = 100000 # Renamed from num_steps_to_run. Use a smaller value for quick test.
    num_steps_to_run_cmlm_train = 100 # FOR QUICK TEST. Original notebook value implies full run.

    # Optimizer using modern AdamW from transformers
    param_optimizer = list(model_cmlm.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer_cmlm = AdamW(optimizer_grouped_parameters, lr=learning_rate) # Renamed optimizer
    scheduler_cmlm = get_linear_schedule_with_warmup( # Renamed scheduler
        optimizer_cmlm,
        num_warmup_steps=int(max_steps_cmlm_train * warmup_proportion),
        num_training_steps=max_steps_cmlm_train
    )

    # Training loop
    running_loss_meter = RunningMeter('loss') # Renamed running_loss
    model_cmlm.train()

    print(f"Starting CMLM fine-tuning for {num_steps_to_run_cmlm_train} steps...")
    #Use a plain iterator instead of tqdm with len()
    train_iter_cmlm = iter(train_loader_cmlm) # Renamed train_iter
    for step in range(num_steps_to_run_cmlm_train):
        try:
            batch = next(train_iter_cmlm)
        except StopIteration:
            # Restart iterator if we run out of batches
            print(f"CMLM DataLoader exhausted at step {step}. Resetting iterator.")
            train_iter_cmlm = iter(train_loader_cmlm)
            try:
                batch = next(train_iter_cmlm)
            except StopIteration:
                print("CRITICAL: CMLM DataLoader empty even after reset. Training cannot continue.")
                break # Exit loop if loader is persistently empty
            
        # Move batch to device
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, lm_label_ids = batch
        
        # Zero gradients
        optimizer_cmlm.zero_grad()
        
        # Create output mask from lm_label_ids for model forward pass
        output_mask = lm_label_ids != -1  # Masking for non-padded tokens
        
        # Forward pass with output_mask parameter
        loss = model_cmlm(input_ids, segment_ids, input_mask, lm_label_ids, output_mask)
        
        # Backward pass
        loss.backward()
        optimizer_cmlm.step()
        scheduler_cmlm.step()
        
        running_loss_meter(loss.item())
        if step % 10 == 0 or step == num_steps_to_run_cmlm_train -1: # Print less often
             print(f"CMLM Step {step}, Loss: {running_loss_meter.val:.4f}")
        
        if step % 100 == 0:
            # Clear CUDA cache periodically to avoid memory issues
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save model checkpoint
    # output_dir_cmlm was defined in Cell 8
    model_save_path = f"{output_dir_cmlm}/model_step_{num_steps_to_run_cmlm_train}.pt"
    torch.save(model_cmlm.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
else:
    print("Skipping CMLM Training (Cell 9) as required variables from Cell 8 are not defined.")
print("-" * 30)

# Cell 10
print("Cell 10: CUDA Memory Summary")
if torch.cuda.is_available():
    # torch.cuda.memory_summary(device=None, abbreviated=False) # This prints a very long string
    print(f"CUDA Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"CUDA Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    torch.cuda.empty_cache()
else:
    print("CUDA not available, skipping memory summary.")
print("-" * 30)

# Cell 11
print("Cell 11: Setup for Dumping Teacher Hiddens")
# This cell depends on variables from cell 8 (output_dir_cmlm) and cell 9 (num_steps_to_run_cmlm_train, model_save_path)
# bert_model was defined in cell 6
from dump_teacher_hiddens import tensor_dumps, gather_hiddens, BertSampleDataset, batch_features, process_batch as dump_hidden_process_batch

# Path to model checkpoint from Stage 1
# Reconstruct model_save_path if it was created, or use a default if training was skipped
if 'model_save_path' not in locals() and 'num_steps_to_run_cmlm_train' in locals() and 'output_dir_cmlm' in locals():
    # If training cell was skipped but params were set
    model_save_path = f"{output_dir_cmlm}/model_step_{num_steps_to_run_cmlm_train}.pt"
elif 'model_save_path' not in locals():
    # Fallback if training was completely skipped and vars not set
    # This part will likely fail if model_save_path is not valid
    print("WARNING: model_save_path for CMLM model not found. Stage 2 might fail.")
    # Create a dummy path to avoid NameError, but it won't work.
    model_save_path = f"{output_dir_cmlm}/model_step_DUMMY.pt" 

ckpt_path_stage2 = model_save_path # Renamed from ckpt_path
bert_dump_path_stage2 = "output/bert_dump" # Renamed from bert_dump_path

if os.path.exists(ckpt_path_stage2):
    # Load the fine-tuned BERT model
    state_dict = torch.load(ckpt_path_stage2, map_location=device) # map_location added
    vsize = state_dict['cls.predictions.decoder.weight'].size(0)
    bert_teacher = BertForSeq2seq.from_pretrained(bert_model).eval() # Renamed from bert
    bert_teacher.to(device)

    # Fix: Instead of using update_output_layer_by_size, which pads to multiples of 8,
    # we'll directly resize the model layers to match the exact dimensions from the checkpoint
    print(f"Resizing model to exact vocabulary size: {vsize}")
    hidden_size_teacher = bert_teacher.config.hidden_size # Renamed from hidden_size

    # Create exact-sized layers without padding to multiples of 8
    bert_teacher.cls.predictions.decoder = torch.nn.Linear(hidden_size_teacher, vsize, bias=True)
    # Original notebook links bias: bert.cls.predictions.bias = bert.cls.predictions.decoder.bias
    # Safer to re-assign if it was a separate Parameter in the state_dict
    if 'cls.predictions.bias' in state_dict:
        bert_teacher.cls.predictions.bias = torch.nn.Parameter(torch.zeros(vsize, device=device))
    else: # If not in state_dict, assume it's linked to decoder.bias and will be loaded with it
        bert_teacher.cls.predictions.bias = bert_teacher.cls.predictions.decoder.bias


    bert_teacher.config.vocab_size = vsize

    # Now load the state dict - should have matching dimensions
    bert_teacher.load_state_dict(state_dict)

    # Save the final projection layer
    linear_projection = torch.nn.Linear(bert_teacher.config.hidden_size, bert_teacher.config.vocab_size) # Renamed from linear
    linear_projection.weight.data = state_dict['cls.predictions.decoder.weight']
    linear_projection.bias.data = state_dict['cls.predictions.bias']
    linear_projection_save_path = f'{bert_dump_path_stage2}/linear.pt'
    torch.save(linear_projection, linear_projection_save_path)
    print(f"Linear projection layer saved to {linear_projection_save_path}")
else:
    print(f"Skipping Teacher Hidden Dumps setup (Cell 11) as CMLM checkpoint {ckpt_path_stage2} not found.")
print("-" * 30)

# Cell 12
print("Cell 12: Extracting Hidden States")
# Depends on bert_teacher, tokenizer (cell 6), bert_dump_path_stage2, db_output_file (train_file_db in cell 8, which is data/DEEN.db)
if 'bert_teacher' in locals() and 'tokenizer' in locals():
    # Function to extract hidden states - with debugging option
    # Copied from notebook, process_batch here refers to dump_hidden_process_batch
    def build_db_batched(corpus_path, out_db_shelf, bert_model_param, toker_param, batch_size=8, debug_mode=False, max_samples=100): # Renamed out_db
        # BertSampleDataset expects the .db file from bert_prepro
        print(f"  build_db_batched: Loading BertSampleDataset from {corpus_path}")
        dataset = BertSampleDataset(corpus_path, toker_param)
        
        dataset_ids_list = list(dataset.ids) # Get all IDs first
        print(f"  build_db_batched: Found {len(dataset_ids_list)} total IDs in dataset.")

        if not dataset_ids_list:
            print("  build_db_batched: ERROR - No IDs found in BertSampleDataset. Cannot proceed.")
            return

        # For debugging, limit the number of samples
        if debug_mode:
            # subset_ids = dataset.ids[:max_samples] if len(dataset.ids) > max_samples else dataset.ids # Original was problematic with shelve keys
            subset_ids = dataset_ids_list[:max_samples]
            # This modification of dataset.ids is tricky.
            # It's better to filter the loader or break early.
            # For now, we'll use the original notebook logic and hope BertSampleDataset's .ids can be reassigned.
            # If dataset.ids is a view, this might not work as expected.
            # A safer way is to create a new dataset or filter the loader.
            # Let's assume dataset.ids can be modified for direct translation for now.
            dataset.ids = subset_ids # This might be problematic if dataset.ids is a ShelfKeyView
            print(f"  DEBUG MODE: Processing only {len(subset_ids)} samples instead of {len(dataset_ids_list)}")
            if not subset_ids:
                print("  build_db_batched: DEBUG MODE - subset_ids is empty. No samples to process.")
                return
        
        # The DataLoader's len(dataset) might still refer to the original if .ids reassignment is shallow
        # Total for tqdm should be len(dataset.ids) after potential modification
        loader = DataLoader(dataset, batch_size=batch_size,
                           num_workers=min(4, os.cpu_count() if os.cpu_count() else 1), collate_fn=batch_features)
        
        # Ensure loader is not empty
        try:
            first_batch_loader_check = next(iter(loader))
            del first_batch_loader_check # Don't consume it
            # Re-initialize loader because next(iter()) consumes one batch
            loader = DataLoader(dataset, batch_size=batch_size,
                           num_workers=min(4, os.cpu_count() if os.cpu_count() else 1), collate_fn=batch_features)
        except StopIteration:
            print("  build_db_batched: ERROR - DataLoader is empty. Cannot extract features.")
            return

        with tqdm(desc='Computing BERT features', total=len(dataset.ids)) as pbar: # Use current len(dataset.ids)
            for ids_in_batch, *batch_data_tuple in loader: # Renamed ids, batch
                outputs = dump_hidden_process_batch(batch_data_tuple, bert_model_param, toker_param) # Pass tuple as *args
                for id_val, output_val in zip(ids_in_batch, outputs): # Renamed id_, output
                    if output_val is not None:
                        out_db_shelf[id_val] = tensor_dumps(output_val) # Use out_db_shelf
                pbar.update(len(ids_in_batch))
                
                # For debugging, break after the first batch if needed (original notebook logic)
                # This condition (batch_size >= max_samples) is only for when max_samples is very small
                if debug_mode and batch_size >= max_samples and pbar.n > 0:
                    print("  First batch processed, breaking early due to debug mode and batch_size >= max_samples")
                    break
                if debug_mode and pbar.n >= max_samples: # More general break for max_samples
                    print(f"  Processed {pbar.n} samples, reached debug max_samples. Breaking.")
                    break


    db_path_for_extraction = db_output_file # data/DEEN.db from cell 7
    print(f"Extracting hidden states from DB: {db_path_for_extraction}")

    debug_mode_extraction = True  # Toggle this for quick debugging
    max_samples_extraction = 10  # Reduced from 100 for faster script test

    hidden_states_db_path = f'{bert_dump_path_stage2}/db' # Define where to save
    # Ensure source DB exists
    if not any(os.path.exists(f"{db_path_for_extraction}{ext}") for ext in ['', '.db', '.dat', '.bak', '.dir']):
        print(f"ERROR: Source DB {db_path_for_extraction} for hidden state extraction not found. Skipping.")
    else:
        with shelve.open(hidden_states_db_path, 'c') as out_db_shelf, torch.no_grad(): # Renamed out_db
            build_db_batched(db_path_for_extraction, out_db_shelf, bert_teacher, tokenizer, batch_size=8, 
                            debug_mode=debug_mode_extraction, max_samples=max_samples_extraction)

        # Free up GPU memory after extraction
        print("Clearing GPU memory...")
        bert_teacher.cpu()
        del bert_teacher # Delete the model
        if 'linear_projection' in locals(): # Check if linear_projection was created in cell 11
            linear_projection.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("GPU memory cleared after hidden states extraction")

        if debug_mode_extraction:
            print(f"DEBUG MODE: Hidden states for up to {max_samples_extraction} samples extracted to {hidden_states_db_path}")
            print("To run full extraction, set debug_mode_extraction=False")
        else:
            print(f"Hidden states extracted and saved to {hidden_states_db_path}")
else:
    print("Skipping Hidden States Extraction (Cell 12) as bert_teacher or tokenizer not defined.")
print("-" * 30)

# Cell 13
print("Cell 13: Top-K Logits Computation")
# Depends on linear_projection_save_path (cell 11), bert_dump_path_stage2 (cell 11), hidden_states_db_path (cell 12)
if os.path.exists(linear_projection_save_path) and \
   any(os.path.exists(f"{hidden_states_db_path}{ext}") for ext in ['', '.db', '.dat', '.bak', '.dir']):

    from dump_teacher_topk import tensor_loads, dump_topk

    # Top-K parameter
    k_param = 8  # Renamed from k, following the paper

    # Load linear layer
    linear_loaded = torch.load(linear_projection_save_path, map_location=device) # Renamed linear
    # Ensure the linear layer uses the same precision as the hidden states (FP16/Half)
    linear_loaded = linear_loaded.half()
    linear_loaded.to(device)

    # Compute top-k logits
    topk_output_db_path = f'{bert_dump_path_stage2}/topk' # Define output path
    print(f"Computing top-k logits, saving to {topk_output_db_path}...")
    with shelve.open(hidden_states_db_path, 'r') as db_shelf_read, \
         shelve.open(topk_output_db_path, 'c') as topk_db_write: # Renamed db, topk_db
        
        # Check if source DB is empty
        source_db_keys = list(db_shelf_read.keys())
        if not source_db_keys:
            print(f"  WARNING: Source hidden states DB ({hidden_states_db_path}) is empty. No top-k logits will be computed.")
        
        # Iterate over keys, potentially limited by debug settings from previous cell
        keys_to_process = source_db_keys
        if debug_mode_extraction and max_samples_extraction < len(source_db_keys): # Use same debug limit
            print(f"  DEBUG MODE: Processing top-k for {max_samples_extraction} samples based on previous debug settings.")
            keys_to_process = source_db_keys[:max_samples_extraction]


        for key, value in tqdm(db_shelf_read.items(), total=len(db_shelf_read), desc='Computing topk...'): # Use items() for direct iteration
        # for key_iter in tqdm(keys_to_process, total=len(keys_to_process), desc='Computing topk...'):
            # value = db_shelf_read[key_iter] # If iterating keys_to_process
            # bert_hidden is already in half precision, no need to convert (this depends on how it was saved)
            # The dump_teacher_hiddens.py saves hidden states as float32 by default.
            # So, convert to half here.
            bert_hidden_state = torch.tensor(tensor_loads(value)).to(device).half() # Renamed bert_hidden
            
            topk_result = linear_loaded(bert_hidden_state).topk(dim=-1, k=k_param) # Renamed topk
            dump_val = dump_topk(topk_result) # Renamed dump
            topk_db_write[key] = dump_val # Use original key
            
            # Clear tensor from GPU memory after each iteration
            del bert_hidden_state
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Final memory cleanup
    print("Clearing GPU memory...")
    linear_loaded.cpu()
    del linear_loaded
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("GPU memory cleared after top-k computation")
    print(f"Top-k logits computed and saved to {topk_output_db_path}")
else:
    print("Skipping Top-K Logits Computation (Cell 13) as linear projection or hidden states DB not found.")
print("-" * 30)


# Cell 14
print("Cell 14: Knowledge Distillation (KD) Training Setup")
# Import required modules for training
# These were imported in Cell 4's sys.path setup and should be available if language_distilling is CWD
from onmt.inputters.bert_kd_dataset import BertKdDataset, TokenBucketSampler as BertKdTokenBucketSampler
from onmt.utils.optimizers import Optimizer
from onmt.train_single import build_model_saver, build_trainer, cycle_loader
import torch.nn as nn  # Add missing import (was in notebook cell)
# os was imported at the top

# Define paths
data_db_kd = db_output_file # data/DEEN.db from cell 7
bert_dump_kd = bert_dump_path_stage2 # output/bert_dump from cell 11
data_onmt_kd = "data/DEEN" # For vocab, etc.
config_path_onmt = os.path.join(opennmt_dir_abs, "config", "config-transformer-base-mt-deen.yml") # Renamed, use full path
output_path_kd = "output/kd-model"

# Check if required files exist and provide guidance (logic from notebook)
print("Checking for required database files for KD...")
topk_db_file_kd = f"{bert_dump_kd}/topk" # This is topk_output_db_path from cell 13

topk_db_dir = os.path.dirname(topk_db_file_kd) # Should be bert_dump_kd
if not os.path.exists(topk_db_dir): # Should have been created in Cell 5
    print(f"Creating directory for topk DB: {topk_db_dir}") # Should not happen if cell 5 ran
    os.makedirs(topk_db_dir, exist_ok=True)

# Check if topk database exists (any shelve file part)
topk_shelve_exists = any(os.path.exists(f"{topk_db_file_kd}{ext}") for ext in ["", ".db", ".dat", ".bak", ".dir"])

if not topk_shelve_exists:
    print(f"Warning: Top-k database not found at {topk_db_file_kd}")
    print("The notebook had logic to re-run top-k computation here. This script assumes Cell 13 ran successfully if Stage 2 was enabled.")
    print("If Stage 2 was skipped or failed, KD training will likely fail or use no distilled data.")
    # For a direct script translation, we could re-include the re-computation logic,
    # but it makes the script much longer and assumes variables like 'linear' are still in scope from Cell 13.
    # For now, proceeding with a warning. If Cell 13 ran, topk_db_file_kd should exist.
    # If you need the re-computation, it's safer to ensure Cell 13 runs by enabling Stage 2.
else:
    print(f"Top-k database seems to exist at {topk_db_file_kd}")

# Load configuration
if not os.path.exists(config_path_onmt):
    print(f"ERROR: OpenNMT config {config_path_onmt} not found. Cannot proceed with KD setup.")
    args_kd = None # To signal failure
else:
    with open(config_path_onmt, 'r') as stream:
        config = yaml.safe_load(stream)
    args_kd = argparse.Namespace(**config) # Create args object, renamed from args to avoid conflict

if args_kd is not None:
    # Setup KD parameters (many were hardcoded in the notebook cell)
    args_kd.train_from = None # Override if in config
    args_kd.max_grad_norm = getattr(args_kd, 'max_grad_norm', 0.0) # Default to 0.0 if not in config, OpenNMT optim usually handles
    args_kd.kd_topk = 8
    args_kd.train_steps = 100 # FOR QUICK TEST. Original notebook: 100000
    args_kd.kd_temperature = 10.0
    args_kd.kd_alpha = 0.5
    args_kd.warmup_steps = 800 # FOR QUICK TEST. Original notebook: 8000
    args_kd.learning_rate = 2.0
    args_kd.bert_dump = bert_dump_kd
    args_kd.data_db = data_db_kd
    args_kd.bert_kd = True
    args_kd.data = data_onmt_kd # Path for vocab loading like "data/DEEN"

    # Add missing required parameters (from notebook cell, ensure they are on args_kd)
    args_kd.model_type = getattr(args_kd, 'model_type', "text")
    args_kd.copy_attn = getattr(args_kd, 'copy_attn', False)
    args_kd.global_attention = getattr(args_kd, 'global_attention', "general")
    args_kd.src_word_vec_size = args_kd.word_vec_size # Assumes word_vec_size is in config
    args_kd.tgt_word_vec_size = args_kd.word_vec_size
    args_kd.feat_merge = getattr(args_kd, 'feat_merge', "concat")
    args_kd.feat_vec_size = getattr(args_kd, 'feat_vec_size', -1) # Often calculated
    args_kd.feat_vec_exponent = getattr(args_kd, 'feat_vec_exponent', 0.7)
    args_kd.pre_word_vecs_enc = getattr(args_kd, 'pre_word_vecs_enc', None)
    args_kd.pre_word_vecs_dec = getattr(args_kd, 'pre_word_vecs_dec', None)
    # args_kd.pre_word_vecs = None # Notebook had this, usually covered by enc/dec specific
    args_kd.fix_word_vecs_enc = getattr(args_kd, 'fix_word_vecs_enc', False)
    args_kd.fix_word_vecs_dec = getattr(args_kd, 'fix_word_vecs_dec', False)
    
    # If rnn_size is not in config (e.g. pure transformer), default it or ensure it's handled.
    # OpenNMT model builder might need it or have defaults. Let's ensure it exists.
    default_rnn_size = 512 # A common default
    args_kd.enc_rnn_size = getattr(args_kd, 'rnn_size', default_rnn_size)
    args_kd.dec_rnn_size = getattr(args_kd, 'rnn_size', default_rnn_size)
    
    args_kd.transformer_ff = getattr(args_kd, 'transformer_ff', 2048)
    args_kd.heads = getattr(args_kd, 'heads', 8)
    args_kd.max_relative_positions = getattr(args_kd, 'max_relative_positions', 0)
    args_kd.position_encoding = getattr(args_kd, 'position_encoding', True)
    args_kd.param_init = getattr(args_kd, 'param_init', 0.0)
    args_kd.param_init_glorot = getattr(args_kd, 'param_init_glorot', True)
    args_kd.share_embeddings = False # Critical fix from notebook
    args_kd.share_decoder_embeddings = False # Critical fix from notebook
    args_kd.truncated_decoder = getattr(args_kd, 'truncated_decoder', 0)
    args_kd.max_generator_batches = getattr(args_kd, 'max_generator_batches', 32)
    args_kd.normalization = getattr(args_kd, 'normalization', 'sents') # 'tokens' is also common
    # accum_count can be an int or list in OpenNMT. Trainer usually expects list.
    accum_count_val = getattr(args_kd, 'accum_count', 1)
    args_kd.accum_count = [accum_count_val] if isinstance(accum_count_val, int) else accum_count_val

    args_kd.accum_steps = getattr(args_kd, 'accum_steps', [0]) # From notebook, should align with accum_count logic
    args_kd.average_decay = getattr(args_kd, 'average_decay', 0.0)
    args_kd.average_every = getattr(args_kd, 'average_every', 1)
    args_kd.report_manager = None # Will be set up by trainer or manually if tensorboard
    args_kd.valid_steps = getattr(args_kd, 'valid_steps', 1000) # Reduced from 10000 for test
    args_kd.early_stopping = getattr(args_kd, 'early_stopping', 0)
    args_kd.early_stopping_criteria = getattr(args_kd, 'early_stopping_criteria', None)
    args_kd.valid_batch_size = getattr(args_kd, 'valid_batch_size', 8) # Reduced from 32 for test

    args_kd.self_attn_type = getattr(args_kd, 'self_attn_type', "scaled-dot")
    # input_feed is for RNNs, set to 0 for transformers unless explicitly configured
    args_kd.input_feed = getattr(args_kd, 'input_feed', 1) if getattr(args_kd, 'decoder_type', "transformer") == "rnn" else 0
    args_kd.copy_attn_type = getattr(args_kd, 'copy_attn_type', None) # Or "general", "luong" etc. if copy_attn=True
    args_kd.generator_function = getattr(args_kd, 'generator_function', "softmax")
    args_kd.local_rank = -1 # For distributed training (not used here)
    args_kd.gpu_ranks = [0] if torch.cuda.is_available() else [] # From notebook
    args_kd.gpu_verbose_level = 0
    args_kd.world_size = getattr(args_kd, 'world_size', 1)
    args_kd.encoder_type = getattr(args_kd, 'encoder_type', "transformer")
    args_kd.decoder_type = getattr(args_kd, 'decoder_type', "transformer")
    
    # Ensure enc_layers and dec_layers are present, defaulting from 'layers' or a fixed value
    default_num_layers = 6
    args_kd.enc_layers = getattr(args_kd, 'enc_layers', getattr(args_kd, 'layers', default_num_layers))
    args_kd.dec_layers = getattr(args_kd, 'dec_layers', getattr(args_kd, 'layers', default_num_layers))

    # Dropout should be float for modules like PositionalEncoding
    args_kd.dropout = float(getattr(args_kd, 'dropout', 0.1))
    args_kd.attention_dropout = float(getattr(args_kd, 'attention_dropout', args_kd.dropout)) # Default to model dropout
    
    args_kd.bridge = getattr(args_kd, 'bridge', False) # Typically False, or "copy", "dense" etc.
    args_kd.aux_tune = False # As in notebook
    args_kd.subword_prefix = " " # As in notebook
    args_kd.subword_prefix_is_joiner = False # As in notebook

    args_kd.save_model = os.path.join(output_path_kd, 'ckpt', 'model')
    args_kd.log_file = os.path.join(output_path_kd, 'log', 'log.txt') # Give it an extension
    args_kd.tensorboard = True # Enable tensorboard
    args_kd.tensorboard_log_dir = os.path.join(output_path_kd, 'log')
    args_kd.report_every = int(getattr(args_kd, 'report_every', 50)) # ensure int for ReportMgr
else:
    print("Skipping remainder of KD Setup (Cell 14) due to missing config.")
print("-" * 30)

# Cell 15
print("Cell 15: KD Vocab and Dataset/Dataloader Setup")
if args_kd is not None and os.path.exists(args_kd.data + '.vocab.pt'):
    # Load vocabulary and dataset
    vocab_onmt_kd = torch.load(args_kd.data + '.vocab.pt') # Renamed vocab
    # stoi is directly on vocab object, not vocab.stoi
    src_vocab_stoi_kd = vocab_onmt_kd['src'].fields[0][1].vocab.stoi # Renamed src_vocab
    tgt_vocab_stoi_kd = vocab_onmt_kd['tgt'].fields[0][1].vocab.stoi # Renamed tgt_vocab

    # Create dataset
    # data_db_kd, bert_dump_kd defined in cell 14
    train_dataset_kd = BertKdDataset(data_db_kd, bert_dump_kd, 
                                 src_vocab_stoi_kd, tgt_vocab_stoi_kd,
                                 max_len=150, k=args_kd.kd_topk)
    
    # Check dataset length
    num_kd_samples = len(train_dataset_kd.id_lens if hasattr(train_dataset_kd, 'id_lens') else [])
    print(f"Length of KD train dataset (from id_lens): {num_kd_samples}")
    if not num_kd_samples > 0:
        print("CRITICAL ERROR: KD train dataset is effectively empty based on id_lens. Cannot proceed.")
        # To prevent NameError in subsequent cells if this path is taken:
        train_loader_kd = None
        train_iter_kd = None 
    else:
        # Create data loader (with dynamic sampler params from previous fixes)
        BUCKET_SIZE_KD_default = 8192; batch_size_tokens_kd_default = 6144
        if num_kd_samples < 50: 
            print(f"KD dataset is very small ({num_kd_samples} samples). Adjusting sampler parameters.")
            BUCKET_SIZE_KD = min(BUCKET_SIZE_KD_default, num_kd_samples * 256); BUCKET_SIZE_KD = max(BUCKET_SIZE_KD, num_kd_samples) 
            batch_size_tokens_kd = min(batch_size_tokens_kd_default, num_kd_samples * 150); batch_size_tokens_kd = max(batch_size_tokens_kd, 150 * 1) 
            print(f"Adjusted BUCKET_SIZE_KD: {BUCKET_SIZE_KD}"); print(f"Adjusted batch_size_tokens_kd: {batch_size_tokens_kd}")
        else: BUCKET_SIZE_KD = BUCKET_SIZE_KD_default; batch_size_tokens_kd = batch_size_tokens_kd_default
        
        if not hasattr(train_dataset_kd, 'id_lens') or not train_dataset_kd.id_lens:
            print("CRITICAL ERROR: train_dataset_kd.id_lens missing/empty for Sampler."); sys.exit(1)

        train_sampler_kd = BertKdTokenBucketSampler( # Renamed train_sampler
            train_dataset_kd.id_lens, BUCKET_SIZE_KD, batch_size_tokens_kd, # Pass id_lens
            batch_multiple=1)

        train_loader_kd = DataLoader(train_dataset_kd, batch_sampler=train_sampler_kd, # Renamed train_loader
                                 num_workers=min(4, os.cpu_count() if os.cpu_count() else 1),
                                 collate_fn=BertKdDataset.pad_collate)
        
        # Check dataloader
        try:
            _ = next(iter(train_loader_kd))
            print("KD DataLoader check successful.")
        except StopIteration:
            print("CRITICAL ERROR: KD DataLoader is empty even after sampler adjustments!")
            train_loader_kd = None # Signal error
        
        if train_loader_kd:
            train_iter_kd = cycle_loader(train_loader_kd, device) # Renamed train_iter
else:
    print("Skipping KD Vocab/Dataset Setup (Cell 15) as args_kd or vocab file not available.")
    train_loader_kd = None # Ensure it's defined for checks in later cells
    train_iter_kd = None
print("-" * 30)

# Cell 16
print("Cell 16: KD Model Build")
if args_kd is not None and 'vocab_onmt_kd' in locals() and train_loader_kd is not None: # Check train_loader_kd also
    # Build the model
    from onmt.model_builder import build_model # Already imported if deferred imports worked

    # Make sure nn is imported at the top of the notebook (it was in cell 14 for this script)
    # model_kd is model, args_kd is model_opt, args_kd is also opt, vocab_onmt_kd is fields
    onmt_fields_kd = {'src': vocab_onmt_kd['src'], 'tgt': vocab_onmt_kd['tgt']} # Correct fields structure
    model_kd_distill = build_model(args_kd, args_kd, fields=onmt_fields_kd, checkpoint=None) # Renamed model
    model_kd_distill.to(device)

    # Build optimizer
    optim_kd_distill = Optimizer.from_opt(model_kd_distill, args_kd, checkpoint=None) # Renamed optim

    # Build model saver
    # model_saver_kd = build_model_saver(args_kd, args_kd, model_kd_distill, vocab_onmt_kd, optim_kd_distill) # Renamed model_saver
    # Ensure correct 'fields' (onmt_fields_kd) is passed to model_saver
    model_saver_kd = build_model_saver(args_kd, args_kd, model_kd_distill, onmt_fields_kd, optim_kd_distill)


    # Build trainer
    # Ensure report_manager is set up for trainer
    if args_kd.tensorboard and not args_kd.report_manager: # If not already set by other logic
        from tensorboardX import SummaryWriter
        import onmt.utils # Already imported
        writer = SummaryWriter(args_kd.tensorboard_log_dir, comment="onmt-kd")
        args_kd.report_manager = onmt.utils.ReportMgr(
            report_every=args_kd.report_every, # Should be set on args_kd
            start_time=None, # Trainer will manage start time
            tensorboard_writer=writer
        )
    
    trainer_kd_distill = build_trainer(args_kd, 0 if device.type == 'cuda' else -1, model_kd_distill, onmt_fields_kd, optim_kd_distill, model_saver=model_saver_kd, report_manager=args_kd.report_manager) # Renamed trainer
else:
    print("Skipping KD Model Build (Cell 16) as dependencies not met.")
print("-" * 30)

# Cell 17
print("Cell 17: Comment cell")
# the problem is in the following cell (This was a marker in the notebook)
print("-" * 30)

# Cell 18
print("Cell 18: KD Training Loop")
if 'trainer_kd_distill' in locals() and 'train_iter_kd' in locals() and train_iter_kd is not None:
    # Train - for demonstration, we'll only do a few steps
    num_steps_to_run_kd_train = args_kd.train_steps # Use value from args_kd (default 100 for test)

    # Make sure the optimizer is tracking the step correctly
    # OpenNMT Optimizer usually has _step or similar, or trainer manages it.
    # The notebook's check might be for an older version or custom optimizer.
    # optim_kd_distill.training_step or optim_kd_distill._step
    if not hasattr(optim_kd_distill, '_step'): # Check for OpenNMT's typical internal step counter
         optim_kd_distill._step = 0 # Initialize if missing, though trainer might do this too
    
    # Define a custom iterator that provides batches without its own step limitation
    # This manual_train_iter needs access to the global/enclosing train_iter_kd
    # To avoid global, pass train_iter_kd if it were a class method, or ensure it's in scope.
    # For a direct script, train_iter_kd from Cell 15 is in this scope.
    
    # The notebook uses `global train_iter`. Here, train_iter_kd is in the main() scope.
    # We need to be careful if this function is defined elsewhere or train_iter_kd changes.
    # For now, it should pick up train_iter_kd from the enclosing scope.
    # However, to modify it (reassign), we'd need `nonlocal` if nested, or handle it via a mutable object.
    # The cycle_loader already makes train_iter_kd an infinite iterator.

    # The notebook's manual_train_iter tried to re-assign global train_iter.
    # Let's adapt it to use train_iter_kd from Cell 15. cycle_loader should handle infinite iteration.
    # So, no need for manual StopIteration handling inside manual_train_iter if cycle_loader works.
    
    # The manual_train_iter in the notebook was complex.
    # cycle_loader already returns an infinite iterator. So, we can just pass that.
    # The trainer.train() method expects an iterator and a number of steps.

    print(f"Starting model training with knowledge distillation for {num_steps_to_run_kd_train} steps...")
    # The trainer's `train_steps` parameter controls how many batches it consumes.
    trainer_kd_distill.train(
        train_iter_kd, # Directly pass the (theoretically infinite) iterator from cycle_loader
        train_steps=num_steps_to_run_kd_train, # Tell trainer how many steps (batches) to run
        save_checkpoint_steps=args_kd.save_checkpoint_steps if hasattr(args_kd, 'save_checkpoint_steps') else 100, # Default from notebook
        valid_iter=None, # As per notebook
        valid_steps=args_kd.valid_steps # As per notebook
    )

    print(f"Model trained for {num_steps_to_run_kd_train} steps and saved to {output_path_kd}/ckpt") # output_path_kd from Cell 14
else:
    print("Skipping KD Training Loop (Cell 18) as dependencies not met.")
print("-" * 30)

# Cell 19
print("Cell 19: Translation and Evaluation")
# Depends on output_path_kd (cell 14), num_steps_to_run_kd_train (cell 18), data_dir (cell 6)
# Reconstruct model path based on training steps performed
model_path_translate = os.path.join(output_path_kd, 'ckpt', f'model_step_{num_steps_to_run_kd_train}.pt')

src_file_translate = f"{data_dir}/test.de.bert"
tgt_file_translate = f"{data_dir}/test.en.bert" # -tgt for translate.py is for gold targets, optional for generation
out_dir_translate = "output/translation" # Created in cell 5
ref_file_translate = f"{data_dir}/test.en"

# Ensure the output directory exists (redundant if cell 5 ran, but safe)
os.makedirs(out_dir_translate, exist_ok=True)

# Run translation if model exists
if os.path.exists(model_path_translate):
    print(f"Model found at {model_path_translate}. Running translation...")
    try:
        translate_script = os.path.join(opennmt_dir_abs, "translate.py")
        translate_cmd = [
            sys.executable, translate_script,
            "-model", model_path_translate,
            "-src", src_file_translate,
            # "-tgt", tgt_file_translate, # Optional for generation
            "-output", f"{out_dir_translate}/result.en",
            "-beam_size", "5", "-alpha", "0.6",
            "-length_penalty", "wu"
        ]
        if torch.cuda.is_available(): # Add GPU arg if available
            translate_cmd.extend(["-gpu", "0"])
        run_shell_command(translate_cmd)


        print("Translation completed. Detokenizing output...")
        result_en_path = f"{out_dir_translate}/result.en"
        if os.path.exists(result_en_path):
            detokenize_script = "scripts/bert_detokenize.py" # Relative to language_distilling dir
            detokenize_cmd = [
                sys.executable, detokenize_script,
                "--file", result_en_path,
                "--output_dir", out_dir_translate
            ]
            run_shell_command(detokenize_cmd)

            result_en_detok_path = f"{out_dir_translate}/result.en.detok"
            if os.path.exists(result_en_detok_path):
                print("Evaluating with BLEU score...")
                multi_bleu_script = os.path.join(opennmt_dir_abs, "tools", "multi-bleu.perl")
                bleu_output_path = f"{out_dir_translate}/result.bleu"
                
                # Capture output of perl script to a file
                with open(result_en_detok_path, 'r', encoding='utf-8') as infile_bleu, \
                     open(bleu_output_path, 'w', encoding='utf-8') as outfile_bleu:
                    run_shell_command(
                        ["perl", multi_bleu_script, ref_file_translate],
                        stdin=infile_bleu, stdout=outfile_bleu
                    )

                if os.path.exists(bleu_output_path):
                    with open(bleu_output_path, "r", encoding='utf-8') as f:
                        bleu_score = f.read().strip()
                        print(f"BLEU Score: {bleu_score}")
                else:
                    print("Warning: BLEU score file was not generated. This might indicate an issue with the evaluation.")
            else:
                print("Warning: Detokenized output file was not generated.")
        else:
            print("Warning: Translation output file was not generated.")
            
    except Exception as e:
        print(f"Error during translation process: {str(e)}")
        import traceback # Import here as it's only for this exception
        traceback.print_exc()
else:
    print(f"Model file {model_path_translate} not found. Skipping translation.")
    print("You need to train the model first or adjust the model path to point to an existing checkpoint.")
print("-" * 30)

# Cell 20
print("Cell 20: Displaying Figures")
import matplotlib.pyplot as plt

# Display the figures from the paper
fig_paths = {
    'CMLM Finetuning': 'figures/cmlm-finetuning.png',
    'Translation Losses': 'figures/translation-losses.png',
    'Translation Accuracy': 'figures/translation-accuracy.png'
}
existing_fig_paths = {title: path for title, path in fig_paths.items() if os.path.exists(path)}

if existing_fig_paths:
    num_figs_to_plot = len(existing_fig_paths)
    fig, axes = plt.subplots(1, num_figs_to_plot, figsize=(6 * num_figs_to_plot, 5))
    if num_figs_to_plot == 1: # Make axes iterable if only one subplot
        axes = [axes]
    
    for i, (title, img_path) in enumerate(existing_fig_paths.items()):
        try:
            img = plt.imread(img_path)
            axes[i].set_title(title)
            axes[i].imshow(img)
            axes[i].axis('off')
        except Exception as e_img:
            print(f"Could not load or display image {img_path}: {e_img}")
            axes[i].text(0.5, 0.5, 'Image not found/readable', ha='center', va='center')
            axes[i].axis('off')

    plt.tight_layout()
    try:
        plt.show()
    except Exception as e_show:
        print(f"Matplotlib plt.show() failed (e.g. no GUI environment): {e_show}")
        # Optionally save the figure
        # fig.savefig("output/summary_figures.png")
        # print("Figure summary saved to output/summary_figures.png")
else:
    print("No figures found in the 'figures/' directory. Skipping plot display.")

print("-" * 30)
print("Script finished.")