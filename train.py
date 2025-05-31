import os
import sys
import torch
import numpy as np
import random
import shelve
import io
import argparse # Ensure argparse is imported
import yaml
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import tensorboardX
import subprocess
import torch.nn as nn
import traceback
import matplotlib.pyplot as plt

# Imports from the repository's scripts/modules are deferred until sys.path is set

def run_shell_command(command, **kwargs):
    """Helper function to run shell commands."""
    print(f"Executing: {' '.join(command)}")
    try:
        # Ensure shell=False for list of args, handle shell=True if command is a string
        is_shell_true = isinstance(command, str) and kwargs.get('shell', False)
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
    # --- CLI Argument Parsing ---
    cli_parser = argparse.ArgumentParser(description="Language Distilling Pipeline Script")

    # Stage control
    cli_parser.add_argument('--run_stage1', action=argparse.BooleanOptionalAction, default=True, help="Run Stage 1: CMLM Fine-tuning")
    cli_parser.add_argument('--run_stage2', action=argparse.BooleanOptionalAction, default=True, help="Run Stage 2: Teacher Hidden States & Top-K")
    cli_parser.add_argument('--run_stage3', action=argparse.BooleanOptionalAction, default=True, help="Run Stage 3: Knowledge Distillation Training")
    cli_parser.add_argument('--run_stage4', action=argparse.BooleanOptionalAction, default=True, help="Run Stage 4: Translation and Evaluation")
    cli_parser.add_argument('--run_stage5', action=argparse.BooleanOptionalAction, default=True, help="Run Stage 5: Display Figures")

    # Stage 1 parameters
    cli_parser.add_argument('--num_steps_cmlm', type=int, default=100, help="Number of steps for CMLM fine-tuning (Stage 1)")

    # Stage 2 parameters
    cli_parser.add_argument('--debug_extraction', action=argparse.BooleanOptionalAction, default=True, help="Enable debug mode for hidden state extraction (Stage 2)")
    cli_parser.add_argument('--max_samples_extraction', type=int, default=10, help="Max samples for extraction in debug mode (Stage 2)")
    cli_parser.add_argument('--force_rerun_stage2', action='store_true', help="Force re-computation in Stage 2 even if output files exist")

    # Stage 3 parameters
    cli_parser.add_argument('--num_steps_kd', type=int, default=100, help="Number of steps for Knowledge Distillation training (Stage 3)")
    cli_parser.add_argument('--kd_warmup_steps', type=int, default=800, help="Warmup steps for KD training (Stage 3)")
    cli_parser.add_argument('--kd_valid_steps', type=int, default=1000, help="Validation frequency for KD training (Stage 3)") # Default from notebook
    cli_parser.add_argument('--kd_save_checkpoint_steps', type=int, default=100, help="Checkpoint saving frequency for KD training (Stage 3)")


    cli_args = cli_parser.parse_args()

    # --- Initial Setup and Downloads ---
    print("--- Stage 0: Initial Setup and Downloads ---")
    if not os.path.exists("language_distilling"):
        run_shell_command(["git", "clone", "https://github.com/ziadtarek12/language_distilling"])
    else:
        print("language_distilling repository already cloned.")
    
    # Change directory relative to the script's location if necessary
    # Assuming the script is run from within the cloned repo's parent, or language_distilling is in CWD
    if os.path.basename(os.getcwd()) != "language_distilling":
        if os.path.exists("language_distilling"):
            os.chdir("language_distilling")
            print(f"Changed directory to: {os.getcwd()}")
        else:
            print("Error: 'language_distilling' directory not found. Please clone the repository first.")
            sys.exit(1)
            
    

    print("\n--- Installing Python packages (if needed) ---")
    packages_to_install = [
        "transformers==4.26.0", "pytorch-pretrained-bert", "cytoolz", "tqdm",
        "torchtext==0.16.0", "torchvision==0.16.0", "torch==2.1.0", "torchaudio==2.1.0",
        "configargparse", "tensorboardX", "PyYAML" # Added PyYAML for completeness
    ]
    # Skip ipdb as it's for debugging and not critical for script execution
    # Consider making these pre-requisites rather than installing in-script for production
    for package_spec in packages_to_install:
        package_name = package_spec.split('==')[0]
        try:
            __import__(package_name if package_name != "pytorch-pretrained-bert" else "pytorch_pretrained_bert")
        except ImportError:
            print(f"Installing {package_spec}...")
            run_shell_command([sys.executable, "-m", "pip", "install", package_spec])


    sys.path.append('.')
    sys.path.append('./opennmt')

    from scripts.bert_tokenize import tokenize, process as bert_tokenize_process
    from scripts.bert_prepro import main as bert_prepro_main
    from cmlm.data import BertDataset, TokenBucketSampler as CMLMTokenBucketSampler
    from cmlm.model import convert_embedding, BertForSeq2seq
    from cmlm.util import RunningMeter
    from vocab_loader import safe_load_vocab
    from dump_teacher_hiddens import tensor_dumps, BertSampleDataset, batch_features, process_batch as dump_process_batch
    from dump_teacher_topk import tensor_loads, dump_topk
    from onmt.inputters.bert_kd_dataset import BertKdDataset, TokenBucketSampler as BertKdTokenBucketSampler
    from onmt.utils.optimizers import Optimizer
    from onmt.train_single import build_model_saver, build_trainer, cycle_loader
    from onmt.model_builder import build_model
    import onmt.utils # For ReportMgr

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

    print("\n--- Creating directories ---")
    dirs_to_create = [
        "data/", "output/cmlm_model", "output/bert_dump",
        "output/kd-model/ckpt", "output_path/kd-model/log", "output/translation"
    ]
    for d in dirs_to_create: os.makedirs(d, exist_ok=True)

    print("\n--- Downloading IWSLT German-English dataset ---")
    if not os.path.exists("data/de-en/train.de"):
        run_shell_command(["bash", "scripts/download-iwslt_deen.sh"])
    else:
        print("Dataset files seem to exist, skipping download.")

    bert_model_name = "bert-base-multilingual-cased"
    tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case='uncased' in bert_model_name)
    data_dir = "data/de-en"
    
    # Common paths to define early for inter-stage dependencies
    cmlm_output_dir = "output/cmlm_model"
    num_steps_to_run_cmlm = cli_args.num_steps_cmlm # From CLI
    cmlm_model_save_path = f"{cmlm_output_dir}/model_step_{num_steps_to_run_cmlm}.pt"

    bert_dump_path = "output/bert_dump"
    linear_projection_layer_path = f'{bert_dump_path}/linear.pt'
    hidden_states_db_path = f'{bert_dump_path}/db' # Base path, shelve adds extensions
    topk_db_path = f'{bert_dump_path}/topk' # Base path

    output_path_kd = "output/kd-model"
    num_steps_to_run_kd = cli_args.num_steps_kd # From CLI
    kd_model_checkpoint_path = f"{output_path_kd}/ckpt/model_step_{num_steps_to_run_kd}.pt"


    if cli_args.run_stage1:
        print("\n--- Stage 1: CMLM Fine-tuning ---")
        print("\n--- BERT Tokenization & Preprocessing ---")
        for language in ['de', 'en']:
            for split in ['train', 'valid', 'test']:
                input_file = f"{data_dir}/{split}.{language}"
                output_file = f"{data_dir}/{split}.{language}.bert"
                if not os.path.exists(output_file):
                    print(f"Tokenizing {input_file} to {output_file}...")
                    with open(input_file, 'r', encoding='utf-8') as reader, open(output_file, 'w', encoding='utf-8') as writer:
                        bert_tokenize_process(reader, writer, tokenizer)
                else:
                    print(f"Skipping tokenization for {input_file}, output exists.")
        
        db_output_file = 'data/DEEN.db'
        if not os.path.exists(db_output_file):
            prepro_args = argparse.Namespace(src=f"{data_dir}/train.de.bert", tgt=f"{data_dir}/train.en.bert", output=db_output_file)
            bert_prepro_main(prepro_args)
        else:
            print(f"Skipping BERT prepro, {db_output_file} exists.")

        vocab_file_onmt = "data/DEEN.vocab.pt"
        if not os.path.exists(vocab_file_onmt):
            print("Creating vocabulary files with OpenNMT preprocess.py...")
            opennmt_preprocess_cmd = [
                sys.executable, "opennmt/preprocess.py", "-train_src", f"{data_dir}/train.de.bert",
                "-train_tgt", f"{data_dir}/train.en.bert", "-valid_src", f"{data_dir}/valid.de.bert",
                "-valid_tgt", f"{data_dir}/valid.en.bert", "-save_data", "data/DEEN",
                "-src_seq_length", "150", "-tgt_seq_length", "150"
            ]
            run_shell_command(opennmt_preprocess_cmd)
        else:
            print(f"Skipping OpenNMT vocab creation, {vocab_file_onmt} exists.")

        print("\n--- CMLM Model Setup ---")
        vocab_dump = safe_load_vocab(vocab_file_onmt)
        vocab_stoi = vocab_dump['tgt'].fields[0][1].vocab.stoi
        train_dataset_cmlm = BertDataset('data/DEEN.db', tokenizer, vocab_stoi, seq_len=512, max_len=150)
        train_sampler_cmlm = CMLMTokenBucketSampler(train_dataset_cmlm.lens, 8192, 6144, batch_multiple=1)
        train_loader_cmlm = DataLoader(train_dataset_cmlm, batch_sampler=train_sampler_cmlm, num_workers=min(4, os.cpu_count() or 1), collate_fn=BertDataset.pad_collate)

        cmlm_model = BertForSeq2seq.from_pretrained(bert_model_name)
        bert_embedding = cmlm_model.bert.embeddings.word_embeddings.weight
        hidden_size = cmlm_model.config.hidden_size
        embedding = convert_embedding(tokenizer, vocab_stoi, bert_embedding)
        cmlm_model.cls.predictions.decoder = torch.nn.Linear(hidden_size, embedding.size(0), bias=True)
        cmlm_model.cls.predictions.bias = torch.nn.Parameter(torch.zeros(embedding.size(0)))
        cmlm_model.config.vocab_size = embedding.size(0)
        cmlm_model.cls.predictions.decoder.weight.data.copy_(embedding.data)
        cmlm_model.to(device)

        print("\n--- CMLM Training Loop ---")
        param_optimizer_cmlm = list(cmlm_model.named_parameters())
        no_decay_cmlm = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters_cmlm = [
            {'params': [p for n, p in param_optimizer_cmlm if not any(nd in n for nd in no_decay_cmlm)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer_cmlm if any(nd in n for nd in no_decay_cmlm)], 'weight_decay': 0.0}
        ]
        optimizer_cmlm = AdamW(optimizer_grouped_parameters_cmlm, lr=5e-5)
        scheduler_cmlm = get_linear_schedule_with_warmup(optimizer_cmlm, num_warmup_steps=int(100000 * 0.1), num_training_steps=100000) # Max steps from paper

        running_loss_cmlm = RunningMeter('loss')
        cmlm_model.train()
        print(f"Starting CMLM fine-tuning for {num_steps_to_run_cmlm} steps...")
        cmlm_train_iter = iter(train_loader_cmlm)
        for step in range(num_steps_to_run_cmlm):
            try: batch = next(cmlm_train_iter)
            except StopIteration: cmlm_train_iter = iter(train_loader_cmlm); batch = next(cmlm_train_iter)
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, lm_label_ids = batch
            optimizer_cmlm.zero_grad()
            loss = cmlm_model(input_ids, segment_ids, input_mask, lm_label_ids, output_mask=(lm_label_ids != -1))
            loss.backward(); optimizer_cmlm.step(); scheduler_cmlm.step()
            running_loss_cmlm(loss.item())
            if step % 10 == 0 or step == num_steps_to_run_cmlm - 1: print(f"CMLM Step {step}, Loss: {running_loss_cmlm.val:.4f}")
            if step % 100 == 0 and device.type == 'cuda': torch.cuda.empty_cache()
        
        torch.save(cmlm_model.state_dict(), cmlm_model_save_path)
        print(f"CMLM Model saved to {cmlm_model_save_path}")
        if device.type == 'cuda': torch.cuda.empty_cache()
    else:
        print("Skipping Stage 1: CMLM Fine-tuning.")
        if not os.path.exists(cmlm_model_save_path):
            print(f"Warning: Stage 1 skipped, but CMLM model at {cmlm_model_save_path} not found. Subsequent stages might fail.")

    # --- Stage 2: Teacher Hidden States and Top-K Logits ---
    if cli_args.run_stage2:
        print("\n--- Stage 2: Teacher Hidden States and Top-K Logits ---")
        if not os.path.exists(cmlm_model_save_path):
            print(f"Error: CMLM model {cmlm_model_save_path} not found. Cannot proceed with Stage 2.")
            sys.exit(1)

        print("\n--- Loading fine-tuned CMLM model for Stage 2 ---")
        bert_teacher_model = BertForSeq2seq.from_pretrained(bert_model_name).eval().to(device)
        state_dict = torch.load(cmlm_model_save_path, map_location=device)
        vsize = state_dict['cls.predictions.decoder.weight'].size(0)
        teacher_hidden_size = bert_teacher_model.config.hidden_size
        bert_teacher_model.cls.predictions.decoder = torch.nn.Linear(teacher_hidden_size, vsize, bias=True)
        if 'cls.predictions.bias' in state_dict:
             bert_teacher_model.cls.predictions.bias = torch.nn.Parameter(torch.zeros(vsize, device=device)) # Recreate if necessary
        else: # Original notebook links bias
             bert_teacher_model.cls.predictions.bias = bert_teacher_model.cls.predictions.decoder.bias
        bert_teacher_model.config.vocab_size = vsize
        bert_teacher_model.load_state_dict(state_dict)

        linear_projection_layer = torch.nn.Linear(bert_teacher_model.config.hidden_size, bert_teacher_model.config.vocab_size)
        linear_projection_layer.weight.data = state_dict['cls.predictions.decoder.weight']
        linear_projection_layer.bias.data = state_dict['cls.predictions.bias']
        torch.save(linear_projection_layer, linear_projection_layer_path)
        print(f"Linear projection layer saved to {linear_projection_layer_path}")

        def build_db_batched_local(corpus_path, out_db_shelf, bert_model_param, toker_param, batch_size=8, debug_mode_local=False, max_samples_local=100):
            dataset = BertSampleDataset(corpus_path, toker_param)
            dataset_ids_list = list(dataset.ids) # Work with a list of IDs

            if debug_mode_local and len(dataset_ids_list) > max_samples_local:
                print(f"DEBUG MODE: Limiting extraction to {max_samples_local} samples.")
                # Create a subset view or new dataset for debug
                # For simplicity, we'll filter the IDs and pass a limited list to a modified loader or loop
                # Here we will rely on the loader using this subset of IDs (if possible) or break early.
                # The current BertSampleDataset might not support subsetting ids directly after init.
                # A practical way is to iterate over a subset of keys if using shelve directly.
                # Let's adjust the loader/loop for debug:
                effective_ids = dataset_ids_list[:max_samples_local]
                # Note: DataLoader will still use full dataset unless dataset itself is subsetted.
                # For this example, we'll break loop early in debug mode.
            else:
                effective_ids = dataset_ids_list

            loader = DataLoader(dataset, batch_size=batch_size, num_workers=min(4, os.cpu_count() or 1), collate_fn=batch_features)
            
            processed_count = 0
            with tqdm(desc='Computing BERT features', total=len(effective_ids) if debug_mode_local else len(dataset_ids_list)) as pbar:
                for ids_in_batch, *batch_data in loader:
                    # Filter outputs if in debug mode and full batch is more than remaining samples
                    if debug_mode_local and processed_count + len(ids_in_batch) > max_samples_local:
                        needed = max_samples_local - processed_count
                        # This part is tricky without modifying process_batch or batch_features
                        # Simplest: process full batch, store only needed, then break.
                        ids_to_process_in_batch = ids_in_batch[:needed] if needed < len(ids_in_batch) else ids_in_batch
                    else:
                        ids_to_process_in_batch = ids_in_batch

                    outputs = dump_process_batch(batch_data, bert_model_param, toker_param) # process_batch needs batch_data (src, seg, mask)

                    for id_, output in zip(ids_in_batch, outputs): # Iterate over original batch IDs
                        if debug_mode_local and id_ not in effective_ids[processed_count : processed_count + len(ids_to_process_in_batch)]:
                            continue # If we are subsetting strictly
                        if id_ not in effective_ids : continue # Skip if id not in the target list (less efficient but safe)
                        if output is not None:
                            out_db_shelf[id_] = tensor_dumps(output)
                    
                    pbar.update(len(ids_in_batch))
                    processed_count += len(ids_in_batch)
                    if debug_mode_local and processed_count >= max_samples_local:
                        print(f"DEBUG MODE: Reached max_samples ({max_samples_local}), breaking extraction early.")
                        break
        
        debug_mode_extraction = cli_args.debug_extraction
        max_samples_extraction = cli_args.max_samples_extraction

        skip_extraction = False
        if not cli_args.force_rerun_stage2 and any(os.path.exists(f"{hidden_states_db_path}{ext}") for ext in [".db", ".dat", ".dir", ".bak", ""]):
            print(f"Hidden states DB found at ~{hidden_states_db_path} and --force_rerun_stage2 not set. Skipping extraction.")
            skip_extraction = True
        
        if not skip_extraction:
            print("\n--- Extracting hidden states ---")
            with shelve.open(hidden_states_db_path, 'c') as out_db, torch.no_grad():
                build_db_batched_local('data/DEEN.db', out_db, bert_teacher_model, tokenizer, batch_size=8, 
                                debug_mode_local=debug_mode_extraction, max_samples_local=max_samples_extraction)
            print(f"Hidden states extraction completed. DB at {hidden_states_db_path}")
        
        bert_teacher_model.cpu(); del bert_teacher_model;
        if device.type == 'cuda': torch.cuda.empty_cache()

        skip_topk = False
        if not cli_args.force_rerun_stage2 and any(os.path.exists(f"{topk_db_path}{ext}") for ext in [".db", ".dat", ".dir", ".bak", ""]):
            print(f"Top-K DB found at ~{topk_db_path} and --force_rerun_stage2 not set. Skipping top-k computation.")
            skip_topk = True

        if not skip_topk:
            print("\n--- Computing top-k logits ---")
            if not os.path.exists(linear_projection_layer_path):
                print(f"Error: Linear projection layer {linear_projection_layer_path} not found. Cannot compute top-k.")
                sys.exit(1)
            if not any(os.path.exists(f"{hidden_states_db_path}{ext}") for ext in [".db", ".dat", ".dir", ".bak", ""]):
                print(f"Error: Hidden states DB ~{hidden_states_db_path} not found. Cannot compute top-k.")
                sys.exit(1)

            linear_for_topk = torch.load(linear_projection_layer_path, map_location=device).half().to(device)
            k_topk = 8
            with shelve.open(hidden_states_db_path, 'r') as db_shelf, \
                 shelve.open(topk_db_path, 'c') as topk_db_shelf:
                db_keys = list(db_shelf.keys())
                if debug_mode_extraction and max_samples_extraction < len(db_keys): # Use same debug limiting for topk
                    db_keys = db_keys[:max_samples_extraction]
                    print(f"DEBUG MODE: Computing top-k for {len(db_keys)} items.")
                
                for key in tqdm(db_keys, total=len(db_keys), desc='Computing topk...'):
                    value = db_shelf[key]
                    bert_hidden = torch.tensor(tensor_loads(value)).to(device).half()
                    topk_results = linear_for_topk(bert_hidden).topk(dim=-1, k=k_topk)
                    topk_db_shelf[key] = dump_topk(topk_results)
                    del bert_hidden; 
                    if device.type == 'cuda': torch.cuda.empty_cache()
            linear_for_topk.cpu(); del linear_for_topk;
            if device.type == 'cuda': torch.cuda.empty_cache()
            print(f"Top-k logits computed and saved to {topk_db_path}")
    else:
        print("Skipping Stage 2: Teacher Hidden States & Top-K.")
        if not (os.path.exists(linear_projection_layer_path) and \
                any(os.path.exists(f"{hidden_states_db_path}{ext}") for ext in [".db", ".dat", ".dir", ".bak", ""]) and \
                any(os.path.exists(f"{topk_db_path}{ext}") for ext in [".db", ".dat", ".dir", ".bak", ""])):
            print(f"Warning: Stage 2 skipped, but one or more required files for Stage 3 (linear projection, hidden states DB, top-K DB) not found. Subsequent stages might fail.")


    # --- Stage 3: Knowledge Distillation Training ---
    if cli_args.run_stage3:
        print("\n--- Stage 3: Knowledge Distillation Training ---")
        
        # Ensure Stage 2 outputs exist
        if not (os.path.exists(linear_projection_layer_path) and \
                any(os.path.exists(f"{hidden_states_db_path}{ext}") for ext in [".db", ".dat", ".dir", ".bak", ""]) and \
                any(os.path.exists(f"{topk_db_path}{ext}") for ext in [".db", ".dat", ".dir", ".bak", ""])):
            print(f"Error: One or more required files from Stage 2 (linear projection, hidden states DB, top-K DB) not found. Cannot proceed with Stage 3.")
            sys.exit(1)

        config_path_kd = "opennmt/config/config-transformer-base-mt-deen.yml"
        with open(config_path_kd, 'r') as stream:
            config_kd = yaml.safe_load(stream)
        args_kd = argparse.Namespace(**config_kd)

        # ---FIX for enc_layers/dec_layers---
        # Ensure enc_layers and dec_layers are set, using 'layers' from config or a hardcoded default if necessary.
        # Since the provided YAML has enc_layers and dec_layers, this mainly makes it robust if those keys are missing.
        default_num_layers = 6
        args_kd.enc_layers = config_kd.get('enc_layers', config_kd.get('layers', default_num_layers))
        args_kd.dec_layers = config_kd.get('dec_layers', config_kd.get('layers', default_num_layers))
        # ---End of FIX---

        # Override/set parameters for KD from CLI or defaults
        args_kd.train_from = None
        args_kd.max_grad_norm = 0.0 # OpenNMT handles this via optim typically
        args_kd.kd_topk = 8
        args_kd.train_steps = cli_args.num_steps_kd # From CLI
        args_kd.kd_temperature = 10.0
        args_kd.kd_alpha = 0.5
        args_kd.warmup_steps = cli_args.kd_warmup_steps # From CLI
        args_kd.learning_rate = 2.0 # From notebook
        args_kd.bert_dump = bert_dump_path
        args_kd.data_db = 'data/DEEN.db'
        args_kd.bert_kd = True
        args_kd.data = 'data/DEEN' # For vocab loading

        # Other necessary OpenNMT args (many from notebook, ensure consistency)
        args_kd.model_type = "text"; args_kd.copy_attn = False; args_kd.global_attention = "general"
        args_kd.src_word_vec_size = args_kd.word_vec_size; args_kd.tgt_word_vec_size = args_kd.word_vec_size
        args_kd.feat_merge = "concat"; args_kd.feat_vec_size = -1; args_kd.feat_vec_exponent = 0.7
        args_kd.pre_word_vecs_enc = None; args_kd.pre_word_vecs_dec = None
        args_kd.fix_word_vecs_enc = False; args_kd.fix_word_vecs_dec = False
        args_kd.enc_rnn_size = args_kd.rnn_size; args_kd.dec_rnn_size = args_kd.rnn_size # rnn_size should be in config
        args_kd.transformer_ff = getattr(args_kd, 'transformer_ff', 2048)
        args_kd.heads = getattr(args_kd, 'heads', 8)
        args_kd.max_relative_positions = 0; args_kd.position_encoding = True
        args_kd.param_init = 0.0; args_kd.param_init_glorot = True
        args_kd.share_embeddings = False; args_kd.share_decoder_embeddings = False
        args_kd.truncated_decoder = 0
        args_kd.max_generator_batches = getattr(args_kd, 'max_generator_batches', 32) # From notebook, check config too
        args_kd.normalization = getattr(args_kd, 'normalization', 'sents') # Check config for 'tokens'
        args_kd.accum_count = getattr(args_kd, 'accum_count', [1])
        if not isinstance(args_kd.accum_count, list): args_kd.accum_count = [args_kd.accum_count]
        args_kd.accum_steps = getattr(args_kd, 'accum_steps', [0]) # Check if this needs alignment with accum_count
        args_kd.average_decay = 0.0; args_kd.average_every = 1
        args_kd.valid_steps = cli_args.kd_valid_steps # From CLI
        args_kd.early_stopping = 0; args_kd.early_stopping_criteria = None
        args_kd.valid_batch_size = getattr(args_kd, 'valid_batch_size', 8) # Check config
        args_kd.self_attn_type = "scaled-dot"; args_kd.input_feed = 1 # input_feed for RNNs
        args_kd.copy_attn_type = None; args_kd.generator_function = "softmax"
        args_kd.local_rank = -1; args_kd.gpu_ranks = [0] if torch.cuda.is_available() else []
        args_kd.gpu_verbose_level = 0; args_kd.world_size = 1
        args_kd.encoder_type = getattr(args_kd, 'encoder_type', "transformer")
        args_kd.decoder_type = getattr(args_kd, 'decoder_type', "transformer")
        args_kd.dropout = getattr(args_kd, 'dropout', [0.1])
        if not isinstance(args_kd.dropout, list): args_kd.dropout = [args_kd.dropout] * len(args_kd.accum_count)
        args_kd.attention_dropout = getattr(args_kd, 'attention_dropout', [0.1])
        if not isinstance(args_kd.attention_dropout, list): args_kd.attention_dropout = [args_kd.attention_dropout] * len(args_kd.accum_count)
        args_kd.bridge = ""; args_kd.aux_tune = False
        args_kd.subword_prefix = " "; args_kd.subword_prefix_is_joiner = False
        args_kd.save_model = os.path.join(output_path_kd, 'ckpt', 'model')
        args_kd.log_file = os.path.join(output_path_kd, 'log', 'log.txt')
        args_kd.tensorboard = True
        args_kd.tensorboard_log_dir = os.path.join(output_path_kd, 'log')

        print("\n--- Loading vocabulary and dataset for KD ---")
        vocab_onmt = torch.load(args_kd.data + '.vocab.pt')
        src_vocab_kd = vocab_onmt['src'].fields[0][1].vocab
        tgt_vocab_kd = vocab_onmt['tgt'].fields[0][1].vocab
        train_dataset_kd = BertKdDataset(args_kd.data_db, args_kd.bert_dump, src_vocab_kd.stoi, tgt_vocab_kd.stoi, max_len=150, k=args_kd.kd_topk)
        train_sampler_kd = BertKdTokenBucketSampler(train_dataset_kd.keys, 8192, 6144, batch_multiple=1)
        train_loader_kd = DataLoader(train_dataset_kd, batch_sampler=train_sampler_kd, num_workers=min(4, os.cpu_count() or 1), collate_fn=BertKdDataset.pad_collate)
        iter_state = {'train_iter_kd': cycle_loader(train_loader_kd, device)}

        print("\n--- Building OpenNMT model, optimizer, and trainer for KD ---")
        onmt_fields = {'src': vocab_onmt['src'], 'tgt': vocab_onmt['tgt']}
        model_kd = build_model(args_kd, args_kd, fields=onmt_fields, checkpoint=None).to(device)
        optim_kd = Optimizer.from_opt(model_kd, args_kd, checkpoint=None)
        
        if args_kd.tensorboard:
            from tensorboardX import SummaryWriter
            writer = SummaryWriter(args_kd.tensorboard_log_dir, comment="unmt")
            args_kd.report_manager = onmt.utils.ReportMgr(
                report_every=args_kd.report_every if hasattr(args_kd,'report_every') else 50, # default report_every
                start_time=None, tensorboard_writer=writer # start_time managed by trainer
            )
        else:
            args_kd.report_manager = None

        model_saver_kd = build_model_saver(args_kd, args_kd, model_kd, onmt_fields, optim_kd)
        trainer_kd = build_trainer(args_kd, device_id=0 if device.type == 'cuda' else -1, model=model_kd, fields=onmt_fields, 
                                 optim=optim_kd, model_saver=model_saver_kd, report_manager=args_kd.report_manager)

        print("\n--- Knowledge Distillation Training Loop ---")
        if not hasattr(optim_kd, '_step'): optim_kd._step = 0
        def manual_train_iter_local():
            nonlocal iter_state
            while True:
                try: batch = next(iter_state['train_iter_kd'])
                except StopIteration: iter_state['train_iter_kd'] = cycle_loader(train_loader_kd, device); batch = next(iter_state['train_iter_kd'])
                yield batch

        print(f"Starting KD training for {args_kd.train_steps} steps...")
        trainer_kd.train(manual_train_iter_local(), train_steps=args_kd.train_steps,
                         save_checkpoint_steps=cli_args.kd_save_checkpoint_steps, # From CLI
                         valid_iter=None, valid_steps=args_kd.valid_steps)
        print(f"KD Model trained and saved to {output_path_kd}/ckpt")
    else:
        print("Skipping Stage 3: Knowledge Distillation Training.")
        if not os.path.exists(kd_model_checkpoint_path):
            print(f"Warning: Stage 3 skipped, but KD model at {kd_model_checkpoint_path} not found. Subsequent stages might fail.")


    # --- Stage 4: Translation and Evaluation ---
    if cli_args.run_stage4:
        print("\n--- Stage 4: Translation and Evaluation ---")
        if not os.path.exists(kd_model_checkpoint_path):
            print(f"Error: KD model {kd_model_checkpoint_path} not found. Cannot proceed with Stage 4.")
            sys.exit(1)

        out_dir_translate = "output/translation"
        os.makedirs(out_dir_translate, exist_ok=True)
        
        print(f"Model found at {kd_model_checkpoint_path}. Running translation...")
        try:
            translate_cmd = [
                sys.executable, "opennmt/translate.py", "-model", kd_model_checkpoint_path,
                "-src", f"{data_dir}/test.de.bert", "-output", f"{out_dir_translate}/result.en",
                "-beam_size", "5", "-alpha", "0.6", "-length_penalty", "wu"
            ]
            if torch.cuda.is_available(): translate_cmd.extend(["-gpu", "0"])
            run_shell_command(translate_cmd)

            result_en_file = f"{out_dir_translate}/result.en"
            if os.path.exists(result_en_file):
                print("Translation completed. Detokenizing output...")
                run_shell_command([sys.executable, "scripts/bert_detokenize.py", "--file", result_en_file, "--output_dir", out_dir_translate])
                
                result_en_detok_file = f"{out_dir_translate}/result.en.detok"
                if os.path.exists(result_en_detok_file):
                    print("Evaluating with BLEU score...")
                    bleu_output_file = f"{out_dir_translate}/result.bleu"
                    ref_file_translate = f"{data_dir}/test.en"
                    with open(result_en_detok_file, 'r', encoding='utf-8') as infile, \
                         open(bleu_output_file, 'w', encoding='utf-8') as outfile:
                        run_shell_command(["perl", "opennmt/tools/multi-bleu.perl", ref_file_translate], stdin=infile, stdout=outfile)
                    if os.path.exists(bleu_output_file):
                        with open(bleu_output_file, "r", encoding='utf-8') as f: print(f"BLEU Score: {f.read().strip()}")
                    else: print("Warning: BLEU score file was not generated.")
                else: print("Warning: Detokenized output file was not generated.")
            else: print("Warning: Translation output file was not generated.")
        except Exception as e: print(f"Error during translation process: {str(e)}\n{traceback.format_exc()}")
    else:
        print("Skipping Stage 4: Translation and Evaluation.")

    # --- Stage 5: Display figures ---
    if cli_args.run_stage5:
        print("\n--- Stage 5: Displaying Figures (if available) ---")
        figures_to_display = {'CMLM Finetuning': 'figures/cmlm-finetuning.png', 'Translation Losses': 'figures/translation-losses.png', 'Translation Accuracy': 'figures/translation-accuracy.png'}
        existing_figures = {title: path for title, path in figures_to_display.items() if os.path.exists(path)}
        if existing_figures:
            num_figs = len(existing_figures)
            fig, axes = plt.subplots(1, num_figs, figsize=(6 * num_figs, 5))
            if num_figs == 1: axes = [axes]
            for i, (title, path) in enumerate(existing_figures.items()):
                axes[i].set_title(title)
                try: axes[i].imshow(plt.imread(path)); axes[i].axis('off')
                except Exception as e: print(f"Could not load/display {path}: {e}")
            plt.tight_layout()
            try: plt.show(); print("Displayed figures. Close plot window to continue.")
            except Exception as e: print(f"Could not show plots (e.g., no GUI): {e}")
        else: print("No figures found in 'figures/' directory. Skipping display.")
    else:
        print("Skipping Stage 5: Display Figures.")

    print("\n--- Script execution finished ---")

if __name__ == "__main__":
    main()
