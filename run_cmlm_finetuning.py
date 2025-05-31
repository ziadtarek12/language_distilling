#!/usr/bin/env python
# coding=utf-8
"""
CMLM finetuning runner - simplified direct execution version.
This script fine-tunes BERT as a conditional masked language model on translation data.
"""
import os
import logging
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from os.path import join, exists
import argparse

# Import transformers components
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup

# Import needed modules
from cmlm.data import BertDataset, TokenBucketSampler
from cmlm.model import convert_embedding, BertForSeq2seq
from cmlm.util import Logger, RunningMeter

# Load vocabulary using our compatibility module
from vocab_loader import safe_load_vocab

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Create logger for TensorBoard
TB_LOGGER = Logger()

def noam_schedule(step, warmup_step=4000):
    """
    Implementation of the Noam learning rate schedule
    """
    if step <= warmup_step:
        return step / warmup_step
    return (warmup_step ** 0.5) * (step ** -0.5)

def main():
    parser = argparse.ArgumentParser(description="CMLM Finetuning")
    
    # Add command-line arguments
    parser.add_argument("--train_file", default="data/DEEN.db", type=str,
                      help="The input train corpus (shelve DB)")
    parser.add_argument("--vocab_file", default="data/DEEN.vocab.pt", type=str,
                      help="Vocabulary file path")
    parser.add_argument("--valid_src", type=str, 
                      help="Source validation file path")
    parser.add_argument("--valid_tgt", type=str,
                      help="Target validation file path")
    parser.add_argument("--bert_model", default="bert-base-multilingual-cased", type=str,
                      help="BERT model to use")
    parser.add_argument("--output_dir", default="output/cmlm_model", type=str,
                      help="Output directory")
    parser.add_argument("--max_seq_length", default=512, type=int,
                      help="Maximum sequence length")
    parser.add_argument("--max_sent_length", default=150, type=int,
                      help="Maximum sentence length")
    parser.add_argument("--train_batch_size", default=6144, type=int,
                      help="Training batch size")
    parser.add_argument("--bucket_size", default=8192, type=int,
                      help="Token bucket size")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                      help="Learning rate")
    parser.add_argument("--num_train_steps", default=5000, type=int,
                      help="Number of training steps")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                      help="Warmup proportion")
    parser.add_argument("--seed", default=42, type=int,
                      help="Random seed")
    parser.add_argument("--num_workers", default=4, type=int,
                      help="Number of data loader workers")
    parser.add_argument("--local_rank", default=-1, type=int,
                      help="Local rank for distributed training")
    parser.add_argument("--data_dir", default="data/de-en", type=str,
                      help="Data directory")
    parser.add_argument("--valid_steps", default=1000, type=int,
                      help="Run validation every X steps")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(join(args.output_dir, 'log'), exist_ok=True)
    os.makedirs(join(args.output_dir, 'ckpt'), exist_ok=True)
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available, using CPU")
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Initialize tensorboard logger
    TB_LOGGER.create(join(args.output_dir, 'log'))
    
    # Load BERT tokenizer
    logger.info(f"Loading tokenizer for {args.bert_model}")
    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case='uncased' in args.bert_model
    )
    
    # Load vocabulary
    logger.info(f"Loading vocabulary from {args.vocab_file}")
    try:
        vocab_dump = safe_load_vocab(args.vocab_file)
        logger.info("Successfully loaded vocabulary with safe_load_vocab")
    except Exception as e:
        logger.warning(f"Safe vocabulary loading failed: {e}. Falling back to torch.load")
        vocab_dump = torch.load(args.vocab_file)
    
    vocab = vocab_dump['tgt'].fields[0][1].vocab.stoi
    
    # Create dataset
    logger.info(f"Creating dataset from {args.train_file}")
    train_dataset = BertDataset(
        args.train_file, tokenizer, vocab, 
        seq_len=args.max_seq_length, 
        max_len=args.max_sent_length
    )
    
    # Define sampler and data loader
    logger.info("Setting up data loader")
    train_sampler = TokenBucketSampler(
        train_dataset.lens, args.bucket_size, 
        args.train_batch_size, batch_multiple=1
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=BertDataset.pad_collate
    )
    
    # Prepare model
    logger.info(f"Loading BERT model: {args.bert_model}")
    model = BertForSeq2seq.from_pretrained(args.bert_model)
    bert_embedding = model.bert.embeddings.word_embeddings.weight
    
    # Print model information before modifications
    hidden_size = model.config.hidden_size
    logger.info(f"Original model: BERT hidden size = {hidden_size}")
    logger.info(f"Original model: BERT vocab size = {bert_embedding.size(0)}")
    logger.info(f"Target vocabulary size = {len(vocab)}")
    
    # Convert vocabulary to embedding form
    logger.info("Converting vocabulary to embedding form")
    embedding = convert_embedding(tokenizer, vocab, bert_embedding)
    
    # Update model architecture to accommodate the new vocabulary size
    logger.info(f"Updating model architecture for vocabulary size: {embedding.size(0)}")
    # Create a new decoder with correct dimensions
    model.cls.predictions.decoder = torch.nn.Linear(hidden_size, embedding.size(0), bias=True)
    model.cls.predictions.bias = torch.nn.Parameter(torch.zeros(embedding.size(0)))
    model.config.vocab_size = embedding.size(0)
    
    # Update the weights
    model.cls.predictions.decoder.weight.data.copy_(embedding.data)
    
    # Move model to device
    model.to(device)
    logger.info(f"Model adapted with vocabulary size: {model.config.vocab_size}")
    
    # Create optimizer
    logger.info("Setting up optimizer")
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                   if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer
                   if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_proportion * args.num_train_steps),
        num_training_steps=args.num_train_steps
    )
    
    # Training loop
    logger.info("Starting training")
    global_step = 0
    running_loss = RunningMeter('loss')
    model.train()
    
    progress_bar = tqdm(total=args.num_train_steps, desc="Training")
    
    while global_step < args.num_train_steps:
        for batch in train_loader:
            if global_step >= args.num_train_steps:
                break
                
            # Move batch to device
            batch = tuple(t.to(device) if t is not None else t for t in batch)
            input_ids, input_mask, segment_ids, lm_label_ids = batch
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            mask = lm_label_ids != -1
            loss = model(input_ids, segment_ids, input_mask,
                         lm_label_ids, mask, True)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            # Track loss
            running_loss(loss.item())
            
            # Increment step counter
            global_step += 1
            
            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix(loss=f"{running_loss.val:.4f}")
            
            # Log to tensorboard
            if global_step % 10 == 0:
                TB_LOGGER.add_scalar('loss', running_loss.val, global_step)
                TB_LOGGER.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                TB_LOGGER.step()
            
            # Save checkpoint
            if global_step % args.valid_steps == 0:
                logger.info(f"Saving model at step {global_step}")
                output_model_file = join(
                    args.output_dir, 'ckpt',
                    f"model_step_{global_step}.pt"
                )
                # Save CPU checkpoint
                state_dict = {k: v.cpu() if isinstance(v, torch.Tensor)
                             else v
                             for k, v in model.state_dict().items()}
                torch.save(state_dict, output_model_file)
                logger.info(f"Model saved to {output_model_file}")
    
    # Save final model
    logger.info(f"Training completed. Saving final model")
    output_model_file = join(
        args.output_dir, 'ckpt',
        f"model_final.pt"
    )
    state_dict = {k: v.cpu() if isinstance(v, torch.Tensor)
                 else v
                 for k, v in model.state_dict().items()}
    torch.save(state_dict, output_model_file)
    logger.info(f"Final model saved to {output_model_file}")
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
