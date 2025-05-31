#!/usr/bin/env python3
"""
BERT Knowledge Distillation for Translation - Setup Script

This script handles:
1. Installing required dependencies
2. Creating output directories
3. Downloading and preparing the IWSLT14 German-English dataset
4. Applying BERT tokenization
5. Creating training database and vocabulary

Usage:
    python setup.py [--skip-deps] [--skip-download] [--debug]
    
Options:
    --skip-deps: Skip dependency installation
    --skip-download: Skip dataset download (use existing data)
    --debug: Use smaller dataset for debugging
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description, check=True):
    """Run a shell command with error handling"""
    print(f"Running: {description}")
    if args.debug:
        print(f"Command: {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=check, 
            capture_output=True, 
            text=True
        )
        
        if result.stdout and args.debug:
            print(f"Output: {result.stdout}")
        if result.stderr and result.returncode == 0 and args.debug:
            print(f"Warnings: {result.stderr}")
            
        return result
    except subprocess.CalledProcessError as e:
        print(f"Failed: {description}")
        print(f"Error: {e.stderr}")
        raise

def install_dependencies():
    """Install required Python packages"""
    print("Installing dependencies...")
    
    dependencies = [
        "transformers==4.26.0",
        "pytorch-pretrained-bert",
        "cytoolz",
        "tqdm",
        "shelve-utils",
        "datasets",
        "matplotlib",
        "pyyaml",
        "numpy",
        "torchtext==0.16.0",
        "torch==2.1.0",
        "torchvision==0.16.0",
        "torchaudio==2.1.0",
        "ipdb",
        "tensorboardX"
    ]
    
    for dep in dependencies:
        try:
            run_command(f"pip install {dep}", f"Installing {dep}")
        except subprocess.CalledProcessError:
            print(f"Failed to install {dep}, continuing...")
    
    print("Dependencies installation completed")

def create_directories():
    """Create necessary output directories"""
    print("Creating output directories...")
    
    directories = [
        "data/",
        "output/cmlm_model",
        "output/bert_dump", 
        "output/kd-model/ckpt",
        "output/kd-model/log",
        "output/translation",
        "output/kd-model-t5"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def download_dataset():
    """Download and prepare the IWSLT14 dataset"""
    print("Downloading IWSLT14 German-English dataset...")
    
    # Check if scripts exist
    download_script = "scripts/download-iwslt_deen.sh"
    prepare_script = "scripts/prepare-iwslt_deen.sh"
    
    if not os.path.exists(download_script):
        print(f"Download script not found: {download_script}")
        return False
        
    if not os.path.exists(prepare_script):
        print(f"Prepare script not found: {prepare_script}")
        return False
    
    # Make scripts executable
    run_command(f"chmod +x {download_script}", "Making download script executable")
    run_command(f"chmod +x {prepare_script}", "Making prepare script executable")
    
    # Download dataset
    run_command(f"bash {download_script}", "Downloading IWSLT14 dataset")
    
    # Prepare dataset
    run_command(f"bash {prepare_script}", "Preparing IWSLT14 dataset")
    
    # Verify download
    data_dir = "data/de-en"
    required_files = [
        f"{data_dir}/train.de",
        f"{data_dir}/train.en", 
        f"{data_dir}/valid.de",
        f"{data_dir}/valid.en",
        f"{data_dir}/test.de",
        f"{data_dir}/test.en"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"Missing files after download: {missing_files}")
        return False
    
    print("Dataset download and preparation completed")
    return True

def apply_bert_tokenization():
    """Apply BERT tokenization to the dataset"""
    print("Applying BERT tokenization...")
    
    # Import required modules
    try:
        sys.path.append('.')
        from scripts.bert_tokenize import process
        from transformers import BertTokenizer
    except ImportError as e:
        print(f"Failed to import required modules: {e}")
        return False
    
    # Load BERT tokenizer
    bert_model = "bert-base-multilingual-cased"
    try:
        tokenizer = BertTokenizer.from_pretrained(
            bert_model, 
            do_lower_case='uncased' in bert_model
        )
    except Exception as e:
        print(f"Failed to load BERT tokenizer: {e}")
        return False
    
    # Define data directory
    data_dir = "data/de-en"
    
    # Tokenize all files
    for language in ['de', 'en']:
        for split in ['train', 'valid', 'test']:
            input_file = f"{data_dir}/{split}.{language}"
            output_file = f"{data_dir}/{split}.{language}.bert"
            
            if not os.path.exists(input_file):
                print(f"Input file not found: {input_file}")
                continue
                
            print(f"Tokenizing {input_file} -> {output_file}")
            
            try:
                with open(input_file, 'r', encoding='utf-8') as reader, \
                     open(output_file, 'w', encoding='utf-8') as writer:
                    process(reader, writer, tokenizer)
            except Exception as e:
                print(f"Failed to tokenize {input_file}: {e}")
                return False
    
    print("BERT tokenization completed")
    return True

def create_training_data():
    """Create training database and vocabulary"""
    print("Creating training database and vocabulary...")
    
    try:
        sys.path.append('.')
        from scripts.bert_prepro import main as bert_prepro
    except ImportError as e:
        print(f"Failed to import bert_prepro: {e}")
        return False
    
    data_dir = "data/de-en"
    
    # Create dataset DB for BERT training
    prepro_args = argparse.Namespace(
        src=f"{data_dir}/train.de.bert",
        tgt=f"{data_dir}/train.en.bert", 
        output='data/DEEN.db'
    )
    
    try:
        bert_prepro(prepro_args)
        print("Training database created: data/DEEN.db")
    except Exception as e:
        print(f"Failed to create training database: {e}")
        return False
    
    # Create vocabulary using OpenNMT preprocess
    preprocess_cmd = f"""python opennmt/preprocess.py \
        -train_src {data_dir}/train.de.bert \
        -train_tgt {data_dir}/train.en.bert \
        -valid_src {data_dir}/valid.de.bert \
        -valid_tgt {data_dir}/valid.en.bert \
        -save_data data/DEEN \
        -src_seq_length 150 -tgt_seq_length 150"""
    
    try:
        run_command(preprocess_cmd, "Creating vocabulary with OpenNMT preprocess")
        
        # Verify vocabulary file was created
        vocab_file = "data/DEEN.vocab.pt"
        if os.path.exists(vocab_file):
            print(f"Vocabulary file created: {vocab_file}")
        else:
            print(f"Vocabulary file not created: {vocab_file}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"Failed to create vocabulary: {e}")
        return False
    
    return True

def verify_setup():
    """Verify that all required files are in place"""
    print("Verifying setup...")
    
    required_files = [
        "data/DEEN.db",
        "data/DEEN.vocab.pt",
        "data/de-en/train.de.bert",
        "data/de-en/train.en.bert",
        "data/de-en/valid.de.bert", 
        "data/de-en/valid.en.bert",
        "data/de-en/test.de.bert",
        "data/de-en/test.en.bert"
    ]
    
    required_dirs = [
        "output/cmlm_model",
        "output/bert_dump",
        "output/kd-model/ckpt",
        "output/kd-model/log", 
        "output/translation"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    missing_dirs = [d for d in required_dirs if not os.path.exists(d)]
    
    if missing_files:
        print(f"Missing required files: {missing_files}")
        return False
        
    if missing_dirs:
        print(f"Missing required directories: {missing_dirs}")
        return False
    
    print("Setup verification completed successfully")
    return True

def main():
    """Main setup function"""
    global args
    parser = argparse.ArgumentParser(description="Setup BERT Knowledge Distillation environment")
    parser.add_argument("--skip-deps", action="store_true", 
                       help="Skip dependency installation")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip dataset download")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    args = parser.parse_args()
    
    print("Starting BERT Knowledge Distillation setup...")
    
    try:
        # Step 1: Install dependencies
        if not args.skip_deps:
            install_dependencies()
        else:
            print("Skipping dependency installation")
        
        # Step 2: Create directories
        create_directories()
        
        # Step 3: Download and prepare dataset
        if not args.skip_download:
            if not download_dataset():
                print("Dataset download failed")
                return 1
        else:
            print("Skipping dataset download")
        
        # Step 4: Apply BERT tokenization
        if not apply_bert_tokenization():
            print("BERT tokenization failed")
            return 1
        
        # Step 5: Create training data
        if not create_training_data():
            print("Training data creation failed") 
            return 1
        
        # Step 6: Verify setup
        if not verify_setup():
            print("Setup verification failed")
            return 1
        
        print("Setup completed successfully!")
        print("You can now run the training scripts:")
        print("1. CMLM finetuning: python run_cmlm_finetuning.py [args]")
        print("2. Knowledge extraction: python dump_teacher_hiddens.py [args]") 
        print("3. Student training: python opennmt/train.py [args]")
        
        return 0
        
    except KeyboardInterrupt:
        print("Setup interrupted by user")
        return 1
    except Exception as e:
        print(f"Unexpected error during setup: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())