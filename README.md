# Distilling BERT for Text Generation 



## Quick Start

### Option 1: Automated Setup (Recommended)

Run the automated setup script to handle all dependencies, data preparation, and initial setup:

```bash
# Full setup (installs dependencies, downloads data, and prepares everything)
python setup.py

# Skip dependency installation if already installed
python setup.py --skip-deps

# Skip dataset download if you already have the data
python setup.py --skip-download

# Enable debug mode for more verbose logging
python setup.py --debug
```

The setup script will:
1. Install all required dependencies
2. Create necessary output directories
3. Download and prepare the IWSLT14 German-English dataset
4. Apply BERT tokenization to the data
5. Create training database and vocabulary files
6. Verify that everything is set up correctly

### Option 2: Manual Setup

If you prefer to set up manually or need to customize the process:

```bash
# 1. Install dependencies
pip install transformers==4.26.0 pytorch-pretrained-bert cytoolz tqdm shelve-utils datasets matplotlib pyyaml

# 2. Create directories
mkdir -p data/ output/cmlm_model output/bert_dump output/kd-model/ckpt output/kd-model/log output/translation

# 3. Download and prepare dataset
bash scripts/download-iwslt_deen.sh
bash scripts/prepare-iwslt_deen.sh

# 4. Apply BERT tokenization
python scripts/bert_tokenize.py --bert bert-base-multilingual-cased --prefixes data/de-en/train.de data/de-en/train.en data/de-en/valid.de data/de-en/valid.en data/de-en/test.de data/de-en/test.en --output_dir data/de-en/

# 5. Create training database and vocabulary
python scripts/bert_prepro.py --src data/de-en/train.de.bert --tgt data/de-en/train.en.bert --output data/DEEN.db
python opennmt/preprocess.py -train_src data/de-en/train.de.bert -train_tgt data/de-en/train.en.bert -valid_src data/de-en/valid.de.bert -valid_tgt data/de-en/valid.en.bert -save_data data/DEEN -src_seq_length 150 -tgt_seq_length 150
```

## Training Pipeline

After setup, run the three-stage training pipeline:

### Stage 1: CMLM Finetuning

```bash
python run_cmlm_finetuning.py \
    --train_file data/DEEN.db \
    --vocab_file data/DEEN.vocab.pt \
    --valid_src data/de-en/valid.de.bert \
    --valid_tgt data/de-en/valid.en.bert \
    --bert_model bert-base-multilingual-cased \
    --output_dir output/cmlm_model \
    --max_seq_length 512 \
    --max_sent_length 150 \
    --train_batch_size 8384 \
    --learning_rate 5e-5 \
    --num_train_steps 100000 \
    --warmup_proportion 0.1 \
    --valid_steps 1000 \
    --gradient_accumulation_steps 1
```

### Stage 2: Extract Knowledge from Teacher

```bash
python dump_teacher_hiddens.py \
    --bert bert-base-multilingual-cased \
    --ckpt output/cmlm_model/ckpt/model_step_100000.pt \
    --db data/DEEN.db \
    --output output/bert_dump

python dump_teacher_topk.py \
    --bert_dump output/bert_dump \
    --topk 8
```

### Stage 3: Train Student Model with Knowledge Distillation

#### Option A: OpenNMT Transformer (Original Paper Implementation)

```bash
python opennmt/train.py \
    --bert_kd \
    --bert_dump output/bert_dump \
    --data_db data/DEEN.db \
    -data data/DEEN \
    -config opennmt/config/config-transformer-base-mt-deen.yml \
    -learning_rate 2.0 \
    -warmup_steps 8000 \
    --kd_alpha 0.5 \
    --kd_temperature 10.0 \
    --kd_topk 8 \
    --train_steps 100000 \
    -save_model output/kd-model/model
```

#### Option B: Hugging Face T5 Model (Alternative Implementation)

Use the Jupyter notebook (`bert_kd_translation.ipynb`) for the T5 implementation, which provides a modern alternative using Hugging Face Transformers.

### Translation and Evaluation

```bash
# Translate test set
python opennmt/translate.py \
    -model output/kd-model/ckpt/model_step_100000.pt \
    -src data/de-en/test.de.bert \
    -output output/translation/result.en \
    -gpu 0 \
    -beam_size 5 \
    -alpha 0.6 \
    -length_penalty wu \
    -verbose

# Detokenize BERT output
python scripts/bert_detokenize.py \
    --file output/translation/result.en \
    --output_dir output/translation \
    --unk UNK

# Calculate BLEU score
perl opennmt/tools/multi-bleu.perl data/de-en/test.en < output/translation/result.en.detok
```

## Alternative: Jupyter Notebook

For an interactive experience with detailed explanations and visualization, use the Jupyter notebook:

```bash
jupyter notebook bert_kd_translation.ipynb
```

The notebook contains all the code cells that correspond to these scripts and provides additional visualization and debugging capabilities.

## Methodology overview

The proposed approach consists of three stages: conditional-MLM finetuning, extracting the logits from newly trained teacher model, and finally training seq2seq model with knowledge distillation.

1. **Conditional-MLM step** is training pre-trained BERT on target dataset - German-English translation.
2. **Distilling knowledge** from frozen finetuned BERT (teacher model) requires to pre-save the logits. 
   This is done in a second stage when top-K options for each training token are pre-computed and saved on the disc. K is equal to 8 across all the experiments.
3. **Training a seq2seq** encoder-decoder model for translation with the use of distilled knowledge in the loss.

## Notes

- For full training as described in the paper, use 100,000 steps for both CMLM finetuning and student training
- The example commands above use the full step counts for optimal results
- GPU memory requirements are significant; adjust batch sizes if you encounter OOM errors
- The notebook provides two implementation approaches: OpenNMT (original) and Hugging Face T5 (alternative)
- Setup logs are saved to `setup.log` for debugging

## Troubleshooting

If you encounter issues:

1. Check the setup log (`setup.log`) for detailed error messages
2. Ensure you have sufficient disk space (dataset is ~100MB, models can be several GB)
3. Verify CUDA is available for GPU training: `python -c "import torch; print(torch.cuda.is_available())"`
4. For memory issues, reduce batch sizes in the training commands

