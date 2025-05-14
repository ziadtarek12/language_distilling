#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
set -x
set -e
export PYTHONUNBUFFERED=1

# Use current directory structure
CURRENT_DIR=$(pwd)
RAW="$CURRENT_DIR/data"
TMP="$CURRENT_DIR/data/tmp"
DUMP="$CURRENT_DIR/data"
DOWNLOAD="$CURRENT_DIR/data/download"

# Download data if not already done
echo "==========================================="
bash scripts/download-iwslt_deen.sh

RAW=$RAW/de-en
TMP=$TMP/de-en
DUMP=$DUMP/de-en

# Create directories if they don't exist
mkdir -p $TMP $DUMP

# BERT tokenization
python scripts/bert_tokenize.py \
    --bert bert-base-multilingual-cased \
    --prefixes $RAW/train.en $RAW/train.de $RAW/valid $RAW/test \
    --output_dir $TMP

# Prepare bert teacher training dataset
mkdir -p $DUMP
python scripts/bert_prepro.py --src $TMP/train.de.bert \
                                   --tgt $TMP/train.en.bert \
                                   --output $DUMP/DEEN.db

# OpenNMT preprocessing
VSIZE=200000
FREQ=0
SHARD_SIZE=200000
python opennmt/preprocess.py \
    -train_src $TMP/train.de.bert \
    -train_tgt $TMP/train.en.bert \
    -valid_src $TMP/valid.de.bert \
    -valid_tgt $TMP/valid.en.bert \
    -save_data $DUMP/DEEN \
    -src_seq_length 150 \
    -tgt_seq_length 150 \
    -src_vocab_size $VSIZE \
    -tgt_vocab_size $VSIZE \
    -vocab_size_multiple 8 \
    -src_words_min_frequency $FREQ \
    -tgt_words_min_frequency $FREQ \
    -share_vocab \
    -shard_size $SHARD_SIZE

# Move needed files to dump
mv $TMP/valid.en.bert $DUMP/dev.en.bert
mv $TMP/valid.de.bert $DUMP/dev.de.bert
mv $TMP/test.en.bert $DUMP/test.en.bert
mv $TMP/test.de.bert $DUMP/test.de.bert
REFDIR=$DUMP/ref/
mkdir -p $REFDIR
cp $RAW/valid.en $REFDIR/dev.en
cp $RAW/test.en $REFDIR/test.en
