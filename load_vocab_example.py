"""
Example script showing how to load the vocabulary and use it for translation
without triggering the problematic imports from OpenNMT.
"""

import torch
import os
import sys
from vocab_loader import safe_load_vocab
from transformers import BertTokenizer

# Add current directory to path to find modules
sys.path.append('.')

# Import your existing modules
from cmlm.model import BertForSeq2seq, convert_embedding

def load_vocabulary(vocab_file):
    """Load vocabulary without OpenNMT dependencies"""
    print(f"Loading vocabulary from {vocab_file}")
    vocab_dump = safe_load_vocab(vocab_file)
    
    # Access the vocabulary stoi (string to index) dictionary
    # Depending on your vocab structure, you might need to adjust this
    try:
        tgt_vocab = vocab_dump['tgt'].fields[0][1].vocab.stoi
        src_vocab = vocab_dump['src'].fields[0][1].vocab.stoi
        print(f"Loaded vocabulary with {len(src_vocab)} source tokens and {len(tgt_vocab)} target tokens")
        return src_vocab, tgt_vocab
    except (KeyError, AttributeError, IndexError) as e:
        print(f"Error accessing vocabulary structure: {e}")
        # Try alternative method if the structure is different
        if hasattr(vocab_dump, 'stoi'):
            return vocab_dump.stoi, vocab_dump.stoi
        raise

def prepare_model(bert_model, vocab):
    """Prepare the BERT model with the vocabulary"""
    print(f"Loading model from {bert_model}")
    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case='uncased' in bert_model)
    
    # Initialize the model
    model = BertForSeq2seq.from_pretrained(bert_model)
    
    # Update with the target vocabulary
    embedding = convert_embedding(tokenizer, vocab, model.bert.embeddings.word_embeddings.weight)
    model.update_output_layer(embedding)
    
    print("Model prepared successfully")
    return model, tokenizer

if __name__ == "__main__":
    # Example usage:
    vocab_file = "path/to/your/vocab.pt"
    bert_model = "bert-base-uncased"
    
    if len(sys.argv) > 1:
        vocab_file = sys.argv[1]
    if len(sys.argv) > 2:
        bert_model = sys.argv[2]
    
    src_vocab, tgt_vocab = load_vocabulary(vocab_file)
    model, tokenizer = prepare_model(bert_model, tgt_vocab)
    
    print("Vocabulary and model loaded successfully")