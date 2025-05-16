#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vocabulary loader module that safely loads OpenNMT vocabulary without triggering problematic imports.
This is especially useful when using newer versions of PyTorch that might have compatibility issues
with the OpenNMT codebase.
"""
import torch
import io
import os
import codecs
from collections import Counter, defaultdict

class Vocab:
    """Replacement for OpenNMT's Vocab class to avoid problematic imports."""
    def __init__(self, counter, specials=None, max_size=None, min_freq=1):
        self.freqs = counter
        self.itos = []
        self.stoi = defaultdict(lambda: 0)
        
        # Add special tokens
        if specials is not None:
            for token in specials:
                if token is not None and token not in self.itos:
                    self.itos.append(token)
        
        # Add tokens from counter
        for token, count in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
            if token not in self.itos:
                if max_size is not None and len(self.itos) >= max_size:
                    break
                if min_freq > 1 and count < min_freq:
                    break
                self.itos.append(token)
        
        # Create stoi mapping
        for i, token in enumerate(self.itos):
            self.stoi[token] = i
    
    def __getitem__(self, token):
        return self.stoi[token]
    
    def __len__(self):
        return len(self.itos)

class Field:
    """Replacement for OpenNMT's Field class."""
    def __init__(self, pad_token=None, unk_token=None, init_token=None, eos_token=None):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.init_token = init_token
        self.eos_token = eos_token
        self.vocab = None
        self.use_vocab = True
        self.sequential = True

def _getstate(obj):
    """Custom getstate for pickling Vocab objects."""
    return dict(obj.__dict__, stoi=dict(obj.stoi))

def _setstate(obj, state):
    """Custom setstate for unpickling Vocab objects."""
    obj.__dict__.update(state)
    obj.stoi = defaultdict(lambda: 0, obj.stoi)

def safe_load_vocab(path):
    """
    Safely load an OpenNMT vocabulary file without triggering problematic imports.
    
    Args:
        path (str): Path to the vocabulary file (.pt)
        
    Returns:
        dict: The loaded vocabulary dictionary
    """
    # Custom unpickler that replaces OpenNMT vocab classes with our own
    class VocabUnpickler(torch.load):
        def find_class(self, module, name):
            if module == 'onmt.inputters.inputter' and name == 'Vocab':
                return Vocab
            if module == 'torchtext.data.field' and name == 'Field':
                return Field
            return super().find_class(module, name)
    
    # Load the vocabulary using our custom unpickler
    try:
        with open(path, 'rb') as f:
            vocab = torch.load(f, pickle_module=VocabUnpickler)
        return vocab
    except Exception as e:
        print(f"Error loading vocabulary: {e}")
        # If that fails, try the more brute-force approach
        return torch.load(path, map_location='cpu')

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        vocab_file = sys.argv[1]
        print(f"Loading vocabulary from {vocab_file}")
        vocab = safe_load_vocab(vocab_file)
        print(f"Vocabulary loaded successfully")
        print(f"Fields: {list(vocab.keys())}")
    else:
        print("Usage: python vocab_loader.py <path_to_vocab.pt>")