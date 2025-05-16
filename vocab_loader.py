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
import sys
import pickle
import codecs
from collections import Counter, defaultdict
import warnings

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
        self.eos_token = init_token
        self.vocab = None
        self.use_vocab = True
        self.sequential = True

class TextMultiField:
    """Replacement for OpenNMT's TextMultiField class."""
    def __init__(self, base_name, fields):
        self.base_name = base_name
        self.fields = fields

def _getstate(obj):
    """Custom getstate for pickling Vocab objects."""
    return dict(obj.__dict__, stoi=dict(obj.stoi))

def _setstate(obj, state):
    """Custom setstate for unpickling Vocab objects."""
    obj.__dict__.update(state)
    obj.stoi = defaultdict(lambda: 0, obj.stoi)

def safe_load_vocab(vocab_file):
    """
    Safely load vocabulary files that might have been saved with a different PyTorch version.
    
    Args:
        vocab_file: Path to the vocabulary file
        
    Returns:
        Loaded vocabulary dictionary
    """
    if not os.path.exists(vocab_file):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")
    
    # Try multiple loading methods to handle version differences
    try:
        # Method 1: Standard torch.load
        return torch.load(vocab_file)
    except Exception as e1:
        warnings.warn(f"Standard loading failed: {e1}. Trying alternative methods...")
        
        try:
            # Method 2: Using pickle directly
            with open(vocab_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e2:
            warnings.warn(f"Pickle loading failed: {e2}. Trying legacy loading...")
            
            try:
                # Method 3: Using torch.load with map_location and encoding handling
                return torch.load(
                    vocab_file, 
                    map_location=lambda storage, loc: storage,
                    pickle_module=pickle
                )
            except Exception as e3:
                # Last resort: Try to handle binary compatibility issues
                warnings.warn(f"Legacy loading failed: {e3}. Using byte-level compatibility handling...")
                
                try:
                    # Method 4: Byte-level compatibility handling
                    with open(vocab_file, 'rb') as f:
                        buffer = io.BytesIO(f.read())
                        return torch.load(buffer)
                except Exception as e4:
                    raise RuntimeError(f"All loading methods failed. Last error: {e4}")

if __name__ == "__main__":
    # Example usage
    import sys
   
    
    vocab = safe_load_vocab("requirements.txt")
    print(f"Vocabulary loaded successfully")
    print(f"Fields: {list(vocab.keys())}")
