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

def safe_load_vocab(path):
    """
    Safely load an OpenNMT vocabulary file without triggering problematic imports.
    
    Args:
        path (str): Path to the vocabulary file (.pt)
        
    Returns:
        dict: The loaded vocabulary dictionary
    """
    # Create a custom unpickler by using pickle.Unpickler as a function, not trying to subclass it
    try:
        # First try our custom unpickler approach
        with open(path, 'rb') as f:
            unpickler = pickle.Unpickler(f)
            
            # Save the original find_class
            original_find_class = unpickler.find_class
            
            # Define a custom find_class function
            def custom_find_class(module, name):
                if module == 'onmt.inputters.inputter' and name == 'Vocab':
                    return Vocab
                if module == 'torchtext.data.field' and name == 'Field':
                    return Field
                if module == 'torchtext.data.field' and name == 'TextMultiField':
                    return TextMultiField
                
                # For all other module/name combinations, try the normal approach
                try:
                    if module == "__builtin__" and name == "str":
                        return str
                    if module == "__builtin__" and name == "object":
                        return object
                    
                    # Try to import the module and get the attribute
                    __import__(module, level=0)
                    mod = sys.modules.get(module, None)
                    if mod is not None:
                        return getattr(mod, name)
                    
                except (ImportError, AttributeError, KeyError):
                    # If we can't import, create a dummy class
                    pass
                    
                # Create a dummy class for any problematic imports
                dummy_class = type(name, (), {})
                return dummy_class
            
            # Replace the find_class method with our custom one
            unpickler.find_class = custom_find_class
            
            # Load the vocabulary
            vocab = unpickler.load()
            return vocab
            
    except Exception as e:
        print(f"Error with custom unpickler: {e}")
        
        # Fall back to torch.load with a custom function that skips problematic modules
        try:
            # Define a simple loader function to use with torch.load
            def custom_loader(obj_str):
                try:
                    return pickle.loads(obj_str)
                except Exception:
                    # Return an empty dictionary if unpickling fails
                    return {}
                    
            return torch.load(path, map_location='cpu')
        except Exception as e2:
            print(f"Error with torch.load fallback: {e2}")
            
            # Last resort: try a direct approach with a custom pickle mapping
            try:
                data = torch.load(path, map_location=lambda storage, loc: storage)
                return data
            except Exception as e3:
                print(f"All attempts to load vocabulary failed: {e3}")
                raise ValueError(f"Could not load vocabulary from {path}")

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