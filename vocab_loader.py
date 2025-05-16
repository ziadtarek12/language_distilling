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
import pickle
import codecs
import sys
import types
from collections import Counter, defaultdict

class Vocab:
    """Replacement for OpenNMT's Vocab class to avoid problematic imports."""
    def __init__(self, counter=None, specials=None, max_size=None, min_freq=1):
        self.freqs = counter or Counter()
        self.itos = []
        self.stoi = defaultdict(lambda: 0)
        
        # Add special tokens
        if specials is not None:
            for token in specials:
                if token is not None and token not in self.itos:
                    self.itos.append(token)
        
        # Add tokens from counter
        for token, count in sorted(self.freqs.items(), key=lambda x: (-x[1], x[0])):
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
    def __init__(self, vocab=None, pad_token=None, unk_token=None, init_token=None, eos_token=None):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.init_token = init_token
        self.eos_token = eos_token
        self.vocab = vocab or Vocab()
        self.use_vocab = True
        self.sequential = True

class TextMultiField:
    """Replacement for OpenNMT's TextMultiField class."""
    def __init__(self, base_name, fields):
        self.base_name = base_name
        self.fields = fields

def safe_load_vocab(path):
    """
    Safely load an OpenNMT vocabulary file without triggering problematic imports.
    
    Args:
        path (str): Path to the vocabulary file (.pt)
        
    Returns:
        dict: The loaded vocabulary dictionary
    """
    # Create a custom unpickler that replaces OpenNMT vocab classes with our own
    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
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
    
    # Load the vocabulary using our custom unpickler
    try:
        # First try to load with our custom unpickler
        with open(path, 'rb') as f:
            vocab = CustomUnpickler(f).load()
        return vocab
    except Exception as e:
        print(f"Error with custom unpickler: {e}")
        
        # Fall back to torch.load with a pickle_module argument that handles missing modules
        class CustomPickleModule:
            @staticmethod
            def load(file_obj, **kwargs):
                return CustomUnpickler(file_obj).load()
            
            # Add other required pickle methods
            @staticmethod
            def loads(data, **kwargs):
                file_obj = io.BytesIO(data)
                return CustomUnpickler(file_obj).load()
        
        try:
            return torch.load(path, pickle_module=CustomPickleModule, 
                            map_location='cpu')
        except Exception as e2:
            print(f"Error with torch.load fallback: {e2}")
            
            # Last resort: try to load with default pickle but catch errors
            try:
                return torch.load(path, map_location='cpu')
            except Exception as e3:
                print(f"All loading methods failed: {e3}")
                raise

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        vocab_file = sys.argv[1]
        print(f"Loading vocabulary from {vocab_file}")
        vocab = safe_load_vocab(vocab_file)
        print(f"Vocabulary loaded successfully")
        if isinstance(vocab, dict):
            print(f"Fields: {list(vocab.keys())}")
        else:
            print(f"Vocab type: {type(vocab)}")
    else:
        print("Usage: python vocab_loader.py <path_to_vocab.pt>")