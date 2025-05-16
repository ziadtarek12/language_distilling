"""
Compatibility module for working with newer versions of PyTorch and torchvision.
This module provides a safe way to import functions from run_cmlm_finetuning
without triggering problematic imports.
"""

# Import necessary functions directly 
from run_cmlm_finetuning import noam_schedule, warmup_linear

# Import other commonly used functions that might be needed
try:
    from run_cmlm_finetuning import all_reduce_and_rescale_tensors, all_gather_list
except ImportError:
    # Provide fallback implementations if needed
    def all_reduce_and_rescale_tensors(*args, **kwargs):
        raise NotImplementedError("Distributed functions not available")
    
    def all_gather_list(*args, **kwargs):
        raise NotImplementedError("Distributed functions not available")

# Add any other functions from the original modules that might be needed