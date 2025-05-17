""" Onmt NMT Model base class definition """
import torch
import torch.nn as nn
from onmt.utils.logging import logger


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, bptt=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence of size ``(tgt_len, batch)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        try:
            # Check input tensors for NaN/Inf values to help diagnose issues
            if torch.isnan(src).any():
                logger.warning("NaN values detected in source input")
            if torch.isnan(tgt).any():
                logger.warning("NaN values detected in target input")
                
            # Log shape information for debugging
            logger.debug(f"Source shape: {src.shape}, Target shape: {tgt.shape}")
            if lengths is not None:
                logger.debug(f"Lengths shape: {lengths.shape}")
                
            # Verify all tensors are on the same device
            src_device = src.device
            tgt_device = tgt.device
            if src_device != tgt_device:
                logger.warning(f"Source and target tensors are on different devices: {src_device} vs {tgt_device}")
                # Move tensors to the same device to prevent CUDA errors
                if src_device.type == 'cuda':
                    tgt = tgt.to(src_device)
                    if lengths is not None:
                        lengths = lengths.to(src_device)
                else:
                    src = src.to(tgt_device)
                    if lengths is not None:
                        lengths = lengths.to(tgt_device)
                logger.info(f"Moved tensors to same device: {src.device}")

            # Use target excluding last element (as per original code)
            tgt = tgt[:-1]  # exclude last target from inputs

            # Forward through encoder with error handling
            try:
                enc_state, memory_bank, lengths = self.encoder(src, lengths)
            except RuntimeError as e:
                if "CUDA" in str(e):
                    # Try to recover from CUDA error by falling back to CPU if needed
                    logger.error(f"CUDA error in encoder: {str(e)}")
                    logger.info("Attempting to recover by moving to CPU")
                    src_cpu = src.cpu()
                    lengths_cpu = lengths.cpu() if lengths is not None else None
                    self.encoder.to('cpu')
                    enc_state, memory_bank, lengths = self.encoder(src_cpu, lengths_cpu)
                    # Move results back to original device if possible
                    try:
                        enc_state = self._move_to_device(enc_state, src_device)
                        memory_bank = memory_bank.to(src_device)
                        if lengths is not None:
                            lengths = lengths.to(src_device)
                        self.encoder.to(src_device)
                    except RuntimeError:
                        # If moving back fails, keep on CPU
                        logger.warning("Keeping model on CPU due to CUDA errors")
                        tgt = tgt.cpu()
                        self.decoder.to('cpu')
                else:
                    # Re-raise non-CUDA errors
                    raise

            # Initialize decoder state if not using truncated BPTT
            if bptt is False:
                self.decoder.init_state(src, memory_bank, enc_state)
                
            # Forward through decoder with error handling
            try:
                dec_out, attns = self.decoder(tgt, memory_bank,
                                             memory_lengths=lengths)
            except RuntimeError as e:
                if "CUDA" in str(e):
                    # Try to recover from CUDA error by moving to CPU
                    logger.error(f"CUDA error in decoder: {str(e)}")
                    logger.info("Attempting to recover by moving to CPU")
                    tgt_cpu = tgt.cpu()
                    memory_bank_cpu = memory_bank.cpu()
                    lengths_cpu = lengths.cpu() if lengths is not None else None
                    self.decoder.to('cpu')
                    # Initialize decoder state again on CPU
                    if bptt is False:
                        src_cpu = src.cpu() if src.device.type == 'cuda' else src
                        enc_state_cpu = self._move_to_device(enc_state, torch.device('cpu'))
                        self.decoder.init_state(src_cpu, memory_bank_cpu, enc_state_cpu)
                    dec_out, attns = self.decoder(tgt_cpu, memory_bank_cpu,
                                                memory_lengths=lengths_cpu)
                    # Attempt to move results back to original device
                    try:
                        dec_out = dec_out.to(src_device)
                        attns = {k: v.to(src_device) for k, v in attns.items()}
                        self.decoder.to(src_device)
                    except RuntimeError:
                        logger.warning("Keeping model on CPU due to CUDA errors")
                else:
                    # Re-raise non-CUDA errors
                    raise

            return dec_out, attns
            
        except Exception as e:
            logger.error(f"Error in NMTModel.forward: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
            
    def _move_to_device(self, obj, device):
        """Recursively move an object (tensor, tuple, list, dict) to a device"""
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, tuple):
            return tuple(self._move_to_device(x, device) for x in obj)
        elif isinstance(obj, list):
            return [self._move_to_device(x, device) for x in obj]
        elif isinstance(obj, dict):
            return {k: self._move_to_device(v, device) for k, v in obj.items()}
        else:
            return obj
