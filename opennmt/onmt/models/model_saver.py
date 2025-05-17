import os
import torch
import torch.nn as nn

from collections import deque
from onmt.utils.logging import logger

from copy import deepcopy


def build_model_saver(model_opt, opt, model, fields, optim):
    model_saver = ModelSaver(opt.save_model,
                             model,
                             model_opt,
                             fields,
                             optim,
                             opt.keep_checkpoint)
    return model_saver


class ModelSaverBase(object):
    """Base class for model saving operations

    Inherited classes must implement private methods:
    * `_save`
    * `_rm_checkpoint
    """

    def __init__(self, base_path, model, model_opt, fields, optim,
                 keep_checkpoint=-1):
        self.base_path = base_path
        self.model = model
        self.model_opt = model_opt
        self.fields = fields
        self.optim = optim
        self.last_saved_step = None
        self.keep_checkpoint = keep_checkpoint
        if keep_checkpoint > 0:
            self.checkpoint_queue = deque([], maxlen=keep_checkpoint)

    def save(self, step, moving_average=None):
        """Main entry point for model saver

        It wraps the `_save` method with checks and apply `keep_checkpoint`
        related logic
        """

        if self.keep_checkpoint == 0 or step == self.last_saved_step:
            return

        if moving_average:
            save_model = deepcopy(self.model)
            for avg, param in zip(moving_average, save_model.parameters()):
                param.data.copy_(avg.data)
        else:
            save_model = self.model

        chkpt, chkpt_name = self._save(step, save_model)
        self.last_saved_step = step

        if moving_average:
            del save_model

        if self.keep_checkpoint > 0:
            if len(self.checkpoint_queue) == self.checkpoint_queue.maxlen:
                todel = self.checkpoint_queue.popleft()
                self._rm_checkpoint(todel)
            self.checkpoint_queue.append(chkpt_name)

    def _save(self, step):
        """Save a resumable checkpoint.

        Args:
            step (int): step number

        Returns:
            (object, str):

            * checkpoint: the saved object
            * checkpoint_name: name (or path) of the saved checkpoint
        """

        raise NotImplementedError()

    def _rm_checkpoint(self, name):
        """Remove a checkpoint

        Args:
            name(str): name that indentifies the checkpoint
                (it may be a filepath)
        """

        raise NotImplementedError()


class ModelSaver(ModelSaverBase):
    """Simple model saver to filesystem"""

    def _save(self, step, model):
        real_model = (model.module
                      if isinstance(model, nn.DataParallel)
                      else model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        # Safe copying of model state dict with error handling
        model_state_dict = {}
        try:
            for k, v in real_model.state_dict().items():
                if 'generator' not in k:
                    # Safely move tensor to CPU first to avoid CUDA errors during save
                    if v.is_cuda:
                        try:
                            model_state_dict[k] = v.detach().cpu()
                        except RuntimeError:
                            logger.warning(f"CUDA error when copying parameter {k}, "
                                          "falling back to parameter clone")
                            model_state_dict[k] = v.detach().clone().cpu()
                    else:
                        model_state_dict[k] = v
        except Exception as e:
            logger.error(f"Error copying model state: {str(e)}")
            # Fallback: try using state_dict directly and filter later
            model_state_dict = {k: v for k, v in real_model.state_dict().items()
                              if 'generator' not in k}

        # Safe copying of generator state dict with error handling
        generator_state_dict = {}
        try:
            for k, v in real_generator.state_dict().items():
                # Safely move tensor to CPU first
                if v.is_cuda:
                    try:
                        generator_state_dict[k] = v.detach().cpu()
                    except RuntimeError:
                        logger.warning(f"CUDA error when copying generator parameter {k}, "
                                      "falling back to parameter clone")
                        generator_state_dict[k] = v.detach().clone().cpu()
                else:
                    generator_state_dict[k] = v
        except Exception as e:
            logger.error(f"Error copying generator state: {str(e)}")
            # Fallback: use state_dict directly
            generator_state_dict = real_generator.state_dict()

        # Handle optimizer state safely
        try:
            optim_state_dict = self.optim.state_dict()
        except Exception as e:
            logger.error(f"Error copying optimizer state: {str(e)}")
            optim_state_dict = {}

        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': self.fields,
            'opt': self.model_opt,
            'optim': optim_state_dict,
        }

        logger.info("Saving checkpoint %s_step_%d.pt" % (self.base_path, step))
        checkpoint_path = '%s_step_%d.pt' % (self.base_path, step)
        
        # Use a safer save method with error handling
        try:
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Successfully saved checkpoint to {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error during checkpoint saving: {str(e)}")
            # Try an alternative saving method with pickle_protocol=4
            try:
                logger.info("Attempting alternative save method...")
                torch.save(checkpoint, checkpoint_path, pickle_protocol=4)
                logger.info(f"Successfully saved checkpoint with alternative method")
            except Exception as e2:
                logger.error(f"Alternative save method also failed: {str(e2)}")
                # Last resort: try saving without optimizer state which is often problematic
                try:
                    checkpoint.pop('optim')
                    logger.info("Attempting to save without optimizer state...")
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint without optimizer state")
                except Exception as e3:
                    logger.error(f"All saving methods failed: {str(e3)}")
        
        return checkpoint, checkpoint_path

    def _rm_checkpoint(self, name):
        os.remove(name)
