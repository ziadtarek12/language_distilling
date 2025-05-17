"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import re
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

import onmt.inputters as inputters
import onmt.modules
from onmt.encoders import str2enc

from onmt.decoders import str2dec

from onmt.modules import Embeddings, CopyGenerator
from onmt.modules.util_class import Cast
from onmt.utils.misc import use_gpu
from onmt.utils.logging import logger
from onmt.utils.parse import ArgumentParser


def build_embeddings(opt, text_field, for_encoder=True):
    """
    Args:
        opt: the option in current environment.
        text_field(TextMultiField): word and feats field.
        for_encoder(bool): build Embeddings for encoder or decoder?
    """
    emb_dim = opt.src_word_vec_size if for_encoder else opt.tgt_word_vec_size

    pad_indices = [f.vocab.stoi[f.pad_token] for _, f in text_field]
    word_padding_idx, feat_pad_indices = pad_indices[0], pad_indices[1:]

    num_embs = [len(f.vocab) for _, f in text_field]
    num_word_embeddings, num_feat_embeddings = num_embs[0], num_embs[1:]

    fix_word_vecs = opt.fix_word_vecs_enc if for_encoder \
        else opt.fix_word_vecs_dec

    emb = Embeddings(
        word_vec_size=emb_dim,
        position_encoding=opt.position_encoding,
        feat_merge=opt.feat_merge,
        feat_vec_exponent=opt.feat_vec_exponent,
        feat_vec_size=opt.feat_vec_size,
        dropout=opt.dropout,
        word_padding_idx=word_padding_idx,
        feat_padding_idx=feat_pad_indices,
        word_vocab_size=num_word_embeddings,
        feat_vocab_sizes=num_feat_embeddings,
        sparse=opt.optim == "sparseadam",
        fix_word_vecs=fix_word_vecs
    )
    return emb


def build_encoder(opt, embeddings):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    enc_type = opt.encoder_type if opt.model_type == "text" else opt.model_type
    return str2enc[enc_type].from_opt(opt, embeddings)


def build_decoder(opt, embeddings):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    dec_type = "ifrnn" if opt.decoder_type == "rnn" and opt.input_feed \
               else opt.decoder_type
    return str2dec[dec_type].from_opt(opt, embeddings)


def load_test_model(opt, model_path=None):
    if model_path is None:
        model_path = opt.models[0]
    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)

    model_opt = ArgumentParser.ckpt_model_opts(checkpoint['opt'])
    ArgumentParser.update_model_opts(model_opt)
    ArgumentParser.validate_model_opts(model_opt)
    vocab = checkpoint['vocab']
    if inputters.old_style_vocab(vocab):
        fields = inputters.load_old_vocab(
            vocab, opt.data_type, dynamic_dict=model_opt.copy_attn
        )
    else:
        fields = vocab

    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint,
                             opt.gpu)
    if opt.fp32:
        model.float()
    model.eval()
    model.generator.eval()
    return fields, model, model_opt


def build_base_model(model_opt, fields, gpu, checkpoint=None, gpu_id=None):
    """Build a model from opts.

    Args:
        model_opt: the option loaded from checkpoint. It's important that
            the opts have been updated and validated. See
            :class:`onmt.utils.parse.ArgumentParser`.
        fields (dict[str, torchtext.data.Field]):
            `Field` objects for the model.
        gpu (bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
        gpu_id (int or NoneType): Which GPU to use.

    Returns:
        the NMTModel.
    """
    try:
        # Build embeddings.
        if model_opt.model_type == "text":
            src_field = fields["src"]
            src_emb = build_embeddings(model_opt, src_field)
        else:
            src_emb = None

        # Build encoder.
        encoder = build_encoder(model_opt, src_emb)

        # Build decoder.
        tgt_field = fields["tgt"]
        tgt_emb = build_embeddings(model_opt, tgt_field, for_encoder=False)

        # Share the embedding matrix - preprocess with share_vocab required.
        if model_opt.share_embeddings:
            # Verify vocab compatibility
            if not hasattr(src_field.base_field, 'vocab') or not hasattr(tgt_field.base_field, 'vocab'):
                logger.warning("Cannot share embeddings because field vocab attributes are missing")
            elif src_field.base_field.vocab != tgt_field.base_field.vocab:
                logger.warning("Source and target vocab are different, cannot share embeddings. "
                              "Process data with -share_vocab option for shared embeddings.")
            else:
                # Only share embeddings if vocabs match
                tgt_emb.word_lut.weight = src_emb.word_lut.weight
                logger.info("Embeddings shared successfully")

        decoder = build_decoder(model_opt, tgt_emb)

        # Build NMTModel(= encoder + decoder).
        if gpu and gpu_id is not None:
            device = torch.device("cuda", gpu_id)
        elif gpu and not gpu_id:
            device = torch.device("cuda")
        elif not gpu:
            device = torch.device("cpu")
        
        # Print model settings for debugging
        logger.info(f"Model type: {model_opt.model_type}")
        logger.info(f"Encoder type: {model_opt.encoder_type}")
        logger.info(f"Decoder type: {model_opt.decoder_type}")
        logger.info(f"Using device: {device}")
        
        model = onmt.models.NMTModel(encoder, decoder)

        # Build Generator.
        if not model_opt.copy_attn:
            if model_opt.generator_function == "sparsemax":
                gen_func = onmt.modules.sparse_activations.LogSparsemax(dim=-1)
            else:
                gen_func = nn.LogSoftmax(dim=-1)
                
            # Get vocabulary size and check for validity
            tgt_vocab_size = len(fields["tgt"].base_field.vocab)
            logger.info(f"Target vocabulary size: {tgt_vocab_size}")
            
            # Ensure dimensions match
            if hasattr(model_opt, 'dec_rnn_size'):
                dec_output_dim = model_opt.dec_rnn_size
            else:
                # Default to rnn_size if dec_rnn_size is not specified
                dec_output_dim = model_opt.rnn_size
                logger.warning(f"dec_rnn_size not specified, using rnn_size ({dec_output_dim}) instead")
                
            logger.info(f"Decoder output dimension: {dec_output_dim}")
            
            generator = nn.Sequential(
                nn.Linear(dec_output_dim, tgt_vocab_size),
                Cast(torch.float32),
                gen_func
            )
            
            if model_opt.share_decoder_embeddings:
                if generator[0].weight.shape == decoder.embeddings.word_lut.weight.shape:
                    generator[0].weight = decoder.embeddings.word_lut.weight
                    logger.info("Decoder embeddings shared with generator successfully")
                else:
                    logger.warning(f"Cannot share decoder embeddings with generator: shape mismatch "
                                  f"({generator[0].weight.shape} vs {decoder.embeddings.word_lut.weight.shape})")
        else:
            tgt_base_field = fields["tgt"].base_field
            vocab_size = len(tgt_base_field.vocab)
            pad_idx = tgt_base_field.vocab.stoi[tgt_base_field.pad_token]
            generator = CopyGenerator(model_opt.dec_rnn_size, vocab_size, pad_idx)

        # Load the model states from checkpoint or initialize them.
        if checkpoint is not None:
            # This preserves backward-compat for models using customed layernorm
            def fix_key(s):
                s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',
                          r'\1.layer_norm\2.bias', s)
                s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',
                          r'\1.layer_norm\2.weight', s)
                return s

            checkpoint['model'] = {fix_key(k): v
                                  for k, v in checkpoint['model'].items()}
            # end of patch for backward compatibility

            model.load_state_dict(checkpoint['model'], strict=False)
            generator.load_state_dict(checkpoint['generator'], strict=False)
        else:
            if model_opt.param_init != 0.0:
                for p in model.parameters():
                    p.data.uniform_(-model_opt.param_init, model_opt.param_init)
                for p in generator.parameters():
                    p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            if model_opt.param_init_glorot:
                for p in model.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)
                for p in generator.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

            if hasattr(model.encoder, 'embeddings'):
                model.encoder.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_enc)
            if hasattr(model.decoder, 'embeddings'):
                model.decoder.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_dec)

        model.generator = generator
        
        # Move model to device with error handling
        try:
            logger.info("Moving model to device...")
            # First try to move components separately to identify the problematic part
            for name, component in [('encoder', model.encoder), ('decoder', model.decoder)]:
                try:
                    component.to(device)
                    logger.info(f"{name} successfully moved to {device}")
                except Exception as e:
                    logger.error(f"Error moving {name} to {device}: {str(e)}")
                    raise
            
            # Then try to move generator
            try:
                generator.to(device)
                logger.info(f"Generator successfully moved to {device}")
            except Exception as e:
                logger.error(f"Error moving generator to {device}: {str(e)}")
                raise
                
            # Set model dtype if needed
            if model_opt.model_dtype == 'fp16':
                model.half()
                logger.info("Model converted to half precision (fp16)")
                
            return model
            
        except RuntimeError as e:
            logger.error(f"CUDA error during model creation: {str(e)}")
            logger.error("Falling back to CPU as a workaround")
            
            # Create a new CPU device and try again
            cpu_device = torch.device("cpu")
            model.to(cpu_device)
            # Update the model_opt to prevent further GPU usage attempts
            model_opt.gpu = -1
            return model
            
    except Exception as e:
        logger.error(f"Error in build_base_model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def build_model(model_opt, opt, fields, checkpoint):
    logger.info('Building model...')
    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint)
    logger.info(model)
    return model
