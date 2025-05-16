"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

C-MLM model
"""
import torch
from torch import nn

# Import our compatibility layer instead of directly importing from transformers
from compat_torch import BertForMaskedLM


IN_WORD = '@@'


def convert_embedding(toker, vocab, emb_weight):
    """ seq2seq vs pretrained BERT embedding conversion"""
    # Check if the embedding weight is 1D or 2D
    if len(emb_weight.shape) == 2:
        # Standard case: [vocab_size, embedding_dim]
        vocab_size = emb_weight.size(0)  # Changed from size(1) to size(0)
        embedding_dim = emb_weight.size(1)
    else:
        # Handle unexpected shape
        raise ValueError(f"Unexpected embedding weight shape: {emb_weight.shape}")
    
    if vocab_size % 8 != 0:
        # pad for tensor cores
        vocab_size += (8 - vocab_size % 8)
        
    vectors = [torch.zeros(embedding_dim) for _ in range(len(vocab))]  # Create vectors with embedding_dim
    for word, id_ in vocab.items():
        word = word.replace(IN_WORD, '')
        if word in toker.vocab:
            bert_id = toker.vocab[word]
        else:
            bert_id = toker.vocab['[UNK]']
        vectors[id_] = emb_weight[bert_id].clone()
    embedding = nn.Parameter(torch.stack(vectors, dim=0))
    return embedding


class BertForSeq2seq(BertForMaskedLM):
    """
    The original output projection is shared w/ embedding. Now for seq2seq, we
    use initilization from bert embedding but untied embedding due to
    tokenization difference
    """
    def __init__(self, config, causal=False):
        super().__init__(config)
        # Update initialization method to be compatible with transformers
        self._init_weights(self.cls)

    # Add compatibility method for init_bert_weights which doesn't exist in new transformers
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def init_bert_weights(self, module):
        """ Legacy method for backward compatibility """
        return self._init_weights(module)

    def update_output_layer(self, output_embedding):
        self.cls.predictions.decoder.weight = output_embedding
        vocab_size = output_embedding.size(0)
        self.cls.predictions.bias = nn.Parameter(torch.zeros(vocab_size))
        self.config.vocab_size = vocab_size

    def update_output_layer_by_size(self, vocab_size):
        if vocab_size % 8 != 0:
            # pad for tensor cores
            vocab_size += (8 - vocab_size % 8)
        emb_dim = self.cls.predictions.decoder.weight.size(1)
        self.cls.predictions.decoder.weight = nn.Parameter(
            torch.Tensor(vocab_size, emb_dim))
        self.cls.predictions.bias = nn.Parameter(torch.zeros(vocab_size))
        self.config.vocab_size = vocab_size

    def update_embedding_layer_by_size(self, vocab_size):
        if vocab_size % 8 != 0:
            # pad for tensor cores
            vocab_size += (8 - vocab_size % 8)
        emb_dim = self.cls.predictions.decoder.weight.size(1)
        self.bert.embeddings.word_embeddings = nn.Embedding(
            vocab_size, emb_dim, padding_idx=0)
        self.config.vocab_size = vocab_size

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                masked_lm_labels=None, output_mask=None, do_padding=True):
        """ only computes masked logits to save some computation"""
        if output_mask is None:
            # In transformers library, masked_lm_labels is now labels
            if masked_lm_labels is not None:
                return super().forward(input_ids=input_ids, 
                                    token_type_ids=token_type_ids, 
                                    attention_mask=attention_mask,
                                    labels=masked_lm_labels).loss
            else:
                return super().forward(input_ids=input_ids, 
                                    token_type_ids=token_type_ids, 
                                    attention_mask=attention_mask).logits

        # Get sequence output from the model's encoder
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # only compute masked outputs
        output_mask = output_mask.bool()  # byte() is deprecated, use bool() instead
        sequence_output_masked = sequence_output.masked_select(
            output_mask.unsqueeze(-1).expand_as(sequence_output)
        ).contiguous().view(-1, self.config.hidden_size)
        n_pred, hid = sequence_output_masked.size()
        if do_padding and (n_pred == 0 or n_pred % 8):
            # pad for tensor cores
            n_pad = 8 - n_pred % 8
            pad = torch.zeros(n_pad, hid,
                              dtype=sequence_output_masked.dtype,
                              device=sequence_output_masked.device)
            sequence_output_masked = torch.cat(
                [sequence_output_masked, pad], dim=0)
        else:
            n_pad = 0
        prediction_scores = self.cls.predictions(sequence_output_masked)

        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            lm_labels = masked_lm_labels.masked_select(output_mask)
            if n_pad != 0:
                pad = torch.zeros(n_pad,
                                  dtype=lm_labels.dtype,
                                  device=lm_labels.device).fill_(-1)
                lm_labels = torch.cat([lm_labels, pad], dim=0)
            masked_lm_loss = loss_fct(prediction_scores, lm_labels)
            return masked_lm_loss
        else:
            return prediction_scores
