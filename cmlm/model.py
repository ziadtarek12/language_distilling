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
    """ 
    Converts BERT embeddings to embeddings for a custom vocabulary.
    
    This function maps tokens from the target vocabulary to their corresponding
    embeddings in the BERT model vocabulary, creating a new embedding matrix
    suitable for the target task's vocabulary.
    
    Args:
        toker (BertTokenizer): The BERT tokenizer containing the source vocabulary
        vocab (dict): The target vocabulary mapping (word -> id)
        emb_weight (torch.Tensor): The embeddings from BERT model
        
    Returns:
        nn.Parameter: Parameter containing embeddings for the target vocabulary
    """
    # Check if the embedding weight is 1D or 2D
    if len(emb_weight.shape) == 2:
        # Standard case: [vocab_size, embedding_dim]
        bert_vocab_size = emb_weight.size(0)
        embedding_dim = emb_weight.size(1)
    else:
        # Handle unexpected shape
        raise ValueError(f"Unexpected embedding weight shape: {emb_weight.shape}")
    
    # Print dimensions for debugging
    print(f"BERT vocabulary size: {bert_vocab_size}, Embedding dimension: {embedding_dim}")
    print(f"Target vocabulary size: {len(vocab)}")
    
    # Generate embeddings for the target vocabulary
    vectors = [torch.zeros(embedding_dim) for _ in range(len(vocab))]
    
    # Track OOV (out of vocabulary) tokens
    oov_count = 0
    
    # Map tokens from the target vocabulary to BERT vocabulary
    for word, id_ in vocab.items():
        # Remove special token markers if present
        word = word.replace(IN_WORD, '')
        
        # Look up word in BERT vocabulary
        if word in toker.vocab:
            bert_id = toker.vocab[word]
        else:
            bert_id = toker.vocab['[UNK]']  # Use UNK token embedding for OOV
            oov_count += 1
            
        # Copy embedding from BERT
        vectors[id_] = emb_weight[bert_id].clone()
    
    # Report OOV statistics
    if oov_count > 0:
        print(f"OOV tokens: {oov_count}/{len(vocab)} ({oov_count/len(vocab):.2%})")
    
    # Stack the vectors to create the embedding parameter
    embedding = nn.Parameter(torch.stack(vectors, dim=0))
    
    # Verify dimensions
    assert embedding.size(1) == embedding_dim, f"Embedding dimensions don't match: {embedding.size(1)} vs {embedding_dim}"
    
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
        """Update the output layer with the given embeddings.
        
        This method ensures proper dimension handling between the BERT hidden size
        and the target vocabulary size. It creates a new decoder with compatible dimensions
        and initializes it with the provided embeddings.
        
        Args:
            output_embedding (nn.Parameter): Embedding weights for the target vocabulary
            
        Returns:
            nn.Linear: The newly created decoder layer
        """
        # Check and ensure dimensions match hidden size
        hidden_size = self.config.hidden_size
        if output_embedding.size(1) != hidden_size:
            raise ValueError(f"Output embedding dimension {output_embedding.size(1)} doesn't match hidden size {hidden_size}")
        
        # Create a new decoder with correct dimensions
        vocab_size = output_embedding.size(0)
        print(f"Creating decoder with dimensions: input={hidden_size}, output={vocab_size}")
        
        # Create new decoder layer
        self.cls.predictions.decoder = nn.Linear(hidden_size, vocab_size, bias=True)
        
        # Copy the weights properly
        self.cls.predictions.decoder.weight.data.copy_(output_embedding.data)
        
        # Initialize bias
        self.cls.predictions.bias = nn.Parameter(torch.zeros(vocab_size))
        
        # Update config to match new vocabulary size
        self.config.vocab_size = vocab_size
        
        # Return the decoder to enable verification
        return self.cls.predictions.decoder

    def update_output_layer_by_size(self, vocab_size):
        if vocab_size % 8 != 0:
            # pad for tensor cores
            vocab_size += (8 - vocab_size % 8)
            
        # Get embedding dimensions and hidden size
        emb_dim = self.cls.predictions.decoder.weight.size(1)
        hidden_size = self.config.hidden_size
        
        # We need to recreate the entire prediction head to ensure dimension consistency
        # First, ensure transform layer is correct (dense layer that transforms hidden states)
        if hasattr(self.cls.predictions, 'transform'):
            # Make sure transform's output dimension matches hidden_size
            self.cls.predictions.transform.dense = nn.Linear(hidden_size, hidden_size)
            # Initialize weights properly
            self.cls.predictions.transform.dense.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            self.cls.predictions.transform.dense.bias.data.zero_()
        
        # Create new decoder layer with correct dimensions
        self.cls.predictions.decoder = nn.Linear(hidden_size, vocab_size, bias=True)
        self.cls.predictions.decoder.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.cls.predictions.bias = self.cls.predictions.decoder.bias
        self.cls.predictions.bias.data.zero_()
        
        # Update config
        self.config.vocab_size = vocab_size

    def update_embedding_layer_by_size(self, vocab_size):
        if vocab_size % 8 != 0:
            # pad for tensor cores
            vocab_size += (8 - vocab_size % 8)
        
        # Get embedding dimensions from existing model
        emb_dim = self.cls.predictions.decoder.weight.size(1)
        hidden_size = self.config.hidden_size
        
        # Create new embedding with proper initialization
        new_embeddings = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        # Initialize with same distribution as original embeddings
        new_embeddings.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        
        # Transfer embeddings for tokens that exist in both vocabularies (up to min size)
        old_embeddings = self.bert.embeddings.word_embeddings
        min_vocab_size = min(old_embeddings.weight.size(0), vocab_size)
        new_embeddings.weight.data[:min_vocab_size] = old_embeddings.weight.data[:min_vocab_size]
        
        # Update model embeddings
        self.bert.embeddings.word_embeddings = new_embeddings
        
        # We need to recreate the entire prediction head to ensure dimension consistency
        # First, ensure transform layer is correct (dense layer that transforms hidden states)
        if hasattr(self.cls.predictions, 'transform'):
            # Make sure transform's output dimension matches hidden_size
            self.cls.predictions.transform.dense = nn.Linear(hidden_size, hidden_size)
            # Initialize weights properly
            self.cls.predictions.transform.dense.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            self.cls.predictions.transform.dense.bias.data.zero_()
            
        # Update decoder with correct dimensions
        self.cls.predictions.decoder = nn.Linear(hidden_size, vocab_size, bias=True)
        self.cls.predictions.decoder.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.cls.predictions.bias = self.cls.predictions.decoder.bias
        self.cls.predictions.bias.data.zero_()
        
        # Update config
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
        
        # Check if we have any predictions to make
        n_pred, hid = sequence_output_masked.size()
        if n_pred == 0:
            # No tokens to predict, return zero loss or empty prediction
            if masked_lm_labels is not None:
                return torch.tensor(0.0, device=input_ids.device)
            else:
                # Return empty prediction tensor with correct shape
                return torch.zeros((0, self.config.vocab_size), device=input_ids.device)
        
        # Pad if necessary for tensor cores
        if do_padding and n_pred % 8:
            n_pad = 8 - n_pred % 8
            pad = torch.zeros(n_pad, hid,
                              dtype=sequence_output_masked.dtype,
                              device=sequence_output_masked.device)
            sequence_output_masked = torch.cat(
                [sequence_output_masked, pad], dim=0)
        else:
            n_pad = 0
        
        # Verify dimensions before forwarding to predictions
        if hasattr(self.cls.predictions, 'transform'):
            hidden_size = self.cls.predictions.transform.dense.weight.size(1)
            if hidden_size != hid:
                raise ValueError(f"Hidden size mismatch: got {hid}, expected {hidden_size}")
        
        # Check if decoder dimensions match
        decoder_input_size = self.cls.predictions.decoder.weight.size(1)
        if decoder_input_size != self.config.hidden_size:
            raise ValueError(f"Decoder input size {decoder_input_size} doesn't match hidden size {self.config.hidden_size}")
        
        # Forward through prediction layer
        try:
            prediction_scores = self.cls.predictions(sequence_output_masked)
        except RuntimeError as e:
            # Provide detailed error info
            decoder_shape = self.cls.predictions.decoder.weight.shape
            input_shape = sequence_output_masked.shape
            raise RuntimeError(f"Dimension mismatch in decoder. Decoder: {decoder_shape}, Input: {input_shape}. Original error: {str(e)}")

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
