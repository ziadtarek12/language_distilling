# -*- coding: utf-8 -*-
from functools import partial

import six
import torch
from torchtext.legacy.data import RawField as TorchtextRawField

# Create RawField class instead of importing from torchtext.data
class RawField:
    """
    A Raw Field handles data that will not be processed by the main Field
    infrastructure. This is necessary for fields like the target sequence
    or fields used for sidechannel information.
    """
    def __init__(self):
        self.is_target = False
    
    def preprocess(self, x):
        return x
    
    def process(self, batch, device=None):
        return batch
    
    def __call__(self, mini_batch):
        return mini_batch

# Import our custom Field implementation instead of using torchtext's
from onmt.inputters.inputter import Field

from onmt.inputters.datareader_base import DataReaderBase


class TextDataReader(DataReaderBase):
    def read(self, sequences, side, _dir=None):
        """Read text data from disk.

        Args:
            sequences (str or Iterable[str]):
                path to text file or iterable of the actual text data.
            side (str): Prefix used in return dict. Usually
                ``"src"`` or ``"tgt"``.
            _dir (NoneType): Leave as ``None``. This parameter exists to
                conform with the :func:`DataReaderBase.read()` signature.

        Yields:
            dictionaries whose keys are the names of fields and whose
            values are more or less the result of tokenizing with those
            fields.
        """
        assert _dir is None or _dir == "", \
            "Cannot use _dir with TextDataReader."
        if isinstance(sequences, str):
            sequences = DataReaderBase._read_file(sequences)
        for i, seq in enumerate(sequences):
            if isinstance(seq, six.binary_type):
                seq = seq.decode("utf-8")
            yield {side: seq, "indices": i}


def text_sort_key(ex):
    """Sort using the number of tokens in the sequence."""
    if hasattr(ex, "tgt"):
        return len(ex.src[0]), len(ex.tgt[0])
    return len(ex.src[0])


# mix this with partial
def _feature_tokenize(
        string, layer=0, tok_delim=None, feat_delim=None, truncate=None):
    """Split apart word features (like POS/NER tags) from the tokens.

    Args:
        string (str): A string with ``tok_delim`` joining tokens and
            features joined by ``feat_delim``. For example,
            ``"hello|NOUN|'' Earth|NOUN|PLANET"``.
        layer (int): Which feature to extract. (Not used if there are no
            features, indicated by ``feat_delim is None``). In the
            example above, layer 2 is ``'' PLANET``.
        truncate (int or NoneType): Restrict sequences to this length of
            tokens.

    Returns:
        List[str] of tokens.
    """

    tokens = string.split(tok_delim)
    if truncate is not None:
        tokens = tokens[:truncate]
    if feat_delim is not None:
        tokens = [t.split(feat_delim)[layer] for t in tokens]
    return tokens


class TextMultiField(TorchtextRawField):
    """Container for subfields.

    Text data might use POS/NER/etc labels in addition to tokens.
    This class associates the "base" :class:`Field` with any subfields.
    It also handles padding the data and stacking it.

    Args:
        base_name (str): Name for the base field.
        base_field (Field): The token field.
        feats_fields (Iterable[Tuple[str, Field]]): A list of name-field
            pairs.

    Attributes:
        fields (Iterable[Tuple[str, Field]]): A list of name-field pairs.
            The order is defined as the base field first, then
            ``feats_fields`` in alphabetical order.
    """

    def __init__(self, base_name, base_field, feats_fields):
        super(TextMultiField, self).__init__()
        self.fields = [(base_name, base_field)]
        for name, ff in sorted(feats_fields, key=lambda kv: kv[0]):
            self.fields.append((name, ff))

    @property
    def base_field(self):
        return self.fields[0][1]

    def preprocess(self, x):
        # x is a list of lists: [[base_toks], [feat1_toks], ...]
        return [f.preprocess(x_i) for (name, f), x_i in zip(self.fields, x)]

    def pad(self, examples_data):
        # examples_data: list of what ex.src returns, so:
        # [[ex1_base_preprocessed, ex1_f1_preprocessed,...], [ex2_base_preprocessed, ex2_f1_preprocessed,...], ...]
        # We need to transpose it to:
        # [[ex1_base_preprocessed, ex2_base_preprocessed,...], [ex1_f1_preprocessed, ex2_f1_preprocessed,...], ...]
        # Each inner list is then passed to the corresponding sub-field's pad method.
        
        padded_by_field = []
        # Transpose examples_data
        # Number of fields (base + features)
        num_sub_fields = len(self.fields)
        # For each sub-field, gather all examples' data for that sub-field
        for i in range(num_sub_fields):
            data_for_sub_field = [ex_data_list[i] for ex_data_list in examples_data]
            _name, field_obj = self.fields[i]
            # field_obj is an onmt.inputters.inputter.Field
            # Its pad method expects a list of sequences and returns padded list (or (padded_list, lengths_list))
            # We only care about the padded list here, lengths are handled by base_field in .process()
            padded_output_for_sub_field = field_obj.pad(data_for_sub_field)
            if field_obj.include_lengths: # if pad returns (padded_list, lengths_list)
                padded_by_field.append(padded_output_for_sub_field[0]) # Just take the padded list
            else:
                padded_by_field.append(padded_output_for_sub_field)
        return padded_by_field # list of lists: [[padded_base_toks_for_all_ex], [padded_feat1_toks_for_all_ex], ...]

    def numericalize(self, padded_data_by_field, device=None):
        # padded_data_by_field is output of self.pad:
        # [[all_padded_base_tokens], [all_padded_feat1_tokens], ...]
        numericalized_tensors = []
        for i in range(len(self.fields)):
            _name, field_obj = self.fields[i] # onmt.inputters.inputter.Field
            data_for_sub_field_tensorization = padded_data_by_field[i]
            
            # field_obj.numericalize expects padded list (or (padded_list, lengths) if include_lengths)
            # But our .pad() method for TextMultiField ensures it only passes the padded_list.
            # So, we pass include_lengths=False effectively to sub-field.numericalize here,
            # because lengths handling is centralized in .process() for the base_field.
            # This is a bit of a simplification. Let's stick to calling field_obj.numericalize directly.
            # If field_obj.include_lengths, its numericalize will return (tensor, lengths_tensor).
            # We'll only use the tensor for stacking.
            
            processed_output = field_obj.numericalize(data_for_sub_field_tensorization, device=device)
            if field_obj.include_lengths:
                numericalized_tensors.append(processed_output[0]) # Just the tensor
            else:
                numericalized_tensors.append(processed_output)
        return numericalized_tensors # list of tensors [base_tensor, feat1_tensor, ...]

    def process(self, batch_data, device=None):
        """
        Process a list of examples' data for this TextMultiField.
        batch_data: List of data for this field from each example.
                    Each item is a list [base_item, feat1_item, ...].
        """
        # 1. Preprocess each example's data (applies to base and all features)
        #    Input to TextMultiField.preprocess is one example's data: [base_raw, feat1_raw, ...]
        #    Output is [base_preprocessed, feat1_preprocessed, ...]
        preprocessed_batch_data = [self.preprocess(ex_data) for ex_data in batch_data]

        # 2. Extract data for base field and process it (including lengths if specified)
        base_field_raw_list = [ex_preprocessed_data[0] for ex_preprocessed_data in preprocessed_batch_data]
        base_processed_output = self.base_field.process(base_field_raw_list, device=device)

        lengths = None
        if self.base_field.include_lengths:
            base_tensor_for_stacking = base_processed_output[0]
            lengths = base_processed_output[1]
        else:
            base_tensor_for_stacking = base_processed_output
        
        # 3. Process each feature field
        feature_tensors_for_stacking = []
        for i in range(1, len(self.fields)): # Start from 1 for features
            _feat_name, feat_field = self.fields[i]
            # Extract raw data for this feature from all examples in the batch
            feat_field_raw_list = [ex_preprocessed_data[i] for ex_preprocessed_data in preprocessed_batch_data]
            # Feature fields usually have include_lengths=False.
            # Their .process() will just return the tensor.
            processed_feat_tensor = feat_field.process(feat_field_raw_list, device=device)
            if feat_field.include_lengths: # Should typically be false for features
                 feature_tensors_for_stacking.append(processed_feat_tensor[0])
            else:
                 feature_tensors_for_stacking.append(processed_feat_tensor)
            
        # 4. Stack base tensor and feature tensors
        levels_to_stack = [base_tensor_for_stacking] + feature_tensors_for_stacking
        # Ensure all tensors are 2D (seq_len, batch) before stacking on a new dim.
        # Or if batch_first, (batch, seq_len).
        # The Field.process should ensure this.
        stacked_data = torch.stack(levels_to_stack, 2) # (seq_len, batch, n_fields) or (batch, seq_len, n_fields)

        if lengths is not None: # i.e., self.base_field.include_lengths was true
            return stacked_data, lengths
        else:
            return stacked_data


def text_fields(**kwargs):
    """Create text fields.

    Args:
        base_name (str): Name associated with the field.
        n_feats (int): Number of word level feats (not counting the tokens)
        include_lengths (bool): Optionally return the sequence lengths.
        pad (str, optional): Defaults to ``"<blank>"``.
        bos (str or NoneType, optional): Defaults to ``"<s>"``.
        eos (str or NoneType, optional): Defaults to ``"</s>"``.
        truncate (bool or NoneType, optional): Defaults to ``None``.

    Returns:
        TextMultiField
    """

    n_feats = kwargs["n_feats"]
    include_lengths = kwargs["include_lengths"]
    base_name = kwargs["base_name"]
    pad = kwargs.get("pad", "<blank>")
    bos = kwargs.get("bos", "<s>")
    eos = kwargs.get("eos", "</s>")
    truncate = kwargs.get("truncate", None)
    fields_ = []
    feat_delim = u"￨" if n_feats > 0 else None
    for i in range(n_feats + 1):
        name = base_name + "_feat_" + str(i - 1) if i > 0 else base_name
        tokenize = partial(
            _feature_tokenize,
            layer=i,
            truncate=truncate,
            feat_delim=feat_delim)
        use_len = i == 0 and include_lengths
        feat = Field(
            init_token=bos, eos_token=eos,
            pad_token=pad, tokenize=tokenize,
            include_lengths=use_len)
        fields_.append((name, feat))
    assert fields_[0][0] == base_name  # sanity check
    field = TextMultiField(fields_[0][0], fields_[0][1], fields_[1:])
    return field
