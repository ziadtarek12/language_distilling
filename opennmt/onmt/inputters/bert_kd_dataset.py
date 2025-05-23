"""
CMLM-Seq2Seq Knowledge Distillation dataloaders
"""
import io
import random
import shelve

import torch
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence

from toolz.sandbox.core import unzip
from cytoolz import partition_all


EOS = '</s>'
BOS = '<s>'
PAD = 1


def _feature_sort_key(feat):
    _, _, src_len, tgt, *_ = feat
    key = (src_len, tgt.size(0))
    return key


class TokenBucketSampler(Sampler):
    def __init__(self, keys, bucket_size, batch_size, droplast=True,
                 batch_multiple=8):
        self._keys = keys
        self._max_tok = batch_size
        self._bucket_size = bucket_size
        self._droplast = droplast
        self._mul = batch_multiple

    def _create_ids(self):
        return list(range(len(self._keys)))

    def sort_fn(self, i):
        return self._keys[i]

    def __iter__(self):
        ids = self._create_ids()
        random.shuffle(ids)
        buckets = [sorted(ids[i:i+self._bucket_size],
                          key=self.sort_fn, reverse=True)
                   for i in range(0, len(ids), self._bucket_size)]
        # fill batches until max_token (include padding)
        batches = []
        for bucket in buckets:
            max_len = 0
            batch_indices = []
            for indices in partition_all(self._mul, bucket):
                # make sure batch size is multiple of 8
                ind_max = max(self._keys[i] for i in indices)
                ind_max = max(ind_max)  # max of src/tgt
                max_len = max(max_len, ind_max)
                if max_len * (len(batch_indices) + self._mul) > self._max_tok:
                    if not batch_indices:
                        raise ValueError(
                            "max_tokens too small / max_seq_len too long")
                    batches.append(batch_indices)
                    batch_indices = list(indices)
                    max_len = ind_max
                else:
                    batch_indices.extend(indices)
            if not self._droplast and batch_indices:
                batches.append(batch_indices)
        random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        raise ValueError("NOT supported. "
                         "This has some randomness across epochs")


class DistributedTokenBucketSampler(TokenBucketSampler):
    def __init__(self, num_replicas, rank, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rank = rank
        self._num_replicas = num_replicas

    def _create_ids(self):
        return super()._create_ids()[self._rank:-1:self._num_replicas]


class InputFeatures(object):
    def __init__(self, src, tgt, batch_size, indices,
                 topk_logit, topk_indices):
        self.src = src
        self.tgt = tgt
        self.batch_size = batch_size
        self.indices = indices
        self.bert_topk = (topk_logit, topk_indices)


def load_topk(dump):
    with io.BytesIO(dump) as reader:
        topk = torch.load(reader)
    return topk


class BertKdDataset(Dataset):
    def __init__(self, corpus_path, bert_dump_path,
                 src_vocab, tgt_vocab, max_len=150, k=8):
        try:
            self.db = shelve.open(corpus_path, 'r')
        except Exception as e:
            raise ValueError(f"Error opening corpus database: {e}. Make sure the file exists at {corpus_path}")
        
        # Try to open the topk database in read mode, and if it fails, provide a helpful error message
        try:
            self.topk_db = shelve.open(f'{bert_dump_path}/topk', 'r')
        except Exception as e:
            raise ValueError(f"Error opening topk database: {e}. Make sure you've completed Stage 2 (extracting knowledge) "
                             f"before running Stage 3. The topk database should exist at {bert_dump_path}/topk")
        
        # Ensure the keys in both databases match
        self.ids = []
        self.keys = []
        # Only include examples that are in BOTH the corpus and topk databases
        db_keys = set(self.db.keys())
        topk_keys = set(self.topk_db.keys())
        common_keys = db_keys.intersection(topk_keys)
        
        print(f"Found {len(db_keys)} keys in corpus database")
        print(f"Found {len(topk_keys)} keys in topk database")
        print(f"Found {len(common_keys)} common keys between databases")
        
        if len(common_keys) == 0:
            raise ValueError("No common keys found between corpus and topk databases! "
                             "This suggests the topk database was not created properly. "
                             "Please regenerate the topk database using dump_teacher_topk.py")
        
        for i in common_keys:
            ex = self.db[i]
            src_len = len(ex['src'])
            tgt_len = len(ex['tgt'])
            if (src_len <= max_len
                    and tgt_len <= max_len
                    and src_len + tgt_len + 3 <= 512):
                self.ids.append(i)
                self.keys.append((src_len, tgt_len))
        
        print(f"Using {len(self.ids)} examples after filtering by length constraints")
        
        # vocab for seq2seq
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        assert k <= 128
        self.k = k

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        try:
            example = self.db[id_]
            src_ids = torch.tensor([self.src_vocab[tok] for tok in example['src']])
            src_len = len(src_ids)
            tgt_ids = torch.tensor([self.tgt_vocab[tok]
                                    for tok in [BOS] + example['tgt'] + [EOS]])
            
            try:
                topk_logits, topk_inds = load_topk(self.topk_db[id_])
                topk_logits = topk_logits[:, :self.k].float()
                topk_inds = topk_inds[:, :self.k]
            except KeyError:
                # Handle missing keys in topk database
                print(f"Warning: Key {id_} not found in topk database. Using zeros.")
                # Create dummy tensors with appropriate shapes
                seq_len = len(example['tgt'])
                topk_logits = torch.zeros(seq_len, self.k, dtype=torch.float32)
                topk_inds = torch.zeros(seq_len, self.k, dtype=torch.long)
            
            return i, src_ids, src_len, tgt_ids, topk_logits, topk_inds
        
        except KeyError as e:
            print(f"Error: Key {id_} not found in main database. Error: {e}")
            # Find a valid key as a fallback
            fallback_id = self.ids[0] if i != 0 else self.ids[1]
            print(f"Using fallback key {fallback_id} instead.")
            return self.__getitem__(0 if i != 0 else 1)

    @staticmethod
    def pad_collate(features):
        """ pad the input features to same length"""
        # need to sort by src lens (support RNN encoder)
        features = sorted(features, key=_feature_sort_key, reverse=True)
        (ids, src_ids, src_lens, tgt_ids,
         topk_logits, topk_inds) = map(list, unzip(features))
        src_ids = pad_sequence(src_ids, batch_first=False, padding_value=PAD
                               ).unsqueeze(2)
        src_len = torch.tensor(src_lens)
        tgt_ids = pad_sequence(tgt_ids, batch_first=False, padding_value=PAD
                               ).unsqueeze(2)
        ids = torch.tensor(ids)
        # pad bert hiddens
        len_, batch, _ = tgt_ids.size()
        k = topk_logits[0].size(-1)
        topk_logit = torch.zeros(len_-1,  # minus BOS
                                 batch, k, dtype=topk_logits[0].dtype)
        topk_index = torch.zeros(len_-1,  # minus BOS
                                 batch, k, dtype=topk_inds[0].dtype)
        for i, (logit, index) in enumerate(zip(topk_logits, topk_inds)):
            topk_logit.data[:logit.size(0), i, :] = logit.data
            topk_index.data[:index.size(0), i, :] = index.data

        batch = InputFeatures(src=(src_ids, src_len), tgt=tgt_ids,
                              indices=ids, batch_size=len(ids),
                              topk_logit=topk_logit,
                              topk_indices=topk_index)
        return batch
