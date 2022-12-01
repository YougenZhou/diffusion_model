import os
from collections import namedtuple

import numpy as np
from torch.utils.data import Dataset

from DDPM import tokenizers
from DDPM.utils import str2bool


class EmpatheticDialogues(Dataset):

    @classmethod
    def add_cmdline_args(cls, parser):
        group = parser.add_argument_group('Dataset')
        group.add_argument('--max_src_len', type=int, default=128,
                           help='The maximum length of source sequence (context in dialogue generation task).')
        group.add_argument('--max_tgt_len', type=int, default=128,
                           help='The maximum length of target sequence (response in dialogue generation task).')
        group.add_argument('--max_seq_len', type=int, default=256,
                           help='The maximum length of entire sequence.')
        group.add_argument('--truncate_first_turn', type=str2bool, default=True,
                           help='Whether truncate the first turn utterance.')
        group.add_argument('--random_seed', type=int, default=0,
                           help='The seed to control the data generation.')
        group.add_argument('--batch_size', type=int, default=32)

        tokenizer_group = parser.add_argument_group('Tokenizer')
        tokenizer_group.add_argument('--tokenizer', type=str, default='SentencePieceTokenizer')
        args, _ = parser.parse_known_args()
        tokenizer_cls = getattr(tokenizers, args.tokenizer)
        tokenizer_cls.add_cmdline_args(parser)
        return group

    def __init__(self, args, phase):
        tokenizer_cls = getattr(tokenizers, args.tokenizer)
        self.tokenizer = tokenizer_cls(args)
        self.vocab = self.tokenizer.vocab
        self.pad_id = args.pad_id = self.tokenizer.pad_id
        self.bos_id = args.bos_id = self.tokenizer.bos_id
        self.eos_id = args.eos_id = self.tokenizer.eos_id
        self.unk_id = args.unk_id = self.tokenizer.unk_id
        self.mask_id = args.mask_id = self.tokenizer.mask_id
        self.vocab_size = args.get('vocab_size', self.tokenizer.vocab_size)
        self.max_src_len = args.max_src_len
        self.max_tgt_len = args.max_tgt_len
        self.truncate_first_turn = args.truncate_first_turn

        assert self.max_src_len + self.max_tgt_len <= args.max_seq_len, 'max_src_len + max_tgt_len > max_seq_len'

        self.examples = self._read_file(args.input_file, phase)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        src = self._parse_src(example.src)
        if len(src['token_ids']) == 1:
            raise ValueError(f'Invalid example: context too long or no context - {example.src}')

        tgt = self._parse_tgt(example.tgt)

        return src, tgt

    def _parse_tgt(self, tgt):
        tgt = tgt.strip()
        tgt_tokens = self.tokenizer.tokenize(tgt)
        tgt_token_ids = self.tokenizer.convert_tokens_to_ids(tgt_tokens)
        tgt_token_ids.append(self.eos_id)

        tgt_token_ids = tgt_token_ids[:self.max_tgt_len - 1]
        tgt_token_ids = [self.bos_id] + tgt_token_ids

        field_values = {
            'token_ids': tgt_token_ids,
            'type_ids': [1] * len(tgt_token_ids),
            'pos_ids': list(range(len(tgt_token_ids)))
        }

        return field_values

    def _parse_src(self, src):
        src_token_ids = []
        src_pos_ids = []

        s_token_ids_list = []
        for s in src:
            s = s.strip()
            s_tokens = self.tokenizer.tokenize(s)
            s_token_ids = self.tokenizer.convert_tokens_to_ids(s_tokens) + [self.eos_id]
            s_token_ids_list.append(s_token_ids)

        idx = len(s_token_ids_list) - 1
        total_token_num = 1
        while idx >= 0:
            total_token_num += len(s_token_ids_list[idx])
            if total_token_num > self.max_src_len:
                if self.truncate_first_turn and idx == 0:
                    truncated_ids = s_token_ids_list[idx][:self.max_src_len - total_token_num]
                    if len(truncated_ids) > 1:
                        s_token_ids_list[idx] = truncated_ids[:-1] + [self.eos_id]
                        idx -= 1
                break
            idx -= 1

        for i, s_token_ids in enumerate(s_token_ids_list[idx + 1:], idx + 1):
            src_token_ids += s_token_ids
            src_pos_ids += list(range(1, len(s_token_ids) + 1))

        field_values = {
            'token_ids': [self.bos_id] + src_token_ids,
            'type_ids': [0] * (len(src_token_ids) + 1),
            'pos_ids': [0] + src_pos_ids
        }

        for k in field_values:
            assert len(field_values[k]) == len(field_values['token_ids']), f'the sequence length must be same.'

        return field_values

    def _read_file(self, input_file, phase):
        src_path = os.path.join(input_file, f'sys_dialog_texts.{phase}.npy')
        tgt_path = os.path.join(input_file, f'sys_target_texts.{phase}.npy')
        emo_path = os.path.join(input_file, f'sys_emotion_texts.{phase}.npy')
        src = np.load(src_path, allow_pickle=True)
        tgt = np.load(tgt_path, allow_pickle=True)
        emo = np.load(emo_path, allow_pickle=True)
        assert len(src) == len(tgt) == len(emo)

        headers = ['src', 'tgt', 'emo', 'data_id']
        Example = namedtuple('Example', headers)

        examples = []
        for i in range(len(emo)):
            line = [src[i], tgt[i], emo[i], i]
            example = Example(*line)
            examples.append(example)

        return examples

    def pad_batch_seq(self, examples):
        src_token_ids = [example[0]['token_ids'] for example in examples]
        src_pos_ids = [example[0]['pos_ids'] for example in examples]
        src_type_ids = [example[0]['type_ids'] for example in examples]
        src_token_ids = self._pad_batch_data(src_token_ids, pad_id=self.pad_id)
        src_pos_ids = self._pad_batch_data(src_pos_ids)
        src_type_ids = self._pad_batch_data(src_type_ids)

        tgt_token_ids = [example[1]['token_ids'] for example in examples]
        tgt_pos_ids = [example[1]['pos_ids'] for example in examples]
        tgt_type_ids = [example[1]['type_ids'] for example in examples]
        tgt_token_ids = self._pad_batch_data(tgt_token_ids, pad_id=self.pad_id)
        tgt_pos_ids = self._pad_batch_data(tgt_pos_ids)
        tgt_type_ids = self._pad_batch_data(tgt_type_ids)

        src = {
            'token_ids': src_token_ids, 'pos_ids': src_pos_ids, 'type_ids': src_type_ids
        }
        tgt = {
            'token_ids': tgt_token_ids, 'pos_ids': tgt_pos_ids, 'type_ids': tgt_type_ids
        }

        return src, tgt

    def _pad_batch_data(self, input_lists, pad_id=0):
        max_len = max(map(len, input_lists))
        output_lists = np.array([list(input_list) + [pad_id] * (max_len - len(input_list)) for input_list in input_lists])
        return output_lists.astype('int64').reshape([-1, max_len])
