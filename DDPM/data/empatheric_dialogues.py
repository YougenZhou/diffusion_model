import os
from collections import namedtuple

import numpy as np
from torch.utils.data import Dataset

from DDPM import tokenizers


class EmpatheticDialogues(Dataset):

    @classmethod
    def add_cmdline_args(cls, parser):
        group = parser.add_argument_group('Dataset')
        group.add_argument('--random_seed', type=int, default=0)

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

        self.examples = self._read_file(args.input_file, phase)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[13]
        print(example)
        src = self._parse_src(example.src)

    def _parse_src(self, src):
        src_token_ids = []

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
