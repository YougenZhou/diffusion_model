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
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass
