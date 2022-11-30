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

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass
