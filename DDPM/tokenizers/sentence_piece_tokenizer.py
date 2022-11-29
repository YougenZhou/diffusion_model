from DDPM.utils import str2bool


class SentencePieceTokenizer(object):

    @classmethod
    def add_cmdline_args(cls, parser):
        group = parser.add_argument_group('Tokenizer')
        group.add_argument('--vocab_path', type=str, required=True)
        group.add_argument('--specials_path', type=str, default='')
        group.add_argument('--do_lower_case', type=str2bool, default=False)
        group.add_argument('--spm_model_file', type=str, required=True)
        return group
