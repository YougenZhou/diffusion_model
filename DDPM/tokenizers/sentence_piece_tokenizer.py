import collections
import re
import unicodedata

import sentencepiece as spm

from DDPM.utils import str2bool


def load_vocab(vocab_file):
    vocab = collections.OrderedDict()
    fin = open(vocab_file)
    for num, line in enumerate(fin):
        items = line.rstrip().split('\t')
        if len(items) > 2:
            break
        token = items[0]
        index = items[1] if len(items) == 2 else num
        token = token.strip()
        vocab[token] = int(index)
    return vocab


def clean_text(text):
    text = text.replace(u"“", u'"') \
        .replace(u'”', u'"') \
        .replace(u'‘', "'") \
        .replace(u'’', u"'") \
        .replace(u'—', u'-')

    output = []
    for char in text:
        if _is_control(char):
            continue
        if _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


class SentencePieceTokenizer(object):

    @classmethod
    def add_cmdline_args(cls, parser):
        group = parser.add_argument_group('Tokenizer')
        group.add_argument('--vocab_path', type=str, required=True)
        group.add_argument('--specials_path', type=str, default='')
        group.add_argument('--do_lower_case', type=str2bool, default=False)
        group.add_argument('--spm_model_file', type=str, required=True)
        return group

    def __init__(self, args):
        self.spm_model = spm.SentencePieceProcessor()
        self.spm_model.Load(args.spm_model_file)
        self.vocab = load_vocab(args.vocab_path)
        self.do_lower_case = args.do_lower_case
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        pat_str = ''
        if args.specials_path != '':
            self.specials = load_vocab(args.specials_path)
            for special in self.specials:
                pat_str += '(' + re.escape(special) + ')|'
        else:
            self.specials = {}
        pat_str += '([a-zA-Z0-9\S]+)'
        self.pat = re.compile(pat_str)

    cached = {}

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def bos_id(self):
        return self.vocab['<s>']

    @property
    def eos_id(self):
        return self.vocab['</s>']

    @property
    def pad_id(self):
        return self.vocab['[PAD]']

    @property
    def unk_id(self):
        return self.vocab['<unk>']

    @property
    def mask_id(self):
        return self.vocab['[MASK]']

    def preprocess(self, text):
        outputs = ' '.join(text.strip().split())
        outputs = unicodedata.normalize('NFKD', outputs)
        outputs = ''.join([c for c in outputs if not unicodedata.combining(c)])
        if self.do_lower_case:
            outputs = outputs.lower()
        return outputs

    def tokenize(self, text):
        text = self.preprocess(text)
        if text in self.cached:
            return self.cached[text]
        tokens = []
        for match in self.pat.finditer(text):
            part_text = match.group(0)
            if part_text in self.specials:
                tokens.append(part_text)
                continue
            part_tokens = self._encode_pieces(part_text)
            tokens.extend(part_tokens)
        self.cached[text] = tokens
        return tokens

    def convert_tokens_to_ids(self, tokens):
        ret = []
        for token in tokens:
            if token in self.vocab:
                ret.append(self.vocab[token])
            else:
                ret.append(self.unk_id)
        return ret

    def convert_ids_to_tokens(self, ids):
        ret = []
        for item in ids:
            ret.append(self.inv_vocab[item])
        return ret

    def merge_subword(self, tokens):
        ret = []
        for token in tokens:
            if token.startswith(u"▁"):
                ret.append(token[1:])
            elif token in self.specials:
                ret.append(token)
            else:
                if len(ret):
                    ret[-1] += token
                else:
                    ret.append(token)

        ret = [token for token in ret if token]
        return ret

    def convert_ids_to_str(self, ids):
        tokens = self.convert_ids_to_tokens(ids)
        tokens = self.merge_subword(tokens)
        res = " ".join(tokens).replace("<s>", "")
        res = res.replace("</s>", "\n").replace("\n ", "\n").strip()
        return res

    def _encode_pieces(self, text):
        text = clean_text(text)
        pieces = self.spm_model.EncodeAsPieces(text)
        return pieces
