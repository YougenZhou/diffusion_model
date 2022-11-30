from torch import nn

from DDPM.models import register_model
from DDPM.core.model import ModelInterface


@register_model('Transformer')
class Transformer(nn.Module):

    @classmethod
    def add_cmdline_args(cls, parser):
        group = ModelInterface.add_cmdline_args(parser)
        return group

    def __init__(self, args):
        super(Transformer, self).__init__()

    def forward(self, inputs):
        print(inputs)
        exit()
