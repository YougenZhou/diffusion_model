from torch import nn

from DDPM.models import register_model
from DDPM.core.model import ModelInterface


@register_model('Transformer')
class Transformer(nn.Module):

    @classmethod
    def add_cmdline_args(cls, parser):
        group = ModelInterface.add_cmdline_args(parser)
        return group

    pass
