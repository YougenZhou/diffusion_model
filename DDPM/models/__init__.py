import torch.nn as nn

from DDPM.core.model import ModelInterface
from DDPM.utils import parse_args

MODEL_REGISTER = {}


__all__ = [
    'MODEL_REGISTER',
    'register_model',
    'add_cmdline_args'
]


def register_model(name):
    def __wrapped__(cls):
        if name in MODEL_REGISTER:
            raise ValueError(f'Cannot register duplicate model ({name})')
        if not issubclass(cls, nn.Module):
            raise ValueError(f'Model ({name}) must extend nn.Module')
        MODEL_REGISTER[name] = cls
        return cls

    return __wrapped__


def create_model(args) -> ModelInterface:
    return ModelInterface(args, MODEL_REGISTER[args.model])


def add_cmdline_args(parser):
    group = parser.add_argument_group('Model')
    group.add_argument('--model', type=str, required=True, choices=list(MODEL_REGISTER.keys()))
    group.add_argument('--config_path', type=str, required=True, help='The path of model configuration.')
    args = parse_args(parser, allow_unknown=True)
    if args.model not in MODEL_REGISTER:
        raise ValueError(f'Unknown model type: {args.model}')
    MODEL_REGISTER[args.model].add_cmdline_args(parser)
    return group


import DDPM.models.transformer
