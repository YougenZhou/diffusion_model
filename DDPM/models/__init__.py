import torch.nn as nn

MODEL_REGISTER = {}


__all__ = [
    'MODEL_REGISTER'
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
