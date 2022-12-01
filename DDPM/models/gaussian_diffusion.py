import torch
from torch import nn
from torch.nn import functional as F

from DDPM.models import register_model
from DDPM.models.transformer import Transformer


def extract(v, t, x_shape):
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


@register_model('GaussianDiffusion')
class GaussianDiffusion(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super(GaussianDiffusion, self).__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double()
        )
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar)
        )
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar)
        )

    def forward(self, x_0):
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.rand_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        )
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss
