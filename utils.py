import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn

from denoising_diffusion_pytorch.denoising_diffusion_pytorch import GaussianDiffusion, Unet, Trainer

from denoising_diffusion_pytorch.denoising_diffusion_pytorch import default, extract
from einops import rearrange, reduce, repeat
from random import random
import torch.nn.functional as F

class GaussianDiffusionDiff(GaussianDiffusion):

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

    def p_losses_diff(self, x_diff_start, prev_x, t, noise = None, offset_noise_strength = None):
        b, c, h, w = x_diff_start.shape

        noise = default(noise, lambda: torch.randn_like(x_diff_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_diff_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample

        x = self.q_sample(x_start = x_diff_start, t = t, noise = noise)

        # piggyback on x_self_cond from original DDPM implementation
        x_self_cond = prev_x

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_diff_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_diff_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img_diff, prev_img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img_diff.shape, img_diff.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img_diff = self.normalize(img_diff)
        return self.p_losses_diff(img_diff, prev_img, t, *args, **kwargs)
