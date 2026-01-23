# pyright: reportAny=none, reportExplicitAny=none
# pyright: reportIgnoreCommentWithoutRule=none
import torch
from torch import Tensor
from typing import Any
from diffusers.utils.torch_utils import randn_tensor # pyright: ignore[reportUnknownVariableType]
from functools import partial
from dataclasses import dataclass

# Skip static typecheck and do runtime duck typing on these types
DenoiserWildCard = Any
SchedulerWildCard = Any
PipelineWildCard = Any

@dataclass
class LanPaintPrams(object):
    step_size: float = 0.1
    n_steps: int = 10
    friction: float = 2
    m: float = 1
    alpha: float = 0
    chara_lamb: float = 10
    chara_beta: float = 2

class LanPaintMixIn(object):
    def __init__(self,
                denoiser: DenoiserWildCard,
                scheduler: SchedulerWildCard,
                pipeline: PipelineWildCard,
                params: LanPaintPrams | None = None):
        self.denoiser: DenoiserWildCard = denoiser
        self.scheduler: SchedulerWildCard = scheduler
        self.pipeline: PipelineWildCard = pipeline
        self.device: torch.device = pipeline.device
        self.lan_paint_params = LanPaintPrams() if params is None else params
        self.step_size: float = self.lan_paint_params.step_size
        self.n_steps: int = self.lan_paint_params.n_steps
        self.friction: float = self.lan_paint_params.friction
        self.m: float = self.lan_paint_params.m
        self.alpha: float = self.lan_paint_params.alpha
        self.chara_lamb: float = self.chara_lamb
        self.chara_beta: float = self.chara_beta


        self.latent_image: Tensor | None = None
    
    def _init_image(self, 
                    batch_size: int,
                    generator: torch.Generator | list[torch.Generator] | None):
        # Sample gaussian noise to begin loop
        if isinstance(self.denoiser.config.sample_size, int):
            image_shape = (
                batch_size,
                self.denoiser.config.in_channels,
                self.denoiser.config.sample_size,
                self.denoiser.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.denoiser.config.in_channels, *self.denoiser.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator, dtype=self.denoiser.dtype)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device, dtype=self.denoiser.dtype)
        image = image * self.scheduler.init_noise_sigma
        image = image.to(self.device)
        return image
    
    def _forward_diffuse(self, y: Tensor):
        # Forward diffusion of motif to get y at each step:
        # For simplicity, we use the scheduler's sigmas and same formula as in ConditionPipeline:
        # abt = 1/(1+sigma^2), so forward diffuse: y_t = sqrt(abt)*y + sqrt(1 - abt)*noise
        y_forward_diffusion: list[Tensor] = []
        device = y.device
        abts = ( 1/(1+self.scheduler.sigmas**2) ).flip(0)
        y_t = y
        for abt, abt_next in zip(abts[:-1], abts[1:]):
            noise = torch.randn_like(y)
            y_t = y_t * (abt_next/abt)**0.5 + noise * ((1 - (abt_next/abt))**0.5)
            y_forward_diffusion.append(y_t.to(device))
        y_forward_diffusion.reverse()
        return y_forward_diffusion
    
    @torch.no_grad()
    def lan_paint(
        self,
        latent_image: Tensor,
        mask: Tensor,
        generator: torch.Generator | list[torch.Generator] | None = None,
        num_inference_steps: int = 1000,
    ) -> Tensor:
        batch_size = latent_image.shape[0]
        _ = self.denoiser.to(self.device) 
        self.scheduler.set_timesteps() # pyright: ignore

        y = latent_image
        self.latent_image = latent_image

        image = self._init_image(batch_size, generator)
        y_forward_diffusion = self._forward_diffuse(y)

        def compress_tensor(mask: Tensor):
            if torch.all( mask - mask[0:1] == 0 ):
                mask = mask[0:1] # if mask is the same for all images, use the first one
            if len(mask.shape) > 1:
                if torch.all( mask - mask[:,0:1] == 0 ):
                    mask = mask[:,0:1] # if mask is the same for all images, use the first one
            return mask
        mask = compress_tensor(mask)
        for t, sigma, y_t in self.pipeline.progress_bar( zip( self.scheduler.timesteps, self.scheduler.sigmas[:-1], y_forward_diffusion ) ):
            # 1. predict noise model_output
            abt = 1/( 1+sigma**2 )
            x_t = self.scheduler.scale_model_input(image, t ).to(y.device)
            scale_factor = torch.mean( image * x_t ) / torch.mean( x_t ** 2 )

            x_t = x_t * (1 - mask) + y_t * mask
            step_size = self.step_size * (1 - abt)

            current_times = (sigma, abt)

            args = None
            for _ in range(self.n_steps):
                score_func = partial( self.score_model, y = y, mask = mask, abt = abt, t = t )
                # detect 
                x_t, args = self.langevin_dynamics(x_t, score_func , mask, step_size, current_times, sigma_x = self.sigma_x(abt), sigma_y = self.sigma_y(abt), args = args)  
            model_output = self.denoiser.cal_eps(x_t, t)
            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, x_t * scale_factor).prev_sample