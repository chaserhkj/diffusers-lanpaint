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

        self._abts: tuple[Tensor, Tensor] | None = None
    
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
    
    # Check if model is flow model
    # Currently this is done by checking if alphas_cumprod is provided by scheduler
    @property
    def is_flow(self)-> bool:
        return hasattr(self.scheduler, "alphas_cumprod")

    # Get abt vectors (abt, abt_next) from scheduler. This handles different scheduler architectures
    @property
    def abts(self) -> tuple[Tensor, Tensor]:
        if self._abts is not None:
            return self._abts
        if self.is_flow:
            # If alphas_cumprod is provided, this is a non-flow model, directly subscript alphas_cumprod using timesteps
            abts = self.scheduler.alphas_cumprod[self.scheduler.timesteps]
        else:
            # If no alphas_cumprod, this is flow model, calculate abt from sigma (which is the same as flow_t)
            sigmas: Tensor = self.scheduler.sigmas[:-1]
            abts = (1-sigmas)**2/((1-sigmas)**2 + sigmas**2)
        assert abts is not None
        self._abts = (abts[:-1], abts[1:])
        return self._abts
    
    # Get sigma vectors (sigma, sigma_next) from scheduler:
    @property
    def sigmas(self) -> tuple[Tensor, Tensor]:
        return self.scheduler.sigmas[:-1], self.scheduler.sigmas[1:]
    
    def _forward_diffuse(self, y: Tensor):
        # Forward diffusion of motif to get y at each step:
        # For simplicity, we use the scheduler's sigmas and same formula as in ConditionPipeline:
        # abt = 1/(1+sigma^2), so forward diffuse: y_t = sqrt(abt)*y + sqrt(1 - abt)*noise
        # TODO: add manual seed control for noise used here
        y_forward_diffusion: list[Tensor] = []
        device = y.device
        abts = tuple(a.flip(0) for a in self.abts)
        y_t = y
        for abt, abt_next in zip(*abts):
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
        self.scheduler.set_timesteps()

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
        for t, abt, sigmas, y_t in self.pipeline.progress_bar( zip( self.scheduler.timesteps, self.abts[0], self.sigmas[0], y_forward_diffusion ) ):
            # 1. predict noise model_output

            # Check scale_model_input equivalence to model.model_sampling.noise_scaling in ComfyUI
            x = self.scheduler.scale_model_input(image, t ).to(y.device)
            scale_factor = torch.mean( image * x ) / torch.mean( x ** 2 )

            x = x * (1 - mask) + y_t * mask

            # Translate x to VP-representation depending on model type
            # VP-representation 
            if self.is_flow:
                x_t = x * (abt**0.5 + (1-abt)**0.5)
            else:
                x_t = x / (1 + sigmas ** 2)**0.5
            args = None
            for _ in range(self.n_steps):
                score_func = partial( self._score_model, y = y, mask = mask, abt = abt, t = t )
                # Use constant step size here and apply the (1-abt) factor in sigma_x and sigma_y functions
                x, args = self.langevin_dynamics(x, score_func , mask, self.step_size, abt, sigma_x = self._sigma_x(abt), sigma_y = self._sigma_y(abt), args = args)  
            # Translate x to VP-representation depending on model type
            if self.is_flow:
                x_t = x * (abt**0.5 + (1-abt)**0.5)
            else:
                x_t = x / (1 + sigmas ** 2)**0.5
            model_output = self.denoiser.cal_eps(x, t)
            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, x * scale_factor).prev_sample
    
    def _sigma_x(self, abt: Tensor) -> Tensor:
        return (1 - abt)**self.m
    
    def _sigma_y(self, abt: Tensor) -> Tensor:
        return self.chara_beta * (1 - abt + abt * self.alpha) ** self.m
    
    def _score_model(self, x_t: Tensor, y: Tensor, mask: Tensor, abt: Tensor, t: Tensor):
        pass