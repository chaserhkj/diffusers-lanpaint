from diffusers import SchedulerMixin

from .protocol import CallableDenoiser 

class LanPaintMixIn(object):
    def __init__(self,
                denoiser: CallableDenoiser,
                scheduler: SchedulerMixin):
        self.denoiser: CallableDenoiser = denoiser
        self.scheduler: SchedulerMixin = scheduler
