from __future__ import annotations

from typing import List, Optional, Union
import torch
import numpy as np

from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_output import BaseOutput
from ..modules.extra_channels import make_extra_channels_tensor
from ..modules.utils import patch_groupnorm, patch_unet, swap_transformer_blocks
from .postprocessing import postprocess_outputs
from dataclasses import dataclass


@dataclass
class CubeDiffPipelineOutput(BaseOutput):
    faces: np.ndarray
    faces_cropped: np.ndarray
    equirectangular: np.ndarray


class CubeDiffPipeline(StableDiffusionPipeline):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load CubeDiffPipeline from pretrained model and automatically apply CubeDiff patches.
        
        The pretrained model should already have the correct input conv layer (7 channels),
        but we still need to patch the attention mechanisms and group norms.
        """

        # Load the base pipeline
        pipeline = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        if pipeline.unet.config.in_channels != 7:
            # Is a base SD model, patch input conv as well
            patch_unet(pipeline.unet, in_channels=7)
        else:
            # Apply attention patches (swap BasicTransformerBlock -> CubeDiffTransformerBlock)
            swap_transformer_blocks(pipeline.unet)
        
        # Apply groupnorm patches (GroupNorm -> CubeDiffGroupNorm)
        patch_groupnorm(pipeline.vae)
    
        return pipeline

    @torch.no_grad()
    def __call__(
        self,
        prompts: Union[str, List[str]],
        *,
        conditioning_image: torch.Tensor,  # (C,H,W)
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        cfg_scale: float = 3.5,
    ):
        device = self._execution_device
        T = 6  # faces

        if isinstance(prompts, str):
            prompts = [prompts] * T
        
        if len(prompts) != T:
            raise ValueError(f"Expected 6 prompts, got {len(prompts)}")


        text_inputs = self.tokenizer(
            prompts,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            return_tensors="pt",
        )
        encoder_hidden_states = self.text_encoder(text_inputs.input_ids.to(device))[0]

        uncond_inputs = self.tokenizer(
            [""] * T,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_inputs.input_ids.to(device))[0]

        # --- scheduler / latents -------------------------------------------
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        latents = torch.randn(
            (T, 4, self.unet.config.sample_size, self.unet.config.sample_size),
            generator=generator,
            device=device,
            dtype=self.unet.dtype,
        )
        latents *= self.scheduler.init_noise_sigma

        static_extra = make_extra_channels_tensor(1, self.unet.config.sample_size, self.unet.config.sample_size).to(device, dtype=self.unet.dtype)

        if conditioning_image.ndim == 3:
            conditioning_image = conditioning_image.unsqueeze(0)
        conditioning_image = conditioning_image.to(device, dtype=self.unet.dtype)
        ref_lat = self.vae.encode(conditioning_image).latent_dist.mean[0]
        ref_lat *= self.vae.config.scaling_factor

        for t in self.scheduler.timesteps:
            latents[0] = ref_lat  # keep front face fixed
            latents_scaled = self.scheduler.scale_model_input(latents, t)
            latents_input = torch.cat([latents_scaled, static_extra], dim=1)

            noise_pred = self.unet(latents_input, t, encoder_hidden_states=encoder_hidden_states).sample
            noise_pred_uncond = self.unet(
                latents_input,
                t,
                encoder_hidden_states=uncond_embeddings,
                cross_attention_kwargs={"front_face_drop": True},
            ).sample

            combined = noise_pred_uncond + cfg_scale * (noise_pred - noise_pred_uncond)
            latents[1:] = self.scheduler.step(combined[1:], t, latents[1:]).prev_sample  

        # --- decode ---------------------------------------------------------
        imgs = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        equirec, uncropped, cropped = postprocess_outputs(imgs)

        return CubeDiffPipelineOutput(
            faces=uncropped,
            faces_cropped=cropped,
            equirectangular=equirec,
        )
