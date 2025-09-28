import torch.nn as nn
import torch
from diffusers import UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from .attention import CubeDiffTransformerBlock
from .norm import CubeDiffGroupNorm


def freeze(module: nn.Module) -> None:
    """
        Freeze all parameters in a module so they do not require gradients.
    """
    for p in module.parameters():
        p.requires_grad = False

def swap_transformer_blocks(root: nn.Module) -> None:
    """Replace every `BasicTransformerBlock` inside `Transformer2DModel`."""
    for child in root.children():
        swap_transformer_blocks(child)
        if isinstance(child, Transformer2DModel):
            for i, blk in enumerate(child.transformer_blocks):
                if isinstance(blk, BasicTransformerBlock):
                    new_blk = CubeDiffTransformerBlock(
                        dim=blk.dim,
                        num_attention_heads=blk.num_attention_heads,
                        attention_head_dim=blk.attention_head_dim,
                        dropout=blk.dropout,
                        cross_attention_dim=blk.cross_attention_dim,
                        activation_fn=blk.activation_fn,
                        num_embeds_ada_norm=getattr(child.config, 'num_embeds_ada_norm', None),
                        attention_bias=blk.attention_bias,
                        only_cross_attention=blk.only_cross_attention,
                        double_self_attention=blk.double_self_attention,
                        norm_elementwise_affine=blk.norm_elementwise_affine,
                        norm_type=blk.norm_type,
                        norm_eps=getattr(child.config, 'norm_eps', 1e-5),
                        upcast_attention=getattr(child.config, 'upcast_attention', False),
                        attention_type=getattr(child.config, 'attention_type', 'default'),
                    )
                    # Load the state dict with proper error handling
                    try:
                        new_blk.load_state_dict(blk.state_dict(), strict=False)
                    except RuntimeError as e:
                        print(f"Warning: Could not load state dict completely: {e}")
                        # Copy compatible weights manually
                        new_state = new_blk.state_dict()
                        old_state = blk.state_dict()
                        for key in new_state.keys():
                            if key in old_state and new_state[key].shape == old_state[key].shape:
                                new_state[key].copy_(old_state[key])
                        new_blk.load_state_dict(new_state)
                    
                    child.transformer_blocks[i] = new_blk

def expand_input_conv(unet: UNet2DConditionModel, new_channels: int) -> None:
    """Grow `conv_in` to `new_channels`, copying the first 4 kernels."""
    old = unet.conv_in
    if old.in_channels == new_channels:
        return
    
    new = nn.Conv2d(
        new_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=old.bias is not None,
    )
    
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, : old.in_channels] = old.weight
        if old.bias is not None:
            new.bias.copy_(old.bias)
    
    unet.conv_in = new


def patch_unet(unet: UNet2DConditionModel, in_channels: int = 7) -> UNet2DConditionModel:
    """Patch a base UNet to CubeDiff architecture."""

    # Swap transformer blocks
    swap_transformer_blocks(unet)
    
    # Expand input conv layer
    expand_input_conv(unet, in_channels)

    # Update config
    unet.register_to_config(in_channels=in_channels)
    
    return unet


def patch_groupnorm(root: nn.Module, num_faces: int = 6) -> None:
    """Recursively replace GroupNorm with CubeDiffGroupNorm (in-place)."""
    for name, child in list(root.named_children()):
        patch_groupnorm(child, num_faces=num_faces)
        if isinstance(child, nn.GroupNorm) and not isinstance(child, CubeDiffGroupNorm):
            setattr(root, name, CubeDiffGroupNorm(child, num_faces=num_faces))
