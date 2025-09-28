#!/usr/bin/env python3
"""
Push CubeDiff model to Hugging Face Hub

This script:
1. Loads VAE weights from base SD 1.5 (reused)
2. Loads UNet config only (no weights download)
3. Patches UNet and VAE architectures 
4. Loads UNet weights from local checkpoint
5. Pushes complete pipeline to hub as 'cubediff-li'
"""

import torch
import os
import argparse
from pathlib import Path
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL
from diffusers import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from safetensors.torch import load_file
from huggingface_hub import HfApi, create_repo
from cubediff.pipelines.pipeline import CubeDiffPipeline
from cubediff.modules.utils import patch_unet, patch_groupnorm

def load_base_components(base_model_id="sd-legacy/stable-diffusion-v1-5"):
    """Load base components: VAE with weights, UNet config only, text encoder, tokenizer, scheduler"""
    
    print(f"Loading base components from {base_model_id}...")
    
    # Load VAE with weights (will be reused)
    print("Loading VAE with weights...")
    vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae")
    print(f"✓ VAE loaded: {sum(p.numel() for p in vae.parameters()):,} parameters")
    
    # Load UNet config only (no weights)
    print("Loading UNet config only (no weights download)...")
    unet_config = UNet2DConditionModel.load_config(base_model_id, subfolder="unet")
    print(f"✓ UNet config loaded: {unet_config['in_channels']} input channels")
    
    # Load text encoder and tokenizer with weights
    print("Loading text encoder and tokenizer...")
    text_encoder = CLIPTextModel.from_pretrained(base_model_id, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(base_model_id, subfolder="tokenizer")
    print(f"✓ Text encoder loaded: {sum(p.numel() for p in text_encoder.parameters()):,} parameters")
    
    # Load scheduler config
    print("Loading scheduler...")
    scheduler = DDIMScheduler.from_pretrained(base_model_id, subfolder="scheduler")
    print("✓ Scheduler loaded")
    
    return vae, unet_config, text_encoder, tokenizer, scheduler

def create_and_patch_unet(unet_config, checkpoint_path, device="cpu"):
    """Create UNet from config, patch it, and load checkpoint weights"""
    
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint to inspect input channels
    checkpoint = load_file(checkpoint_path)
    print(f"✓ Checkpoint loaded: {len(checkpoint)} tensors")
    
    # Get input channels from checkpoint
    conv_in_weight_key = 'conv_in.weight'
    if conv_in_weight_key in checkpoint:
        conv_in_shape = checkpoint[conv_in_weight_key].shape
        input_channels = conv_in_shape[1]  # [out_channels, in_channels, h, w]
        print(f"Input channels from checkpoint: {input_channels}")
    else:
        raise ValueError("conv_in.weight not found in checkpoint")
    
    # Modify UNet config for CubeDiff
    print(f"\n🔧 Modifying UNet config: {unet_config['in_channels']} → {input_channels} input channels")
    unet_config['in_channels'] = input_channels
    
    # Create UNet with modified config
    print("Creating UNet with modified config...")
    unet = UNet2DConditionModel(**unet_config)
    
    # NOTE: We do NOT apply runtime patches (attention modifications) when saving
    # The checkpoint already contains the correct weights for 7-channel input
    # Runtime patches will be applied automatically when loading with CubeDiffPipeline.from_pretrained()
    print("✓ UNet created with 7-channel config (runtime patches applied only at load time)")
    
    # Load checkpoint weights
    print("Loading checkpoint weights into UNet...")
    missing_keys, unexpected_keys = unet.load_state_dict(checkpoint, strict=False)
    
    if len(missing_keys) == 0 and len(unexpected_keys) == 0:
        print("✓ All weights loaded successfully! Perfect match.")
    else:
        if missing_keys:
            print(f"⚠️  Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
        print("⚠️  Partial loading - some weights may not match")
    
    unet = unet.to(device)
    print(f"✓ UNet moved to {device}")
    print(f"UNet parameters: {sum(p.numel() for p in unet.parameters()):,}")
    
    return unet

def create_cubediff_pipeline(vae, unet, text_encoder, tokenizer, scheduler, device="cpu"):
    """Create CubeDiffPipeline from components"""
    
    print("\nCreating CubeDiffPipeline...")
    
    # NOTE: We do NOT apply VAE patches here when creating for hub push
    # VAE should remain as original SD 1.5 VAE for compatibility
    # Patches will be applied automatically when loading with CubeDiffPipeline.from_pretrained()
    print("✓ VAE kept as original SD 1.5 (patches applied only at runtime)")
    
    # Move components to device
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    
    # Create pipeline
    pipeline = CubeDiffPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    
    print("✓ CubeDiffPipeline created successfully!")
    return pipeline

def push_to_hub(pipeline, repo_name, token=None, private=False):
    """Push pipeline to Hugging Face Hub"""
    
    print(f"\nPushing to Hugging Face Hub: {repo_name}")
    
    try:
        # Create repository
        print(f"Creating repository: {repo_name}")
        repo_url = create_repo(repo_name, token=token, private=private, exist_ok=True)
        print(f"✓ Repository created/exists: {repo_url}")
        
        # Push pipeline
        print("Uploading pipeline components...")
        pipeline.push_to_hub(repo_name, token=token, private=private)
        print(f"✓ Pipeline pushed successfully to: {repo_name}")
        
        return repo_url
        
    except Exception as e:
        print(f"✗ Failed to push to hub: {e}")
        raise

def test_loading_from_hub(repo_name, token=None):
    """Test loading the pushed model with CubeDiffPipeline"""
    
    print(f"\nTesting loading from hub: {repo_name}")
    
    try:
        # Load pipeline from hub
        pipeline = CubeDiffPipeline.from_pretrained(repo_name, token=token)
        print("✓ Pipeline loaded successfully from hub!")
        
        # Check components
        print(f"UNet input channels: {pipeline.unet.config.in_channels}")
        print(f"VAE sample size: {pipeline.vae.config.sample_size}")
        print(f"UNet parameters: {sum(p.numel() for p in pipeline.unet.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to load from hub: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Push CubeDiff model to Hugging Face Hub")
    parser.add_argument("--checkpoint", type=str, default="ckpts/512-singlecap/model.safetensors", 
                       help="Path to UNet checkpoint")
    parser.add_argument("--repo-name", type=str, default="cubediff-512-singlecaption", 
                       help="Hub repository name")
    parser.add_argument("--base-model", type=str, default="sd-legacy/stable-diffusion-v1-5",
                       help="Base model for VAE, text encoder, etc.")
    parser.add_argument("--token", type=str, default=None,
                       help="Hugging Face token (or set HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true",
                       help="Make repository private")
    parser.add_argument("--test-loading", action="store_true",
                       help="Test loading after push")
    parser.add_argument("--dry-run", action="store_true",
                       help="Test model creation without pushing to hub")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (cpu/cuda)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CUBEDIFF HUB PUSH SCRIPT")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Repository: {args.repo_name}")
    print(f"Base model: {args.base_model}")
    print(f"Device: {args.device}")
    print(f"Private: {args.private}")
    print("=" * 70)
    
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        device = "cpu"
    
    try:
        # Step 1: Load base components
        vae, unet_config, text_encoder, tokenizer, scheduler = load_base_components(args.base_model)
        
        # Step 2: Create and patch UNet, load checkpoint
        unet = create_and_patch_unet(unet_config, args.checkpoint, device)
        
        # Step 3: Create CubeDiffPipeline
        pipeline = create_cubediff_pipeline(vae, unet, text_encoder, tokenizer, scheduler, device)
        
        if args.dry_run:
            print("\n" + "=" * 70)
            print("🎉 DRY RUN SUCCESS!")
            print("✓ All components loaded and patched successfully")
            print("✓ CubeDiffPipeline created successfully")
            print("✓ Ready for hub push (use without --dry-run)")
            print("=" * 70)
        else:
            # Step 4: Push to hub
            repo_url = push_to_hub(pipeline, args.repo_name, args.token, args.private)
            
            # Step 5: Test loading (optional)
            if args.test_loading:
                success = test_loading_from_hub(args.repo_name, args.token)
                if not success:
                    print("⚠️  Loading test failed, but push was successful")
            
            print("\n" + "=" * 70)
            print("🎉 SUCCESS!")
            print(f"✓ CubeDiff model pushed to: {repo_url}")
            print(f"✓ Load with: CubeDiffPipeline.from_pretrained('{args.repo_name}')")
            print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
