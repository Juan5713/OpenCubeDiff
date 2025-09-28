import os

os.environ["WANDB_DISABLE_KERAS"] = "1"
os.environ["WANDB_DISABLE_TENSORBOARD"] = "1"

import argparse
import time
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from omegaconf import DictConfig, OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from diffusers import DDIMScheduler
from diffusers.utils.torch_utils import is_compiled_module
from training.dataset import CubemapDataset, cubemap_collate_fn
from cubediff.pipelines.pipeline import CubeDiffPipeline
from cubediff.modules.extra_channels import make_extra_channels_tensor

def main(cfg: DictConfig):

    # ---------------------- Accelerator, wandb Setup --------------------------

    num_gpus = cfg.training.num_gpus
    per_gpu_batch_size = cfg.training.per_gpu_batch_size
    global_batch_size = cfg.training.batch_size
    acc_steps = global_batch_size // (per_gpu_batch_size * num_gpus)

    accelerator = Accelerator(
        mixed_precision=cfg.training.mixed_precision,
        gradient_accumulation_steps= acc_steps,
    )

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if accelerator.is_main_process:
        print(f"[INFO] Accelerator detected {accelerator.num_processes} processes (GPUs)")
        print(f"[INFO] Mixed precision: {cfg.training.mixed_precision}")

        # Validate checkpoint_interval_type
        if cfg.training.checkpoint_interval_type not in ["epochs", "steps"]:
            raise ValueError(f"Invalid checkpoint_interval_type: {cfg.training.checkpoint_interval_type}. Must be 'epochs' or 'steps'")
        print(f"[INFO] Checkpoint interval type: {cfg.training.checkpoint_interval_type}")

        import wandb

        # Initialize wandb with the configuration
        wandb.init(entity=cfg.wandb.entity_name, 
                project=cfg.wandb.project_name, 
                config=OmegaConf.to_container(cfg, resolve=True),
                name=cfg.name)
    
    # ---------------------- Seeding ----------------------------

    rank = accelerator.process_index
    base_seed = cfg.training.seed    

    local_seed = base_seed + rank
    torch.manual_seed(local_seed)
    np.random.seed(local_seed)
    
    set_seed(base_seed, device_specific=True)
    
    # ---------------------- Load Pipeline & Patch ----------------------
    if accelerator.is_main_process:
        print("[DEBUG] Loading pipeline...")

    pipe = CubeDiffPipeline.from_pretrained(cfg.model.id, cache_dir=cfg.directories.cache_dir, local_files_only=True)
    
    if accelerator.is_main_process:
        print("[DEBUG] Pipeline loaded.")

    if cfg.training.prediction_type == "v_prediction":
        # Change the scheduler to DDIM with v-prediction (the other ones are all epsilon by default)
        pipe.scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="scheduler", cache_dir=cfg.directories.cache_dir, local_files_only=True)
        pipe.scheduler.config.prediction_type = "v_prediction"
    elif cfg.training.prediction_type == "epsilon":
        # Instantiate a DDIM scheduler with the same config as the original scheduler
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.config.prediction_type = "epsilon"
    else:
        raise ValueError(f"Invalid prediction type: {cfg.training.prediction_type}")

    # Set latent size according to image size
    if cfg.model.image_size == 512:
        latent_H = 64
        latent_W = 64
    elif cfg.model.image_size == 768:
        latent_H = 96
        latent_W = 96
    else:
        raise ValueError(f"Invalid image size: {cfg.model.image_size}, must be 512 or 768")

    # ---------------------- Freeze and Unfreeze Parameters ----------------------
    for param in pipe.vae.parameters():
        param.requires_grad = False

    for param in pipe.unet.parameters():
        param.requires_grad = False

    for param in pipe.text_encoder.parameters():
        param.requires_grad = False
    
    for name, param in pipe.unet.conv_in.named_parameters():
        param.requires_grad = True

    for name, param in pipe.unet.named_parameters():
        if "attn" in name:
            param.requires_grad = True
    
    # ---------------------- Dataset and DataLoader ----------------------
    if accelerator.is_main_process:
        print(f"[DEBUG] Creating dataset from {cfg.directories.data_dir}")

    dataset = CubemapDataset(root_dir=cfg.directories.data_dir, 
                             face_size=cfg.model.image_size, 
                             fov=95, 
                             checkpoint_dir=cfg.directories.checkpoint_dir, 
                             use_cached_data=False)
                    
    if accelerator.is_main_process:
        print(f"[DEBUG] Dataset size: {len(dataset)}")

    cubemap_faces, conditioning_prompt, _, _, conditioning_single_caption = dataset[1] # Get a sample from the dataset
    conditioning_image = cubemap_faces[0] # Get first image from dataset
    conditioning_image = conditioning_image.to(accelerator.device)

    # ---------------------- Effective Batch Size Setup ----------------------
    per_gpu_batch_size = cfg.training.per_gpu_batch_size
    global_batch_size = cfg.training.batch_size

    # -------------------- Dataloader --------------------
    dataloader = DataLoader(
        dataset,
        batch_size=per_gpu_batch_size,
        shuffle=True,
        collate_fn=cubemap_collate_fn,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    
    # ---------------------- Optimizer & LR Scheduler ----------------------
    params = [p for p in pipe.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.training.learning_rate, betas=tuple(cfg.training.betas), eps=cfg.training.eps)
    
    def lr_lambda(current_step):
        return current_step / cfg.training.warmup_steps if current_step < cfg.training.warmup_steps else 1.0
    
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # --------------------- Prepare components -----------------------
    unet, optimizer, dataloader = accelerator.prepare(
        pipe.unet, optimizer, dataloader
    )
    
    if accelerator.is_main_process:
        eff_bs = per_gpu_batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps
        print(f"[INFO] Per-GPU batch size: {per_gpu_batch_size}")
        print(f"[INFO] Effective global batch size: {eff_bs}")

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    pipe.vae.to(accelerator.device, dtype=weight_dtype)
    pipe.text_encoder.to(accelerator.device, dtype=weight_dtype)

    # ---------------------- Resume Training from Checkpoint ----------------------
    global_step = 0
    if cfg.training.resume_checkpoint:
        resume_ckpt = os.path.join(cfg.directories.checkpoint_dir, cfg.training.resume_checkpoint_filename)
        if not os.path.exists(resume_ckpt):
            raise ValueError(f"Checkpoint folder not found: {resume_ckpt}")
        ckpt_name = os.path.basename(resume_ckpt)

        # saving format is epoch_<num>_step_<num>
        start_epoch = int(ckpt_name.split("epoch_")[-1].split("_")[0])
        global_step = int(ckpt_name.split("step_")[-1].split("_")[0])
        
        accelerator.load_state(resume_ckpt)
        if accelerator.is_main_process:
            print(f"[INFO] Resumed training from checkpoint: {resume_ckpt}")
            print(f"[INFO] Resuming at epoch {start_epoch}")

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda, last_epoch=global_step - 1
        )

    else:
        start_epoch = 0

    # ---------------------- Training Loop ----------------------
    epochs = cfg.training.epochs
    T = 6
    checkpoint_dir = os.path.join(cfg.directories.checkpoint_dir, f"{cfg.name}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    step_start_time = time.time()
    if accelerator.is_main_process:
        print("[DEBUG] Starting training...")
    
    for epoch in range(start_epoch, epochs):
        if accelerator.is_main_process:
            print(f"Epoch {epoch + 1}")

        unet.train()
        optimizer.zero_grad()

        # Loss accumulation for logging (accumulate across gradient accumulation steps)
        total_loss = 0.0
        for _, batch in enumerate(dataloader):

            cubemap_batch, prompts, _, _, single_captions = batch
            with accelerator.accumulate(unet):
                with accelerator.autocast():
                    with torch.no_grad():
                        latents = pipe.vae.encode(cubemap_batch).latent_dist.mean
                        latents = latents * pipe.vae.config.scaling_factor

                        B = latents.shape[0] // T
                        _, _, H, W = latents.shape

                        # Drop conditioning prompts with a 10% chance
                        mask = torch.rand(B, device=accelerator.device) < 0.1
                        single_captions = [sc if not m else "" for sc, m in zip(single_captions, mask)]

                        full_mask = mask.repeat_interleave(T)  # Now shape [B * T]
                        prompts = [p if not m else "" for p, m in zip(prompts, full_mask)]

                        # Conditioning prompts always dropped for image only
                        if cfg.training.type == "image_only":
                            prompts = [""] * len(prompts)
                        elif cfg.training.type == "single_caption":
                            # Use single caption for all faces, repeat for each face in the batch
                            prompts = []
                            for sc in single_captions:
                                prompts.extend([sc] * T)  # Repeat single caption for all 6 faces
                        elif cfg.training.type == "multitext":
                            # Use individual face prompts (keep original behavior)
                            pass  # prompts already contain the per-face prompts
                        else:
                            raise ValueError(f"Invalid training type: {cfg.training.type}")

                        # Prepare text inputs and encoder hidden states
                        text_inputs = pipe.tokenizer(
                            prompts, padding="max_length", truncation=True, max_length=77, return_tensors="pt"
                        )

                        encoder_hidden_states = pipe.text_encoder(text_inputs.input_ids.to(accelerator.device))[0]
                        
                        # Generate random timesteps for each sample
                        timesteps = torch.randint(
                            0, pipe.scheduler.config.num_train_timesteps, (B,),
                            device=latents.device, dtype=torch.long
                        )

                        # Propagate it to all 6 faces
                        timesteps = timesteps.repeat_interleave(T)

                        # Create mask to identify non-front faces
                        face_indices = torch.arange(B * T, device=latents.device)
                        face_ids = face_indices % T
                        non_front_mask = face_ids != 0

                        # Create noise only for the non-front faces
                        noise = torch.randn_like(latents[non_front_mask])

                        # Copy original latents for velocity prediction
                        orig_latents = latents.clone()

                    # For the inputs only noise the non-front faces, keep front face clean
                    latents[non_front_mask] = pipe.scheduler.add_noise(
                        latents[non_front_mask], noise, timesteps[non_front_mask]
                    ).to(latents.dtype)

                    extra_channels = make_extra_channels_tensor(B, H, W).to(latents.device)
                    latent_input = torch.cat([latents, extra_channels], dim=1)

                    # For the whole mini-batch # Hacky
                    front_face_drop = torch.rand(1, device=accelerator.device) < 0.1

                    model_pred = unet(
                        latent_input,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs={"front_face_drop": front_face_drop}, # This triggers masking in the attention
                    ).sample

                    if cfg.training.prediction_type == "v_prediction":
                        v_target = pipe.scheduler.get_velocity(
                            orig_latents[non_front_mask], noise, timesteps[non_front_mask]
                        )
                        loss = nn.functional.mse_loss(model_pred[non_front_mask], v_target)
                    
                    elif cfg.training.prediction_type == "epsilon":
                        loss = nn.functional.mse_loss(model_pred[non_front_mask], noise)
                    else:
                        raise ValueError(f"Invalid prediction type: {cfg.training.prediction_type}")
                
                    # Accumulate loss for logging (before backward pass scales it)
                    total_loss += loss.detach().float()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    # Clip the gradients to one
                    trainable_params = [p for p in unet.parameters() if p.requires_grad]
                    grad_norm = accelerator.clip_grad_norm_(trainable_params, max_norm=1.0)

                optimizer.step()

                if accelerator.sync_gradients:
                    # Step the learning rate scheduler
                    lr_scheduler.step()

                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    global_step += 1
                    # Record time
                    time_elapsed = time.time() - step_start_time
                    step_start_time = time.time()

                    # Calculate average loss for this process over accumulation steps
                    avg_loss_this_process = total_loss / accelerator.gradient_accumulation_steps
                    # Gather average losses from all processes
                    gathered_losses = accelerator.gather_for_metrics(avg_loss_this_process.unsqueeze(0))
                    # Calculate the mean loss across all processes
                    avg_loss = torch.mean(gathered_losses)
                    # Reset loss accumulator for the next global batch
                    total_loss = 0.0

                    if accelerator.is_main_process:
                        wandb.log({"loss": avg_loss.item(), # Log the averaged loss
                                "time_per_global_step": time_elapsed,
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "grad_norm": grad_norm.item()
                            }, step=global_step)

                    # Step-based checkpointing
                    if cfg.training.checkpoint_interval_type == "steps" and global_step % cfg.training.checkpoint_interval == 0:
                        ckpt_folder = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}_step_{global_step}")
                        accelerator.save_state(ckpt_folder)

                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            print(f"[INFO] Checkpoint saved: {ckpt_folder}")
                            # Clean up old checkpoints
                            all_ckpts = sorted(
                                [f for f in os.listdir(checkpoint_dir) if f.startswith("epoch_")],
                                key=lambda f: int(f.split("_")[1].split("_")[0])
                            )
                            max_ckpt_len = 40
                            if len(all_ckpts) > max_ckpt_len:
                                ckpts_to_delete = all_ckpts[:-max_ckpt_len]
                                for old_ckpt in ckpts_to_delete:
                                    full_path = os.path.join(checkpoint_dir, old_ckpt)
                                    if os.path.isdir(full_path):
                                        import shutil
                                        shutil.rmtree(full_path)
                                    else:
                                        os.remove(full_path)
                                    print(f"[INFO] Deleted old checkpoint: {old_ckpt}")
                        accelerator.wait_for_everyone()

                    # Step-based validation
                    if cfg.training.checkpoint_interval_type == "steps" and global_step % cfg.training.validation_interval == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            print(f"[INFO] Generating sample images for step {global_step}...")
                            # Generate and log sample images
                            pipe.unet = unwrap_model(unet)
                            pipe.unet.eval()

                            # Determine prompt based on training type
                            if cfg.training.type == "image_only":
                                validation_prompt = ""
                                validation_captions = [""] * 6
                            elif cfg.training.type == "single_caption":
                                validation_prompt = conditioning_single_caption
                                validation_captions = [conditioning_single_caption] * 6
                            elif cfg.training.type == "multitext":
                                validation_prompt = conditioning_prompt
                                validation_captions = conditioning_prompt
                            else:
                                validation_prompt = ""
                                validation_captions = [""] * 6

                            pipeline_output = pipe(
                                prompts=validation_prompt,
                                conditioning_image=conditioning_image,
                                num_inference_steps=50,
                                cfg_scale=3.5,
                            )

                            pil_equirec = Image.fromarray(pipeline_output.equirectangular)
                            pil_faces = [Image.fromarray(face) for face in pipeline_output.faces]

                            wandb.log({
                                "epoch": epoch + 1,
                                "equirectangular_sample": wandb.Image(pil_equirec, caption=f"Step {global_step} Equirec"),
                                "face_samples": [wandb.Image(face, caption=f"{validation_captions[i]}") for i, face in enumerate(pil_faces)],
                            }, step=global_step)

                            del pipeline_output, pil_equirec, pil_faces
                            torch.cuda.empty_cache()
                        accelerator.wait_for_everyone()

        # Epoch-based checkpointing
        if cfg.training.checkpoint_interval_type == "epochs" and (epoch + 1) % cfg.training.checkpoint_interval == 0:
            ckpt_folder = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}_step_{global_step}")
            accelerator.save_state(ckpt_folder)

            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                print(f"[INFO] Checkpoint saved: {ckpt_folder}")
                # Clean up old checkpoints
                all_ckpts = sorted(
                    [f for f in os.listdir(checkpoint_dir) if f.startswith("epoch_")],
                    key=lambda f: int(f.split("_")[1].split("_")[0])
                )
                max_ckpt_len = 40
                if len(all_ckpts) > max_ckpt_len:
                    ckpts_to_delete = all_ckpts[:-max_ckpt_len]
                    for old_ckpt in ckpts_to_delete:
                        full_path = os.path.join(checkpoint_dir, old_ckpt)
                        if os.path.isdir(full_path):
                            import shutil
                            shutil.rmtree(full_path)
                        else:
                            os.remove(full_path)
                        print(f"[INFO] Deleted old checkpoint: {old_ckpt}")
            accelerator.wait_for_everyone() 

        # Epoch-based validation
        if cfg.training.checkpoint_interval_type == "epochs" and (epoch + 1) % cfg.training.validation_interval == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                print(f"[INFO] Generating sample images for epoch {epoch + 1}...")
                # Generate and log sample images
                pipe.unet = unwrap_model(unet)
                pipe.unet.eval()

                # Determine prompt based on training type
                if cfg.training.type == "image_only":
                    validation_prompt = ""
                    validation_captions = [""] * 6
                elif cfg.training.type == "single_caption":
                    validation_prompt = conditioning_single_caption
                    validation_captions = [conditioning_single_caption] * 6
                elif cfg.training.type == "multitext":
                    validation_prompt = conditioning_prompt
                    validation_captions = conditioning_prompt
                else:
                    validation_prompt = ""
                    validation_captions = [""] * 6

                pipeline_output = pipe(
                    prompts=validation_prompt,
                    conditioning_image=conditioning_image,
                    num_inference_steps=50,
                    cfg_scale=3.5,
                )
                
                pil_equirec = Image.fromarray(pipeline_output.equirectangular)
                pil_faces = [Image.fromarray(face) for face in pipeline_output.faces]

                wandb.log({
                    "epoch": epoch + 1,
                    "equirectangular_sample": wandb.Image(pil_equirec, caption=f"Epoch {epoch + 1} Equirec"),
                    "face_samples": [wandb.Image(face, caption=f"{validation_captions[i]}") for i, face in enumerate(pil_faces)],
                }, step=global_step)

                del pipeline_output, pil_equirec, pil_faces
                torch.cuda.empty_cache()
            accelerator.wait_for_everyone()
    
    # Save the final model
    ckpt_filename = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}_step_{global_step}_final")
    accelerator.save_state(ckpt_filename)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print(f"[INFO] Checkpoint saved: {ckpt_filename}")
        wandb.finish()

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    # Set NCCL env variables
    os.environ["NCCL_TIMEOUT"] = "3600"        # 1hr timeout
    os.environ["NCCL_DEBUG"] = "INFO"          # Enable debug logging

    # Use argparse to select which Hydra config to use
    parser = argparse.ArgumentParser(description="Train CubeDiff with Hydra configs")
    parser.add_argument("--config", type=str, default="training",
                        help="Name of the Hydra config to use (without .yaml extension)")
    
    # Allow additional overrides to be passed in
    args, overrides = parser.parse_known_args()

    from hydra import initialize, compose
    with initialize(config_path="training/configs", version_base=None):
        cfg = compose(config_name=args.config, overrides=overrides)
        main(cfg)
