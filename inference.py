import torch
import os
from PIL import Image
from torchvision import transforms
from cubediff.pipelines.pipeline import CubeDiffPipeline

if __name__ == "__main__":
    # ============== USER CONFIGURATION ==============
    # Modify these variables directly in the script
    
    # Image filename
    IMAGE_FILENAME = "your-image-file"  # Change this to your image filename
    
    # Either "", or single prompt, or list of 6 prompts depending on the type of checkpoint you are loading
    PROMPTS = ""
    
    # Model checkpoint path
    CHECKPOINT = "hlicai/cubediff-512-imgonly"  # Change this to one of the three types of checkpoints
    # "hlicai/cubediff-512-singlecaption"
    # "hlicai/cubediff-512-multitxt"

    # Output directory
    OUTPUT_DIR = f"output/{IMAGE_FILENAME}_generations/"  # Change this to your desired output directory
    
    # Classifier-free guidance scale
    CFG_SCALE = 3.5
    
    # ================================================

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------- Load Pipeline ----------------

    print(f"Loading pipeline from: {CHECKPOINT}")
    pipe = CubeDiffPipeline.from_pretrained(
        CHECKPOINT,
    )
    pipe = pipe.to(device)
    print(f"Pipeline loaded successfully and moved to {device}")

    image_size = pipe.vae.config.sample_size

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n[INFO] Loading image: {IMAGE_FILENAME}")
    image = Image.open(IMAGE_FILENAME).convert("RGB")
    conditioning_image = transform(image)

    print("[INFO] Running inference...")
    output = pipe(
        prompts=PROMPTS,
        conditioning_image=conditioning_image.unsqueeze(0).to(device),
        num_inference_steps=50,
        cfg_scale=CFG_SCALE,
    )

    # ---------------- Save Results ----------------
    
    # Save the face images
    for face_name, face_img in zip(["front", "back", "left", "right", "top", "bottom"], output.faces_cropped):
        face_img = Image.fromarray(face_img)
        face_img.save(os.path.join(OUTPUT_DIR, f"{face_name}.png"))
        print(f"[INFO] Saved {face_name} face to {OUTPUT_DIR}/{face_name}.png")
    
    # Save the equirectangular image
    equirec_img = Image.fromarray(output.equirectangular)
    equirec_img.save(os.path.join(OUTPUT_DIR, "equirectangular.png"))
    print(f"[INFO] Saved equirectangular image to {OUTPUT_DIR}/equirectangular.png")

    print("\n[INFO] Inference completed successfully!")
