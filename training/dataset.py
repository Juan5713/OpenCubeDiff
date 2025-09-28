import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import filetype
import py360convert
import pickle
import concurrent.futures

def wrap_yaw(yaw):
    return ((yaw + 180) % 360) - 180

def rotate_middle_captions(prompts, yaw_deg):
    """
    Rotate the middle four captions (front, back, left, right) according to yaw shift.
    Hard-coded rotation logic for clarity.

    Args:
        prompts: List of 6 strings or embeddings, ordered as [front, back, left, right, top, bottom]
        yaw_deg: Integer, yaw between [-180, 180]

    Returns:
        List of 6 strings with rotated middle faces, top and bottom unchanged.
    """
    
    if yaw_deg < -180 or yaw_deg > 180:
        raise ValueError(f"Invalid yaw_deg: {yaw_deg}. Must be in [-180, 180].")

    if -45 <= yaw_deg < 45:
        steps = 0  # front
    elif 45 <= yaw_deg < 135:
        steps = 1  # right
    elif -135 <= yaw_deg < -45:
        steps = 3  # left
    else:
        steps = 2  # back

    # Unpack the prompts
    front, back, left, right, top, bottom = prompts

    if steps == 0: # [-45, 45)
        rotated_middle = [front, back, left, right]
    elif steps == 1:  # [45, 135)
        rotated_middle = [right, left, front, back]
    elif steps == 2:  # [135, 180) union [-180, -135)
        rotated_middle = [back, front, right, left]
    elif steps == 3:  # [-135, -45)
        rotated_middle = [left, right, back, front]
    else:
        pass

    return rotated_middle + [top, bottom]


def is_valid_image_header(file_path):
    """
    Quickly checks if a file has an image header using filetype.
    
    Args:
        file_path (str): Path to the file.
    Returns:
        bool: True if the file appears to be an image; False otherwise.
    """
    try:
        kind = filetype.guess(file_path)
        if kind is None:
            return False
        # Valid if the mime type begins with 'image/'
        return kind.mime.startswith("image/")
    except Exception as e:
        print(f"[ERROR] Header check failed for {file_path}: {e}")
        return False

class CubemapDataset(Dataset):
    def __init__(self, root_dir, face_size=512, fov=95, max_attempts=10, checkpoint_dir=None, use_cached_data=False, augment=True):
        """
        Initializes the dataset by scanning the directory structure for image files.
        All files are filtered during the pre-scan using a file header check.
        
        Args:
            root_dir (str): The root directory containing scene subdirectories.
            face_size (int): Size of each cubemap face.
            fov (int): Field of view used for conversion.
            max_attempts (int): Maximum attempts to retrieve a valid sample.
            augment (bool): Whether to apply augmentations.
        """
        self.root_dir = root_dir
        self.face_size = face_size
        self.fov = fov
        self.max_attempts = max_attempts
        # List all scene directories.
        self.scenes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        # Create a list of (scene, image_file) tuples.
        self.samples = []
        self.face_order = ["front", "back", "left", "right", "top", "bottom"]
        self.augment = augment

        pickle_file = os.path.join(checkpoint_dir, "valid_samples.pkl") if checkpoint_dir else None

        # If using cached data and pickle exists, load it.
        if use_cached_data and pickle_file is not None and os.path.exists(pickle_file):
            print(f"[INFO] Loading valid image samples from {pickle_file}...")
            with open(pickle_file, "rb") as f:
                self.samples = pickle.load(f)
            print(f"[INFO] Loaded {len(self.samples)} valid image samples from pickle.")
        else:
            print("[INFO] Pre-scanning for image files...")

            # Helper function for validating a single image file.
            def validate_sample(scene, scene_path, image_file):
                if image_file.startswith("."):
                    return None
                img_path = os.path.join(scene_path, image_file)
                if not is_valid_image_header(img_path):
                    return None

                return (scene, image_file)
            tasks = []
            # Parallelize validation
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for scene in self.scenes:
                    scene_path = os.path.join(root_dir, scene)
                    images = [f for f in os.listdir(scene_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    for img in images:
                        # Submit each sample for validation in parallel.
                        tasks.append(executor.submit(validate_sample, scene, scene_path, img))
                # Collect valid samples as they complete.
                for future in concurrent.futures.as_completed(tasks):
                    result = future.result()
                    if result is not None:
                        self.samples.append(result)

            print(f"[INFO] Total image samples found: {len(self.samples)}")

            if pickle_file is not None:
                os.makedirs(checkpoint_dir, exist_ok=True)
                with open(pickle_file, "wb") as f:
                    pickle.dump(self.samples, f)
                print(f"[INFO] Saved valid image samples to {pickle_file}")

        # Check for missing prompt files and report statistics
        self._check_prompt_files()

    def _check_prompt_files(self):
        """
        Check for missing prompt files and print statistics.
        """
        print("[INFO] Checking for prompt files...")
        
        # Get unique scenes from samples
        unique_scenes = set(sample[0] for sample in self.samples)
        total_scenes = len(unique_scenes)
        missing_prompts = 0
        missing_prompts_single = 0
        
        for scene in unique_scenes:
            scene_path = os.path.join(self.root_dir, scene)
            prompts_path = os.path.join(scene_path, "prompts.json")
            prompts_single_path = os.path.join(scene_path, "prompts_single.json")
            
            if not os.path.exists(prompts_path):
                missing_prompts += 1
            
            if not os.path.exists(prompts_single_path):
                missing_prompts_single += 1
        
        # Calculate percentages
        if total_scenes > 0:
            missing_percentage = (missing_prompts / total_scenes) * 100
            missing_single_percentage = (missing_prompts_single / total_scenes) * 100
            print(f"[INFO] Prompt files missing: {missing_prompts}/{total_scenes} ({missing_percentage:.1f}%)")
            print(f"[INFO] Single prompt files missing: {missing_prompts_single}/{total_scenes} ({missing_single_percentage:.1f}%)")
        else:
            print("[INFO] No scenes found to check for prompt files.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Loads an image sample lazily. If processing fails, the sample is removed
        from the sample list and another sample is tried (up to max_attempts).
        """
        attempts = 0
        while attempts < self.max_attempts and len(self.samples) > 0:
            # Ensure index is within current range
            idx = idx % len(self.samples)
            scene, image_file = self.samples[idx]
            scene_path = os.path.join(self.root_dir, scene)
            img_path = os.path.join(scene_path, image_file)

            try:
                # Load the image
                img = cv2.imread(img_path)
                if img is None or img.size == 0:
                    raise ValueError("Empty or unreadable image.")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                # Normalize the image to range [-1, 1]
                img = (img * 2.0 - 1.0).clip(-1, 1)

                # Generate random yaw and compute face angles
                yaw_front = np.random.randint(-180, 181) if self.augment else 0
                face_angles = {
                    "front": (wrap_yaw(yaw_front), 0),
                    "back": (wrap_yaw(yaw_front + 180), 0),
                    "left": (wrap_yaw(yaw_front - 90), 0),
                    "right": (wrap_yaw(yaw_front + 90), 0),
                    "top": (wrap_yaw(yaw_front), 90),
                    "bottom": (wrap_yaw(yaw_front), -90)
                }

                cubemap_faces = []
                for face in self.face_order:
                    yaw, pitch = face_angles[face]
                    face_img = py360convert.e2p(img.copy(), self.fov, yaw, pitch, (self.face_size, self.face_size))
                    face_img = torch.tensor(face_img.transpose(2, 0, 1))
                    cubemap_faces.append(face_img)

                cubemap_tensor = torch.stack(cubemap_faces, dim=0)

                # Load prompts if available
                prompts_path = os.path.join(scene_path, "prompts.json")
                if os.path.exists(prompts_path):
                    try:
                        with open(prompts_path, 'r') as f:
                            prompt_data = json.load(f)
                        prompts = [prompt_data.get(face, "") for face in self.face_order]

                        # Rotate the middle captions based on yaw_front
                        prompts = rotate_middle_captions(prompts, yaw_front)

                    # Handle JSON decoding errors
                    except Exception as e:
                        print(f"[WARNING] Failed to load prompts from {prompts_path}: {e}. Using empty prompts.")
                        prompts = ["" for _ in self.face_order]

                else:
                    print("[WARNING] Prompts file not found for scene:", scene)
                    prompts = ["" for _ in self.face_order]

                # Load single scene caption if available
                prompts_single_path = os.path.join(scene_path, "prompts_single.json")
                single_caption = ""
                if os.path.exists(prompts_single_path):
                    try:
                        with open(prompts_single_path, 'r') as f:
                            single_prompt_data = json.load(f)
                        single_caption = single_prompt_data.get("caption", "")
                        # get rid of the first 4 words (usually "The panoramic image shows....")
                        single_caption = " ".join(single_caption.split(" ")[4:])
                    except Exception as e:
                        print(f"[WARNING] Failed to load single caption from {prompts_single_path}: {e}. Using empty caption.")
                        single_caption = ""

                return cubemap_tensor, prompts, img_path, img, single_caption

            except Exception as e:
                print(f"[WARNING] Failed to load image {img_path}: {e}. Removing it from dataset and trying another sample.")
                # Remove the problematic sample from the list
                del self.samples[idx]
                if len(self.samples) == 0:
                    raise RuntimeError("All samples have been removed due to errors.")
                attempts += 1
        raise RuntimeError("Exceeded maximum attempts to find a valid sample.")

def cubemap_collate_fn(batch):
    # Filter out invalid samples if any remain
    batch = [item for item in batch if item is not None and isinstance(item[0], torch.Tensor)]

    # If the batch is empty after filtering, return None
    if len(batch) == 0:
        return None

    cubemaps = torch.cat([item[0] for item in batch], dim=0) # [B*6, C, H, W]
    prompts = [prompt for item in batch for prompt in item[1]]
    filenames = [item[2] for item in batch]
    equirecs = [item[3] for item in batch]
    single_captions = [item[4] for item in batch]

    return cubemaps, prompts, filenames, equirecs, single_captions
