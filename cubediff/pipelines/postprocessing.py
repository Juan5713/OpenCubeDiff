import numpy as np
import py360convert
from math import tan, radians
import torch

def crop_image_by_fov(image: np.ndarray, input_fov_deg: float, crop_fov_deg: float) -> np.ndarray:
    """
    Crop the center of a square image to a lower field-of-view (FOV).

    Parameters:
    - image: np.ndarray, the input square image (H x W x channels or H x W). 
             Must satisfy H == W.
    - input_fov_deg: float, the full field-of-view (in degrees) corresponding to the input image.
    - crop_fov_deg: float, the desired (lower) field-of-view (in degrees) for the cropped image.
                    Must be lower than input_fov_deg.

    Returns:
    - np.ndarray: the cropped image.
    
    Raises:
    - ValueError: if the image is not square or if crop_fov_deg is not lower than input_fov_deg.
    """
    # Ensure the image is square
    H, W = image.shape[:2]
    if H != W:
        raise ValueError("Input image must be square (height must equal width).")
    
    # Validate that crop_fov_deg is indeed less than input_fov_deg
    if crop_fov_deg >= input_fov_deg:
        raise ValueError("The crop FOV must be lower than the input FOV.")
    
    new_W_float = W * (tan(radians(crop_fov_deg) / 2) / tan(radians(input_fov_deg) / 2))
    new_W = int(round(new_W_float))
    
    # Compute the start and end coordinates for cropping (center crop)
    start = (W - new_W) // 2
    end = start + new_W

    # Crop the image
    cropped_image = image[start:end, start:end]
    return cropped_image


def stitch_to_cubemap(cubemap_faces: np.ndarray) -> np.ndarray:
    """
    Stitch the 6 faces of a cubemap into a single equirectangular image.

    Parameters:
    - cubemap_faces: np.ndarray, the 6 faces of the cubemap in the following order:
                     [front, back, left, right, top, bottom].
                     Each face is a square image (H x W x channels or H x W).
    Returns:
    - np.ndarray: the stitched equirectangular image.
    """

    # Cubemap_faces will be passed in order [front, back, left, right, top, bottom]
    
    cube_dict = {
        "F": cubemap_faces[0],
        "B": cubemap_faces[1],
        "L": cubemap_faces[2],
        "R": cubemap_faces[3],
        "U": cubemap_faces[4],
        "D": cubemap_faces[5]
    }

    return py360convert.c2e(cube_dict, h=1024, w=2048, cube_format='dict')
    

def postprocess_outputs(cubemap_tensor: torch.Tensor):
    """
    Postprocess the predicted cubemap tensor to an equirectangular image.

    Parameters:
    - cubemap_tensor: torch.Tensor, the predicted cubemap tensor of shape (T, C, H, W).
    Returns:
    - equirectangular_image: np.ndarray, the equirectangular image.
    - uncropped_faces: np.ndarray, the uncropped cubemap faces. (95 FOV)
    - cropped_faces: np.ndarray, the cropped cubemap faces. (90 FOV)
    """

    cubemap_list = []
    for i in range(cubemap_tensor.shape[0]):
        img = cubemap_tensor[i].detach().cpu().permute(1, 2, 0).float().numpy()
        cubemap_list.append(img)

    # Convert to nparray
    cubemap_np = np.array(cubemap_list)

    # Crop each face to 90 degrees FOV
    cropped_faces = np.array([crop_image_by_fov(face, 95, 90) for face in cubemap_np])

    # Stitch to equirectangular
    equirectangular_image = stitch_to_cubemap(cropped_faces)

    # Final clamping + convert to uint8 for display/saving
    equirectangular_image = (np.clip(equirectangular_image, 0, 1) * 255).astype(np.uint8)
    cropped_faces = (np.clip(cropped_faces, 0, 1) * 255).astype(np.uint8)
    uncropped_faces = (np.clip(cubemap_np, 0, 1) * 255).astype(np.uint8)

    return equirectangular_image, uncropped_faces, cropped_faces
