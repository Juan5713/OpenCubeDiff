import torch.nn as nn
import torch
import math

def calculate_positional_encoding(resolution=(128, 128), fov_deg=95.0):
    """
    Computes (u,v) positional encodings per CubeDiff Eq.(1) for all six cubemap faces
    using unit cube formulation and consistent global normalization.
    """
    extent = math.tan(math.radians(fov_deg / 2))  # Extent of cube face per FOV
    faces = ["front", "back", "left", "right", "top", "bottom"]
    encodings = {}

    for face in faces:
        u_range = torch.linspace(-extent, extent, resolution[0])
        v_range = torch.linspace(extent, -extent, resolution[1])
        u_grid, v_grid = torch.meshgrid(u_range, v_range, indexing='xy')

        if face == "front":
            x, y, z = u_grid, v_grid, torch.ones_like(u_grid)
        elif face == "back":
            x, y, z = -u_grid, v_grid, -torch.ones_like(u_grid)
        elif face == "left":
            x, y, z = -torch.ones_like(u_grid), v_grid, u_grid
        elif face == "right":
            x, y, z = torch.ones_like(u_grid), v_grid, -u_grid
        elif face == "top":
            x, y, z = u_grid, torch.ones_like(u_grid), -v_grid
        elif face == "bottom":
            x, y, z = u_grid, -torch.ones_like(u_grid), v_grid

        # Positional encoding via CubeDiff Equation (1)
        u_enc = torch.atan2(x, z)
        v_enc = torch.atan2(y, torch.sqrt(x ** 2 + z ** 2))

        # Normalize to [0, 1] using global angle range
        u_enc = (u_enc / math.pi + 1.0) / 2.0
        v_enc = (v_enc / math.pi + 1.0) / 2.0

        encodings[face] = torch.stack([u_enc, v_enc], dim=0)  # Shape: (2, H, W)

    return encodings

def mask_tensors(batch_size, latent_height, latent_width, num_faces=6):
    mask = torch.zeros((batch_size * num_faces, 1, latent_height, latent_width), dtype=torch.float16) # Shape: (B*T, 1, H, W)
    front_indices = torch.arange(0, batch_size * num_faces, num_faces) # Front face indices (0, 6, 12, ...)
    mask[front_indices] = 1.0
    return mask

def encoding_tensors(batch_size, latent_height, latent_width, face_order=None, encodings=None):
    """
    Generate stacked u_enc and v_enc channels for each face in a CubeDiff batch.
    Returns tensor of shape (B*T, 2, H, W)
    """
    if face_order is None:
        face_order = ["front", "back", "left", "right", "top", "bottom"]

    if encodings is None:
        encodings = calculate_positional_encoding((latent_height, latent_width))

    per_face_tensor = torch.stack([encodings[face] for face in face_order], dim=0)  # (T, 2, H, W)
    expanded_tensor = per_face_tensor.repeat(batch_size, 1, 1, 1, 1)  # (B, T, 2, H, W)
    stacked = expanded_tensor.reshape(batch_size * len(face_order), 2, latent_height, latent_width).to(dtype=torch.float16)  # (B*T, 2, H, W)
    return stacked

def make_extra_channels_tensor(batch_size, latent_height, latent_width, face_order=None, encodings=None):
    """
    Combine encoding tensors and mask tensors into a single (B*T, 3, H, W) tensor.
    Channel 0-1: u_enc, v_enc, Channel 2: binary mask
    """
    enc_tensor = encoding_tensors(batch_size, latent_height, latent_width, face_order, encodings)  # (B*T, 2, H, W)
    mask_tensor = mask_tensors(batch_size, latent_height, latent_width)  # (B*T, 1, H, W)
    return torch.cat([enc_tensor, mask_tensor], dim=1).to(dtype=torch.float16)  # (B*T, 3, H, W)
