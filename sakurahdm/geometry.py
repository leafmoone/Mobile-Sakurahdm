"""Crop images and carry their original camera coordinates into latent space."""
import random

import torch
from PIL import Image


def shifted_square_crop(images, target_size, latent_scale, *, source_images=None, center=False, crop_seeds=None):
    from xut.modules.axial_rope import make_cropped_pos, make_axial_pos_no_cache

    h, w = target_size
    targets, sources, positions, aspects = [], [], [], []
    for i, image in enumerate(images):
        scale = max(w/image.width, h/image.height)
        rw, rh = max(w, int(image.width*scale)), max(h, int(image.height*scale))
        seed = crop_seeds[i] if crop_seeds is not None else None
        rng = random if seed is None else random.Random(seed)
        left = (rw-w)//2 if center else rng.randrange(rw-w) if rw>w else 0
        top = (rh-h)//2 if center else rng.randrange(rh-h) if rh>h else 0
        box = (left, top, left+w, top+h)
        targets.append(image.convert("RGB").resize((rw, rh), Image.Resampling.BICUBIC).crop(box))
        if source_images is not None:
            sources.append(source_images[i].convert("RGB").resize((rw, rh), Image.Resampling.BICUBIC).crop(box))
        if h == w:
            pos = make_cropped_pos(top//latent_scale, left//latent_scale, rh//latent_scale, rw//latent_scale)
        else:
            pos = make_axial_pos_no_cache(rh//latent_scale, rw//latent_scale).reshape(rh//latent_scale,rw//latent_scale,2)
            pos = pos[top//latent_scale:top//latent_scale+h//latent_scale,
                      left//latent_scale:left//latent_scale+w//latent_scale].flatten(0,1)
        positions.append(pos)
        aspects.append(image.width/image.height)
    return targets, sources if source_images is not None else None, torch.stack(positions), torch.tensor(aspects)


def bucketed_crop(images, assignments, latent_scale, *, source_images=None, crop_seeds=None):
    from sakuramoon.data.image_ops import resize_and_crop
    from xut.modules.axial_rope import make_axial_pos

    targets, sources, positions, aspects = [], [], [], []
    for index, (image, assignment) in enumerate(zip(images, assignments)):
        seed = random.getrandbits(64) if crop_seeds is None else crop_seeds[index]
        cropped = resize_and_crop(image, assignment, crop_seed=seed)
        targets.append(cropped.image)
        if source_images is not None:
            size = assignment.resized_width, assignment.resized_height
            sources.append(
                source_images[index].resize(size, Image.Resampling.LANCZOS).crop(cropped.crop_box)
            )
        h, w = assignment.bucket.height, assignment.bucket.width
        positions.append(make_axial_pos(h // latent_scale, w // latent_scale))
        aspects.append(assignment.source_width / assignment.source_height)
    return (
        targets, sources if source_images is not None else None,
        torch.stack(positions), torch.tensor(aspects),
    )
