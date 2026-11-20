"""Sample trained Mobile-Sakurahdm weights with the HDM flow solver."""

import argparse

import torch
from PIL import Image


@torch.inference_mode()
def euler_flow(model, latents, context, *, steps, guidance_scale, negative=None, source=None, pos_map=None, aspect_ratio=None):
    dtype = latents.dtype
    latents = latents.float()
    # HDM predicts noise - data with t=1 at pure noise and t=0 at the image.
    times = torch.linspace(1, 0, steps + 1, device=latents.device)
    for t, next_t in zip(times[:-1], times[1:]):
        timestep = t.expand(latents.shape[0])
        model_input = latents.to(dtype)
        velocity = model.denoise(model_input, timestep, context, source_latents=source, pos_map=pos_map, aspect_ratio=aspect_ratio).float()
        if negative is not None:
            unconditional = model.denoise(
                model_input, timestep, negative, source_latents=source, pos_map=pos_map, aspect_ratio=aspect_ratio
            ).float()
            velocity = unconditional.lerp(velocity, guidance_scale)
        latents = latents + (next_t - t) * velocity
    return latents.to(dtype)


def to_image(decoded):
    pixels = decoded[0].float().add(1).mul(127.5).clamp(0, 255).round()
    return Image.fromarray(pixels.to(torch.uint8).permute(1, 2, 0).cpu().numpy())


@torch.inference_mode()
def sample(model, processor, prompt, *, config, source_image=None, negative_prompt=""):
    from .train import encode_prompts, image_tensor

    options = config["sample"]
    training = config["training"]
    base_height, base_width = training["target_size"]
    height = options.get("height", base_height)
    width = options.get("width", base_width)
    device = model.dit.device
    generator = torch.Generator(device=device).manual_seed(options.get("seed", training["seed"]))

    images = None if source_image is None else [source_image]
    prompt_options = {
        "chat_kwargs": training.get("chat_kwargs"),
        "tokenizer_kwargs": training.get("tokenizer_kwargs"),
    }

    native_geometry = "sakuramoon" in config.get("data", {})

    def condition(text):
        inputs = encode_prompts(processor, [text], images, **prompt_options)
        return model.condition(inputs.to(device))

    context = condition(prompt)
    guidance_scale = options["guidance_scale"]
    negative = condition(negative_prompt) if guidance_scale != 1 else None
    source = None
    if images is not None:
        source = model.encode_images(image_tensor(images, (height, width), device), generator=generator)
    scale = model.latent_scale_factor
    latent_shape = (1, model.dit.config.input_dim, height // scale, width // scale)
    latents = torch.randn(latent_shape, device=device, generator=generator)
    pos_map = aspect_ratio = None
    if native_geometry:
        from xut.modules.axial_rope import make_axial_pos_no_cache
        pos_map = make_axial_pos_no_cache(*latents.shape[-2:], device=device)[None]
        aspect_ratio = latents.new_tensor([width / height])
    latents = euler_flow(
        model, latents, context, steps=options["steps"], guidance_scale=guidance_scale,
        negative=negative, source=source, pos_map=pos_map, aspect_ratio=aspect_ratio,
    )
    return to_image(model.decode_latents(latents))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--checkpoint", required=True, help="Path to trainable.safetensors")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--source-image")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from safetensors.torch import load_file

    from .models import build_model, build_processor
    from .train import configure_model, read_config

    config = read_config(args.configs)
    processor = build_processor(config["model"])
    model = build_model(config["model"])
    configure_model(model, config["training"])
    model.load_state_dict(load_file(args.checkpoint), strict=False)
    device = config["sample"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    source_image = None
    if args.source_image:
        with Image.open(args.source_image) as image:
            source_image = image.convert("RGB")
    sample(
        model, processor, args.prompt, config=config, source_image=source_image,
        negative_prompt=args.negative_prompt,
    ).save(args.output)


if __name__ == "__main__":
    main()
