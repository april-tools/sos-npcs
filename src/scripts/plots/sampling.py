import argparse
import os

import torch
from PIL import Image as pillow
from torchvision.utils import make_grid

from sampling import inverse_transform_sample
from scripts.logger import Logger
from scripts.utils import set_global_seed, setup_data_loaders, setup_model

parser = argparse.ArgumentParser(
    description="Plot image samples",
)
parser.add_argument("checkpoint_path", type=str, help="The checkpoints path")
parser.add_argument("dataset", type=str, help="Dataset name")
parser.add_argument("model", type=str, help="Model name")
parser.add_argument("checkpoint_id", type=str, help="Checkpoint string id")
parser.add_argument("trial_id", type=str, help="Trial string id")
parser.add_argument("--exp-alias", type=str, default="", help="The experiment run alias")
parser.add_argument(
    "--num-units", type=int, default=16, help="The number of units per layer"
)
parser.add_argument(
    "--complex",
    action="store_true",
    default=False,
    help="Whether to use complex parameters in SOS and ExpSOS PCs",
)
parser.add_argument("--device", default="cuda", type=str, help="The device to use")


if __name__ == "__main__":
    args = parser.parse_args()
    seed = 42
    set_global_seed(seed)
    logger = Logger("evaluate-ll", verbose=True)
    metadata, (_, _, test_dataloader) = setup_data_loaders(
        "miniboone", "datasets", logger
    )
    (num_channels, image_height, image_width) = metadata["image_shape"]
    if num_channels != 1:
        raise NotImplementedError()

    model = setup_model(
        args.model,
        metadata,
        logger=logger,
        region_graph="qt",
        structured_decomposable=True,
        num_units=args.num_units,
        num_input_units=args.num_units,
        mono_clamp=True if args.model in ["MPC", "ExpSOS"] else False,
        complex=True if args.complex else False,
        seed=seed,
    )

    device = torch.device(args.device)
    model = model.to(device)
    checkpoint_filepath = os.path.join(
        args.checkpoint_path, args.dataset, args.model, args.exp_alias,
        args.checkpoint_id, f"checkpoint-{args.trial_id}.pt"
    )
    checkpoint = torch.load(checkpoint_filepath, map_location=device)
    model.load_state_dict(checkpoint["weights"])

    grid_height, grid_width = 3, 3
    num_samples = grid_height * grid_width
    samples = []
    for i in range(num_samples):
        sample = inverse_transform_sample(
            model, vdomain=255, num_samples=1, device=device
        )
        sample = sample[0, 0]
        samples.append(sample)
    samples = (torch.stack(samples) / 255.0).view(
        -1,
        1,
    )
    grid = make_grid(samples, nrow=grid_width, padding=0)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()

    path = os.path.join("figures", "image-sampling")
    os.makedirs(path, exist_ok=True)
    filename = f"{args.dataset}-{args.model}-K{args.num_units}-samples.png"
    sampled_images = pillow.fromarray(ndarr.transpose([1, 2, 0]))
    sampled_images.save(os.path.join(path, filename), format="png")
