from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


LABEL_PATTERN = re.compile(r"label_(\d+)", re.IGNORECASE)
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}


class ImageNetFilenameSubset(Dataset):
    def __init__(self, root: str | Path, image_size: int = 224) -> None:
        self.root = Path(root)
        self.paths = sorted(
            path
            for path in self.root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and LABEL_PATTERN.search(path.stem)
        )
        if not self.paths:
            raise FileNotFoundError(f"No labeled ImageNet subset images found in {self.root}")
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        match = LABEL_PATTERN.search(path.stem)
        if match is None:
            raise ValueError(f"Could not parse ImageNet label from {path.name}")
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return {
            "image": tensor,
            "label": int(match.group(1)),
            "sample_id": path.stem,
            "dataset_index": index,
        }


def build_imagenet1k_subset(root: str | Path, image_size: int = 224) -> Dataset:
    return ImageNetFilenameSubset(root=root, image_size=image_size)
