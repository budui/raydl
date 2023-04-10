import sys
from pathlib import Path
from typing import Optional, Sequence, Union

import fire
import torch
from torchvision.datasets.folder import is_image_file

import raydl


def diff(path1, path2, save_path=None, adaptive=False):
    path1 = Path(path1)
    path2 = Path(path2)
    if path1.is_file() and path2.is_file():
        save_path = Path(save_path) if save_path is not None else path1.parent
        images_iterator = [(path1, path2, save_path / f"{path1.stem}_diff_{path2.stem}{path1.suffix}")]
    elif path1.is_dir() and path2.is_dir():
        save_path = Path(save_path) if save_path is not None else path1.parent / f"{path1.name}_diff_{path2.name}"
        if not save_path.exists():
            print(f"mkdir {save_path}")
            save_path.mkdir()
        files1 = [f for f in Path(path1).glob("*") if is_image_file(f.name) and (path2 / f.name).exists()]
        files2 = [path2 / f.name for f in files1]
        images_iterator = zip(files1, files2, [save_path / f.name for f in files1])
    else:
        raise ValueError(f"{path1} and {path2} are not both files or both folders")
    for image_path1, image_path2, path in images_iterator:
        image1 = raydl.load_images(image_path1, value_range=(0, 1))
        image2 = raydl.load_images(image_path2, value_range=(0, 1))
        assert image1.size() == image2.size()
        range_max = (1**1 * 3) ** 0.5 if not adaptive else None
        image_diff = torch.norm(image1 - image2, p=2, dim=1)
        print(
            f"{image_path1} and {image_path2} "
            f"max difference is {image_diff.max():.4f} ({100 * image_diff.max() / ((1 ** 1 * 3) ** 0.5):.2f}%)"
        )
        diff_heatmap = raydl.create_heatmap(image_diff, scale_each=False, range_min=0, range_max=range_max)
        if path.exists():
            _path = path.parent / f"{path.stem}_diff{path.suffix}"
            print(f"{path} existed! rename save path to {_path}")
            path = _path
        raydl.save_images(diff_heatmap, path)


def _infer_save_path(save_path, images_to_compare, batch_id, batch_size):
    if batch_size == 1:
        if is_image_file(save_path.name):
            save_path.parent.mkdir(exist_ok=True)
            return save_path.parent / images_to_compare[batch_id]
        save_path.mkdir(exist_ok=True)
        return save_path / images_to_compare[batch_id]

    if batch_size != len(images_to_compare):
        save_path.parent.mkdir(exist_ok=True)
        return save_path.parent / f"{save_path.stem}_{batch_id}{save_path.suffix}"

    return save_path


def concat(
    *image_folders,
    save_path: Union[Path, str],
    root: Optional[str] = None,
    captions: Union[bool, None, Sequence[str]] = None,
    resize: Union[int, tuple[int, int], None] = None,
    nrow: Optional[int] = None,
    transpose: bool = True,
    batch_size: Optional[int] = 1,
):
    image_folders = [Path(f) for f in image_folders]
    if root is not None:
        image_folders.extend([f for f in Path(root).iterdir() if f.is_dir()])

    images_to_compare = list(
        set.intersection(*[{p.name for p in folder.iterdir() if is_image_file(p.name)} for folder in image_folders])
    )
    if len(images_to_compare) == 0:
        print("can not found the same name images in these image_folders")
        sys.exit(1)

    if isinstance(captions, bool):
        captions = [f.name for f in image_folders] if captions else None
    assert captions is None or len(captions) == len(image_folders)

    images_to_compare = sorted(images_to_compare)
    save_path = Path(save_path)

    batch_size = batch_size or 1
    i = 0

    while i < len(images_to_compare):
        batch = min(batch_size, len(images_to_compare) - i)
        batch_id = (i + batch_size - 1) // batch_size
        images = torch.cat(
            [
                raydl.load_images([folder / name for name in images_to_compare[i : i + batch]], resize=resize)
                for folder in image_folders
            ]
        )
        i += batch

        if captions is not None:
            _captions: Union[None, Sequence[Union[None, str, tuple[str, str]]]] = list(map(str, captions))
            assert len(_captions) == len(
                image_folders
            ), f"{_captions} length is {len(_captions)} v.s len(image_folders)={len(image_folders)}"
            if not transpose:
                _captions = sum([[c, *[None] * (batch - 1)] for c in captions], [])
            else:
                _captions.extend([None] * (len(_captions) * (batch - 1)))
        else:
            _captions = None
        default_nrow = batch

        if transpose:
            images = raydl.grid_transpose(images, batch)
            default_nrow = images.size(0) // default_nrow

        sp = _infer_save_path(save_path, images_to_compare, batch_id, batch_size)
        print(sp)
        raydl.save_images(
            images,
            save_path=sp,
            captions=_captions,
            resize=resize,
            nrow=nrow or default_nrow,
        )


if __name__ == "__main__":
    fire.Fire({"concat": concat})
