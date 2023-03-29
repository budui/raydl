import sys
from pathlib import Path
from typing import Optional, Sequence, Union

import fire
import torch
from torchvision.datasets.folder import is_image_file

import raydl


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
