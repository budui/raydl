import math
import platform
import warnings
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union, cast

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid

DEFAULT_FONT = "DejaVuSans.ttf" if platform.system() == "Linux" else "Arial.ttf"


def resize_images(
    images: torch.Tensor, resize=None, resize_mode: str = "bilinear", smart_resize=False
) -> torch.Tensor:
    """
    resize images, when resize is not None.
    :param images: torch.Tensor[NxCxHxW]
    :param resize: None means do nothing, or target_size[int].
        target_size will be convert to (target_size, target_size)
    :param resize_mode: interpolate mode.
    :return: resized images.
    """
    if resize is None:
        return images
    if isinstance(resize, (int, float)):
        if not smart_resize:
            resize = (int(resize), int(resize * images.shape[-1] / images.shape[-2]))
        else:
            resize = (
                (int(resize), int(resize * images.shape[-1] / images.shape[-2]))
                if images.size(-2) > images.size(-1)
                else (int(resize * images.shape[-2] / images.shape[-1]), int(resize))
            )
    resize = (resize, resize) if isinstance(resize, int) else resize
    if resize[0] != images[0].size(-2) or resize[1] != images[0].size(-1):
        align_corners = False if resize_mode in ["linear", "bilinear", "bicubic", "trilinear"] else None
        return F.interpolate(images, size=resize, mode=resize_mode, align_corners=align_corners, antialias=True)
    return images


def pil_loader(path: Union[str, Path], mode="RGB") -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert(mode)


def load_images(
    images_path: Union[str, Path, Iterable],
    resize: Optional[Union[int, tuple]] = None,
    value_range: tuple[int, int] = (-1, 1),
    device: Union[torch.device, str] = "cpu",
    image_mode: str = "RGB",
    resize_mode: str = "bilinear",
) -> torch.Tensor:
    """
    read images into tensor
    :param images_path:
    :param resize:
    :param value_range:
    :param device:
    :param image_mode: accept a string to specify the mode of image, must in
     https://pillow.readthedocs.io/en/latest/handbook/concepts.html#modes
    :param resize_mode: accept a string to specify the mode of resize(interpolate), must in
     https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate
    :return: images (Tensor[num_images, image_channels, image_height, image_width])
    """
    if isinstance(images_path, (str, Path)):
        images_path = [
            images_path,
        ]
    images = []
    for image_path in images_path:
        pil_image = pil_loader(image_path, image_mode)
        # 1xCxHxW, value_range: [0, 1]
        image = torchvision.transforms.functional.to_tensor(pil_image).unsqueeze_(0)
        images.append(resize_images(image, resize=resize, resize_mode=resize_mode))
    assert len(images) > 0, f"{images_path=}, can not find enough images"
    images = torch.cat(images).to(device)
    images = images * (value_range[1] - value_range[0]) + value_range[0]
    return images


def captioning_pil_image(
    pil_image: Image.Image,
    captions: Sequence[Union[str, tuple[str, str], None]],
    grid_cell_size: tuple[int, int],
    grid_cell_padding: int,
    caption_color: str = "#ff0000",
    caption_font=DEFAULT_FONT,
) -> Image.Image:
    """
    draw captions over grid image. use grid_cell_size to specify minimal cell size in grid.
    :param pil_image: grid image
    :param captions: a sequence of value in (None, str, tuple).
     value can be None, which means skip this grid cell;
     can be str, which is captions;
     can be tuple of (caption, color), for specify color for this caption.
    :param grid_cell_size: tuple (height, width)
    :param grid_cell_padding: padding when make grid
    :param caption_color: the color of the captions, default is red "#ff0000"
    :param caption_font: the font of the captions, default is DejaVuSans.ttf,
        will find font as https://pillow.readthedocs.io/en/latest/reference/ImageFont.html#PIL.ImageFont.truetype
    :return:
    """
    h, w = grid_cell_size
    padding = grid_cell_padding
    nrow = pil_image.width // w
    im_draw = ImageDraw.Draw(pil_image)
    try:
        im_font = ImageFont.truetype(caption_font, size=max(h // 15, 12))
    except OSError:
        warnings.warn(f"can not find {caption_font}, so use the default font, better than nothing")
        im_font = ImageFont.load_default()
    for i, cap in enumerate(captions):
        if cap is None:
            continue
        if isinstance(cap, (tuple, list)):
            cap, fill_color = cap
        else:
            fill_color = caption_color
        im_draw.text(
            ((i % nrow) * (w + padding) - padding, (i // nrow) * (h + padding) - padding),
            cap,
            fill=fill_color,
            font=im_font,
        )
    return pil_image


def infer_pleasant_nrow(length: int):
    sqrt_nrow_candidate = int(math.sqrt(length))
    if sqrt_nrow_candidate**2 == length:
        return sqrt_nrow_candidate
    return 2 ** int(math.log2(math.sqrt(length)) + 1)


def to_pil_images(
    images: Union[torch.Tensor, list[torch.Tensor]],
    captions: Union[bool, None, Sequence[Union[None, str, tuple[str, str]]]] = None,
    resize: Optional[Union[int, tuple[int, int]]] = None,
    separately: bool = False,
    nrow: Optional[int] = None,
    normalize: bool = True,
    value_range: Optional[tuple[int, int]] = (-1, 1),
    scale_each: bool = False,
    padding: int = 0,
    pad_value: int = 0,
    caption_color: str = "#ff0000",
    caption_font=DEFAULT_FONT,
) -> Union[Image.Image, list[Image.Image]]:

    if isinstance(captions, bool):
        captions = list(map(str, range(len(images)))) if captions else None

    images = images if isinstance(images, torch.Tensor) else torch.cat(images)
    images = resize_images(images, resize=resize)
    if not separately:
        if nrow is None:
            nrow = infer_pleasant_nrow(len(images))
        grid_image = make_grid(images, nrow, padding, normalize, value_range, scale_each, pad_value)
        pil_image = Image.fromarray(
            grid_image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        )
        if captions is not None:
            pil_image = captioning_pil_image(
                pil_image,
                cast(Sequence[Union[str, tuple[str, str]]], captions),
                grid_cell_size=(images[0].size(-2), images[0].size(-1)),
                grid_cell_padding=padding,
                caption_color=caption_color,
                caption_font=caption_font,
            )
        return pil_image

    pil_images = []
    captions = [captions] * len(images) if isinstance(captions, str) else captions
    assert captions is None or len(captions) >= len(images)

    for i in range(len(images)):
        image = images[i : i + 1]
        caption = None if captions is None else captions[i : i + 1]
        image = make_grid(image, 1, 0, normalize, value_range, scale_each, 0)
        pil_image = Image.fromarray(
            image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        )
        if caption is not None:
            pil_image = captioning_pil_image(
                pil_image,
                cast(Union[str, tuple[str, str]], caption),
                grid_cell_size=(image.size(-2), image.size(-1)),
                grid_cell_padding=padding,
                caption_color=caption_color,
                caption_font=caption_font,
            )
        pil_images.append(pil_image)
    return pil_images


def save_images(
    images: Union[torch.Tensor, list[torch.Tensor]],
    save_path: Union[str, Path, Sequence[Union[str, Path]]],
    captions: Union[bool, None, Sequence[Union[None, str, tuple[str, str]]]] = None,
    resize: Optional[Union[int, tuple[int, int]]] = None,
    separately: bool = False,
    nrow: Optional[int] = None,
    normalize: bool = True,
    value_range: Optional[tuple[int, int]] = (-1, 1),
    scale_each: bool = False,
    padding: int = 0,
    pad_value: int = 0,
    caption_color: str = "#ff0000",
    caption_font=DEFAULT_FONT,
):
    """
    save images
    :param images: (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
    :param save_path: path to save images
    :param captions:
    :param resize: if not None, resize images.
    :param separately: if True, save images separately rather make grid
    :param nrow: Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``.
    :param normalize: If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
    :param value_range: tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image.
    :param scale_each: If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
    :param padding: amount of padding. Default: ``0``.
    :param pad_value:  Value for the padded pixels. Default: ``0``.
    :param caption_color: the color of the captions, default is red "#ff0000"
    :param caption_font: the font of the captions, default is DejaVuSans.ttf,
        will find font as https://pillow.readthedocs.io/en/latest/reference/ImageFont.html#PIL.ImageFont.truetype
    :return: None
    """
    assert not (isinstance(save_path, (list, tuple)) and not separately), f"{save_path} separately: {separately}"

    pil_images = to_pil_images(
        images,
        captions=captions,
        resize=resize,
        separately=separately,
        nrow=nrow,
        normalize=normalize,
        value_range=value_range,
        scale_each=scale_each,
        padding=padding,
        pad_value=pad_value,
        caption_color=caption_color,
        caption_font=caption_font,
    )

    if isinstance(pil_images, Image.Image):
        save_path = save_path if isinstance(save_path, (str, Path)) else save_path[0]
        pil_images.save(save_path)
        return

    if isinstance(save_path, (str, Path)):
        save_path = Path(save_path)
        save_path = [save_path.with_name(f"{save_path.stem}_{i}{save_path.suffix}") for i in range(len(images))]
    assert len(save_path) >= len(images)

    for i in range(len(pil_images)):
        pil_images[i].save(save_path[i])


def create_heatmap(
    images: torch.Tensor,
    range_min: Union[float, torch.Tensor, None] = None,
    range_max: Union[float, torch.Tensor, None] = None,
    scale_each: bool = False,
    color_map: str = "jet",
) -> torch.Tensor:
    """
    create heatmap from BxHxW tensor.
    :param images: Tensor[BxHxW]
    :param range_min: max value used to normalize the image. By default, min and max are computed from the tensor.
    :param range_max: min value used to normalize the image. By default, min and max are computed from the tensor.
    :param scale_each: If True, scale each image in the batch of images separately rather
     than the (min, max) over all images. Default: False.
    :param color_map: The colormap to apply, colormap from
     https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html#ga9a805d8262bcbe273f16be9ea2055a65
    :param return_tensor: if True, return Tensor[Bx3xHxW], otherwise return tuple of numpy.array(0-255)
    :return:
    """
    device = images.device
    assert images.dim() == 3
    try:
        import cv2

        color_map = getattr(cv2, f"COLORMAP_{color_map.upper()}")
    except AttributeError as e:
        raise ValueError(f"invalid color_map {color_map}") from e

    with torch.no_grad():
        images = images.detach().clone().to(dtype=torch.float32, device=torch.device("cpu"))
        if range_min is None:
            range_min = images.amin(dim=[-1, -2], keepdim=True) if scale_each else images.amin()
        if range_max is None:
            range_max = images.amax(dim=[-1, -2], keepdim=True) if scale_each else images.amax()
        heatmaps = []
        for m in images.add_(-range_min).div_(range_max - range_min + 1e-5).clip_(0.0, 1.0):
            heatmaps.append(cv2.applyColorMap(np.uint8(m.numpy() * 255), color_map))
        heatmaps = torch.from_numpy(np.stack(heatmaps)).permute(0, 3, 1, 2)
        # BGR -> RGB & [0, 255] -> [0, 1]
        heatmaps = heatmaps[:, [2, 1, 0], :, :].contiguous().float().to(device) / 255
        return heatmaps
