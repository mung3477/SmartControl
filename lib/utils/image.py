import os
from typing import Tuple, TypedDict

from PIL import Image, ImageChops, ImageDraw


class TextDraw_Options(TypedDict):
    fill: Tuple

def image_grid(imgs, rows, cols, caption, options: TextDraw_Options = { "fill" : (0, 0, 0)}):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))

    # https://www.geeksforgeeks.org/adding-text-on-image-using-python-pil/
    I1 = ImageDraw.Draw(grid)
    I1.text((10, 10), caption, fill=options['fill'])
    return grid

def calc_diff(img1_path: str, img2_path: str, name: str):
    assert os.path.exists(img1_path), f"{img1_path} not exists"
    assert os.path.exists(img2_path), f"{img2_path} not exists"

    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    assert img1.size == img2.size, f"{img1_path}: {img1.size} does not match with {img2_path}: {img2.size}"

    diff = ImageChops.subtract(img1, img2)
    grid = image_grid(
        [img1.resize((256, 256)), img2.resize((256, 256)), diff.resize((256, 256))],
        rows=1,
        cols=3,
        caption=f"diff btw {img1_path} and {img2_path}",
        options={ "fill": (255, 255, 255) }
    )
    grid.save(f"{name}.png")
    print(f"attention difference were saved at {name}.png")
