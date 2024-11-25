from typing import Tuple, TypedDict

from PIL import Image, ImageDraw


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
