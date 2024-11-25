from typing import List, Optional, TypedDict

import numpy as np
import torch


class AlphaOptions(TypedDict):
    alpha_mask: List[int]
    fixed: bool

def split_dim(dim_len: int, split_num: int):
	sections = [dim_len // split_num] * split_num

	if dim_len % split_num != 0:
		last_dim = sections.pop()
		sections.append(last_dim + dim_len % split_num)

	return sections

def generate_mask(alpha_mask: List[int], shape: torch.Size, device: torch.device):
	if len(alpha_mask) == 1:
		mask = torch.zeros(shape, dtype=torch.float16).to(device)
		mask.fill_(alpha_mask[0])
		return mask

	split_num = int(np.sqrt(len(alpha_mask)))
	assert split_num ** 2 == len(alpha_mask), f"Alpha mask should be square, but you gave {len(alpha_mask)} alphas."
	assert split_num < shape[0] and split_num < shape[1], f"Each side of alpha mask should have larger dimension than {split_num} since you gave {len(alpha_mask)} alphas and inferred mask is {str(shape)}."

	sec_row, sec_col = tuple(map(lambda dim: split_dim(dim, split_num), shape))

	masks = []
	for i, r in enumerate(sec_row):
		mask_rows = []
		for j, c in enumerate(sec_col):
			idx = i * len(sec_col) + j
			section = torch.zeros((r, c), dtype=torch.float16)
			section.fill_(alpha_mask[idx])
			mask_rows.append(section)
		mask_row = torch.concat(mask_rows, dim=1)
		masks.append(mask_row)

	return torch.concat(masks).to(device=device)
