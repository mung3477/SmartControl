import copy
from typing import Dict, List

import torch


def _aggregate(attns: Dict[str, torch.Tensor], focus_indexes: List[int]) -> Dict[str, torch.Tensor]:
	agg = dict()

	for block, v in attns.items():
		multihead_avg = v.sum(1) / v.shape[1] # batch attn_head h w attn_dim -> batch h w attn_dim
		multihead_avg = multihead_avg.permute(0, 3, 1, 2)	# batch h w attn_dim -> batch attn_dim h w
		multihead_avg = multihead_avg.chunk(2)[1]
		focused = multihead_avg[:, focus_indexes, :, :].mean(dim=1)
		block_name = block.split(".")[1]

		try:
			block_num = int(block.split(".")[2])
		# if there is no block number (single block case)
		except ValueError:
			block_num = 0

		if block_name not in agg:
			agg[block_name] = dict()

		if block_num not in agg[block_name]:
			agg[block_name][block_num] = focused
		else:
			agg[block_name][block_num] = torch.cat((agg[block_name][block_num], focused))

	return agg

def _average(agg_res: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
	agg = copy.deepcopy(agg_res)

	for block, CAs in agg_res.items():
		for block_num, attn in CAs.items():
			assert isinstance(attn, torch.Tensor), "aggregation should return a dictionary who has a tensor as value"
			agg[block][block_num] = attn.sum(dim=0)

	return agg

def agg_by_blocks(attns: Dict[str, torch.Tensor], focus_indexes: List[int]):
	agg = _aggregate(attns, focus_indexes)

	return _average(agg)


