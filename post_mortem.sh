# visualize attention map difference
python3 post_mortem.py \
	--attn_diff \
	--diff_layer="./log/attn_maps/smartcontrol-A photo of tiger-depth-deer-control-1.0-alpha-diff-tiger vs deer-no <sot> <eot>-threshold 0.0-seed-12345/999/ControlNetModel.mid_block.attentions.0.transformer_blocks.0.attn2/batch-0" \
	--gen_phrase="tiger" \
	--cond_phrase="deer" \
	--ignore_special_tkns
