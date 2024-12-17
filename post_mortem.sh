# visualize attention map difference
python3 post_mortem.py \
	--attn_diff \
	--diff_layer="./log/attn_maps/smartcontrol-A photo of tiger-depth-deer-control-1.0-alpha-diff-A photo of tiger vs A photo of deer-threshold 0.2-seed-12345/999/ControlNetModel.down_blocks.1.attentions.1.transformer_blocks.0.attn2/batch-0" \
	--gen_phrase="A photo of tiger" \
	--cond_phrase="A photo of deer"
