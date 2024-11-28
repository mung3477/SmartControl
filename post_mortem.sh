# visualize attention map difference
python3 post_mortem.py \
	--attn_diff \
	--diff_layer="./log/attn_maps/smartcontrol-a photo of tiger-depth-deer-control-1.0-alpha-[1]-multiplied-seed-12345/999/ControlNetModel.down_blocks.0.attentions.0.transformer_blocks.0.attn2/batch-0" \
	--attn1="4-<tiger>.png" \
	--attn2="sub-4-<deer>.png"
