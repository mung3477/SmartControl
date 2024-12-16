# visualize attention map difference
python3 post_mortem.py \
	--attn_diff \
	--diff_layer="./log/attn_maps/smartcontrol-High-heeled shoe encrusted with diamonds-canny-shoes-control-1.0-alpha-diff-High-heeled shoe diamonds vs Enamel boots-no <sot> <eot>-seed-12345/999/ControlNetModel.down_blocks.1.attentions.0.transformer_blocks.0.attn2/batch-0" \
	--gen_phrase="High-heeled shoe diamonds" \
	--cond_phrase="Enamel ankle boots with a strap"
