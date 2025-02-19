import torch
import torch.nn as nn

from lib import AlphaOptions, register_alpha_map_hook
from smart_unet import (ca_forward, crossattnupblock2d_forward,
                        upblock2d_forward)


def load_smartcontrol(unet,smart_ckpt):
    c_13_list = [1280,2560,2560,2560,2560,2560,2560,2560,1280,1280,1280,640,640]
    c_skip_list = [1280,2560,2560,2560,2560,2560,1280,1280,1280,640,640,640,640]
    c_pre_list = nn.ModuleList()
    count = 0
    for c in c_13_list:
        c_skip = c_skip_list[count]
        layers=[]
        if count==0:
            c_init = c+c_skip+c_skip
        else:
            c_init = c+c_skip+c_skip//2

        c_block = nn.Sequential(
                    nn.Conv2d(c_init, c_init//4, 3, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(c_init//4, c_init//8, 3, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(c_init//8, 1, 3, padding=1))

        count = count+1

        c_pre_list.append(c_block)

    state_dict={}
    pretrained_weights = torch.load(smart_ckpt, map_location="cpu")

    for k, v in pretrained_weights.items():
        name = k[11:]
        state_dict[name] = v
    unet.c_pre_list = c_pre_list.to(dtype=torch.float16)
    unet.c_pre_list.load_state_dict(state_dict)

def register_forward_hooks(model: torch.nn.Module, alpha_masks: dict):
    target_module_names = ['UNetMidBlock2DCrossAttn', 'CrossAttnUpBlock2D', 'UpBlock2D']
    for name, module in model.named_modules():
        if module.__class__.__name__ in target_module_names:
            register_alpha_map_hook(module, module_name=name, store_loc=alpha_masks)

def replace_call_methods(module: torch.nn.Module):
    for name, subnet in module.named_children():
        if subnet.__class__.__name__ == 'CrossAttnUpBlock2D':
            subnet.forward = crossattnupblock2d_forward(subnet)
        if subnet.__class__.__name__ == 'UpBlock2D':
            subnet.forward = upblock2d_forward(subnet)
        elif hasattr(subnet, 'children'):
            replace_call_methods(subnet)

def register_unet(pipe, smart_ckpt, mask_options: AlphaOptions, reset_masks=True):
    if smart_ckpt is not None:
        load_smartcontrol(pipe.unet, smart_ckpt)

    if reset_masks:
        pipe.unet.alpha_masks = dict()
    pipe.unet.forward = ca_forward(pipe.unet.cuda(), mask_options=mask_options)
    register_forward_hooks(pipe.unet, pipe.unet.alpha_masks)
    replace_call_methods(pipe.unet)
