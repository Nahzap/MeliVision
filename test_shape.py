import torch
m = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg', pretrained=False)
x = torch.randn(1, 3, 224, 224)
def hook(module, i, o):
    print('norm1 input shape:', i[0].shape)
m.blocks[-1].norm1.register_forward_hook(hook)
m.get_intermediate_layers(x, n=[3, 11], reshape=False, norm=False)
