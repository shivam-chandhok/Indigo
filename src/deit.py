import torch
import torch.nn as nn
from functools import partial
import numpy as np

from timm.models.vision_transformer import VisionTransformer, _cfg, Block
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer_hybrid import vit_tiny_r_s16_p8_224

# class HybridEmbed(nn.Module):
#     """ CNN Feature Map Embedding
#     Extract feature map from CNN, flatten, project to embedding dim.
#     """
#     def __init__(self, backbone, img_size=224, patch_size=1, feature_size=None, in_chans=3, embed_dim=768):
#         super().__init__()
#         assert isinstance(backbone, nn.Module)
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.backbone = backbone
#         if feature_size is None:
#             with torch.no_grad():
#                 # NOTE Most reliable way of determining output dims is to run forward pass
#                 training = backbone.training
#                 if training:
#                     backbone.eval()
#                 o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
#                 if isinstance(o, (list, tuple)):
#                     o = o[-1]  # last feature if backbone outputs list/tuple of features
#                 feature_size = o.shape[-2:]
#                 feature_dim = o.shape[1]
#                 backbone.train(training)
#         else:
#             feature_size = to_2tuple(feature_size)
#             if hasattr(self.backbone, 'feature_info'):
#                 feature_dim = self.backbone.feature_info.channels()[-1]
#             else:
#                 feature_dim = self.backbone.num_features
#         assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[1] == 0
#         self.grid_size = (feature_size[0] // patch_size[0], feature_size[1] // patch_size[1])
#         self.num_patches = self.grid_size[0] * self.grid_size[1]
#         self.proj = nn.Conv2d(feature_dim, embed_dim, kernel_size=patch_size, stride=patch_size)

#     def forward(self, x):
#         x = self.backbone(x)
#         if isinstance(x, (list, tuple)):
#             x = x[-1]  # last feature if backbone outputs list/tuple of features
#         x = self.proj(x).flatten(2).transpose(1, 2)
#         return x

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = (img_size,img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
#         _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
#         _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]


def normalize(x):
    mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).cuda()
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    
    return (x-mean)/std

class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        
        self.norm = partial(nn.LayerNorm, eps=1e-6)(self.embed_dim)

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
#         x = self.head(x)
#         x_dist = self.head_dist(x_dist)
        return x, x_dist
       
        
class MyDistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.fc = nn.Sequential( nn.Linear(1024,self.embed_dim),
                                nn.LayerNorm(self.embed_dim)
                               )
        
        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x, att):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)
        att = self.fc(att)
        att = att[:, None, :]
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
#         att = att.expand(B, -1, -1)

        x = torch.cat((cls_tokens, att, x), dim=1)
#         x = torch.cat((att, att, x), dim=1)
        x = x + self.pos_embed
#         x = torch.cat((x, att), dim=1)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x, att):
        x, x_dist = self.forward_features(x, att)
#         x = self.head(x)
#         x_dist = self.head_dist(x_dist)
        return x, x_dist

# class MyDistilledVisionTransformer(VisionTransformer):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
#         num_patches = self.patch_embed.num_patches
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
#         self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
#         self.conv = nn.Sequential( nn.ConvTranspose2d(2048,3, kernel_size=32, stride=32), nn.Sigmoid())
# #         self.patch_embed2 = PatchEmbed(img_size=7, patch_size=1, in_chans=2048, embed_dim=self.embed_dim, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        
#         trunc_normal_(self.dist_token, std=.02)
#         trunc_normal_(self.pos_embed, std=.02)
#         self.head_dist.apply(self._init_weights)

#     def forward_features(self, x, att):
#         # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
#         # with slight modifications to add the dist_token
#         B = x.shape[0]
#         x = self.patch_embed(x)
#         att = normalize(self.conv(att))
                    
# #         att = self.patch_embed2(nn.functional.interpolate(att, (14,14), mode='bicubic'))
#         att = self.patch_embed(att)
        
#         cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
#         dist_token = self.dist_token.expand(B, -1, -1)

# #         x = torch.cat((cls_tokens, dist_token, x), dim=1)
#         x = torch.cat((cls_tokens, dist_token, att), dim=1)
#         x = x + self.pos_embed
# #         x = torch.cat((x, att), dim=1)
#         x = self.pos_drop(x)

#         for blk in self.blocks:
#             x = blk(x)

#         x = self.norm(x)
#         return x[:, 0], x[:, 1]

#     def forward(self, x, att):
#         x, x_dist = self.forward_features(x, att)
# #         x = self.head(x)
# #         x_dist = self.head_dist(x_dist)
#         return x, x_dist


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
#         checkpoint = torch.load("/home/code-base/user_space/cumix/checkpoints/deit_tiny_patch16_224-a1311bcf.pth", map_location="cpu")
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load("/home/code-base/user_space/cumix/checkpoints/deit_tiny_distilled_patch16_224-b40b3cf7.pth", map_location="cpu")
#         checkpoint = torch.hub.load_state_dict_from_url(
#             url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
#             map_location="cpu", check_hash=True
#         )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def mydeit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    model = MyDistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load("/home/code-base/user_space/cumix/checkpoints/deit_tiny_distilled_patch16_224-b40b3cf7.pth", map_location="cpu")
#         checkpoint = torch.hub.load_state_dict_from_url(
#             url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
#             map_location="cpu", check_hash=True
#         )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model



@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
#         checkpoint = torch.hub.load_state_dict_from_url(
#             url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
#             map_location="cpu", check_hash=True
#         )
        checkpoint = torch.load("/home/code-base/user_space/cumix/checkpoints2/deit_small_distilled_patch16_224-649709d9.pth", map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def mydeit_small_distilled_patch16_224(pretrained=False, **kwargs):
    model = MyDistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
#         checkpoint = torch.hub.load_state_dict_from_url(
#             url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
#             map_location="cpu", check_hash=True
#         )
        checkpoint = torch.load("/home/code-base/user_space/cumix/checkpoints2/deit_small_distilled_patch16_224-649709d9.pth", map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    return model

@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model