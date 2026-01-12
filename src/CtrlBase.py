import einops
import torch
import inspect
import torch as th
from typing import Optional, Any, Union, List, Dict
import torch.nn as nn
from diffusers import UNet2DConditionModel
from dataclasses import dataclass
from diffusers.utils import BaseOutput, logging
from src.Base import BaseTrainer, expand_hidden_states, log_txt_as_img
import torchvision
from .utils import (
    count_params,
    pl_on_train_tart,
    module_requires_grad,
    get_obj_from_str,
    instantiate_from_config
)
import numpy as np
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from einops import rearrange
import string
from diffusers.models.resnet import Upsample2D, Downsample2D


class ControlBase(BaseTrainer):

    def __init__(self, control_config, base_config):
        super().__init__(base_config)
        self.control_model = instantiate_from_config(control_config)
        self.control_scales = [1.0] * 13


    def apply_model(self, dict_input):
        t = dict_input["timestep"]
        zt = dict_input["latent"]
        c = dict_input["cond"]
        hint = dict_input["hint"]
        control = self.control_model(hint, zt, t, c)
        control = [c * scale for c, scale in zip(control, self.control_scales)]
        unet = self.unet
        noise_pred = unet(x=zt, timestep=t, encoder_hidden_states=c, control=control).sample
        return noise_pred

    @torch.no_grad()
    def sample_loop(
            self,
            latents: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            hint: torch.Tensor,
            timesteps: torch.Tensor,
            do_classifier_free_guidance: bool = True,
            guidance_scale: float = 2,
            return_intermediates: Optional[bool] = False,
            extra_step_kwargs: Optional[Dict] = {},  
            **kwargs,
    ):
        intermediates = []
        res = {}

        for i, t in enumerate(timesteps):
            latent_model_input = (
                torch.cat(
                    [latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.noise_scheduler.scale_model_input(
                latent_model_input, t
            ).to(dtype=self.unet.dtype)


            if do_classifier_free_guidance:
                hint_input = torch.cat([hint] * 2)
            else:
                hint_input = hint
            control_input = self.control_model(hint_input, latent_model_input, t, encoder_hidden_states)

            noise_pred = self.unet(
                x=latent_model_input, timestep=t, encoder_hidden_states=encoder_hidden_states, control=control_input
            ).sample

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                )

            scheduler_res = self.noise_scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            )
            latents = scheduler_res.prev_sample
            assert torch.isnan(latents).sum() == 0, print("scheduler_res")
            if return_intermediates:
                intermediates.append(scheduler_res.pred_original_sample)

        latents = 1 / self.NORMALIZER * latents  
        res["latents"] = latents
        if len(intermediates) != 0:
            intermediates = [1 / self.NORMALIZER * x for x in intermediates]
            res["intermediates"] = intermediates
        return res

    def convert_latent2image(self, latent, tocpu=True):
        image = self.vae.decode(latent).sample
        image = image.clamp(0, 1) 
        if tocpu:
            image = image.cpu()
        return image

    @torch.no_grad()
    def sample(
            self,
            batch: Dict[str, Union[torch.Tensor, List[str]]],
            guidance_scale: float = 2,
            num_sample_per_image: int = 1,
            num_inference_steps: int = 50,
            eta: float = 0.0,  
            generator: Optional[torch.Generator] = None,
            return_intermediates: Optional[bool] = False,
            **kwargs,
    ):
        do_classifier_free_guidance = guidance_scale > 1.0

        if self.config.cond_on_text_image:
            cond_texts = batch["texts"].to(self.device)
            uncond_texts = torch.zeros_like(batch["texts"]).to(self.device)
        else:
            cond_texts = batch["texts"]
            uncond_texts = [""] * len(cond_texts)
        c = self.get_text_conditioning(cond_texts)
        uc = self.get_text_conditioning(uncond_texts)

        hint = batch["hint"].to(self.device)

        B, _, H, W = batch["img"].shape
        image_latents = torch.randn((B, 4, H // 8, W // 8), device=self.device)

        if num_sample_per_image > 1:
            image_latents = expand_hidden_states(
                image_latents, num_sample_per_image
            )
            uc = expand_hidden_states(
                uc, num_sample_per_image
            )
            c = expand_hidden_states(
                c, num_sample_per_image
            )
            hint = expand_hidden_states(
                hint, num_sample_per_image
            )
        encoder_hidden_states = torch.cat([uc, c])

        self.noise_scheduler.set_timesteps(
            num_inference_steps, device=self.vae.device)
        timesteps = self.noise_scheduler.timesteps

        image_latents = image_latents * self.noise_scheduler.init_noise_sigma

        accepts_eta = "eta" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        accepts_generator = "generator" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator

        latent_results = self.sample_loop(
            image_latents,
            encoder_hidden_states,
            hint,
            timesteps,
            do_classifier_free_guidance,
            guidance_scale,
            return_intermediates=return_intermediates,
            extra_step_kwargs=extra_step_kwargs,
            **kwargs,
        )

        image_results = {}
        images = self.convert_latent2image(latent_results["latents"])
        image_results["images"] = images
        if return_intermediates:
            intermediate_images = [
                self.convert_latent2image(x) for x in latent_results["intermediates"]
            ]
            image_results["intermediate_images"] = intermediate_images
        return image_results


    @torch.no_grad()
    def log_images(self, batch, generation_kwargs, stage="train", cat_gt=False):
        image_results = dict()
        if (
                stage == "train" or stage == "validation" or stage == "valid"
        ):  
            num_sample_per_image = generation_kwargs.get(
                "num_sample_per_image", 1)

            sample_results = self.sample(batch, **generation_kwargs)

            _, _, h, w = batch["img"].shape
            torch_resize = torchvision.transforms.Resize([h, w])
            for i, caption in enumerate(batch["texts"]):
                target_image = batch["img"][i].cpu()  
                target_image = target_image.clamp(0., 1.)
                target_image = torch_resize(target_image)

                style_image = batch["hint"][i].cpu() 
                style_image = style_image.clamp(0., 1.)
                style_image = torch_resize(style_image)


                cond_image = log_txt_as_img((w, h), batch["texts"][i], self.config.font_path, size= w // 10)
                cond_image = cond_image.clamp(0., 1.)
                cond_image = torch_resize(cond_image)


                sample_res = sample_results["images"][
                             i * num_sample_per_image: (i + 1) * num_sample_per_image
                             ]
                if cat_gt:
                    image_results[f"{i}-{caption}"] = torch.cat(
                        [
                         target_image.unsqueeze(0),
                         style_image.unsqueeze(0),
                         cond_image.unsqueeze(0),
                         sample_res], dim=0
                    )
                else:
                    image_results[f"{i}-{caption}"] = sample_res

        return (image_results,)

    def configure_optimizers(self):
        lr = self.learning_rate
        params = [{"params": self.control_model.parameters()}]
        if not self.sd_locked:
            params.append({"params": self.unet.parameters()})
        else:
            for name, parameter in self.unet.named_parameters():
                parameter.requires_grad = False
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        non_linearity="swish",
        time_embedding_norm="default",
        kernel=None,
        output_scale_factor=1.0,
        use_in_shortcut=None,
        up=False,
        down=False,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            self.time_emb_proj = torch.nn.Linear(temb_channels, out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()

        self.upsample = self.downsample = None
        if self.up:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.upsample = lambda x: upsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.upsample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
            else:
                self.upsample = Upsample2D(in_channels, use_conv=False)
        elif self.down:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.downsample = lambda x: downsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.downsample = partial(F.avg_pool2d, kernel_size=2, stride=2)
            else:
                self.downsample = Downsample2D(in_channels, use_conv=False, padding=1, name="op")

        self.use_in_shortcut = self.in_channels != self.out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, input_tensor, temb):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


@dataclass
class UNet2DConditionOutput(BaseOutput):
    sample: torch.FloatTensor

class ControlUNetModel(UNet2DConditionModel):
    def forward(self, x, timestep=None, encoder_hidden_states=None, control=None, return_dict: bool = True,**kwargs):
        default_overall_up_factor = 2 ** self.num_upsamplers

        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in x.shape[-2:]):
            forward_upsample_size = True

        if self.config.center_input_sample:
            x = 2 * x - 1.0

        with torch.no_grad():
            timesteps = timestep
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=x.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(x.device)

            timesteps = timesteps.expand(x.shape[0])
            t_emb = self.time_proj(timesteps)
            t_emb = t_emb.to(dtype=self.dtype)
            emb = self.time_embedding(t_emb)
            x = self.conv_in(x)
            xs = (x,)
            for downsample_block in self.down_blocks:
                if hasattr(downsample_block, "attentions") and downsample_block.attentions is not None:
                    x, res_x = downsample_block(hidden_states=x, temb=emb, encoder_hidden_states=encoder_hidden_states)
                else:
                    x, res_x = downsample_block(hidden_states=x, temb=emb)
                xs += res_x

        x = self.mid_block(x, emb, encoder_hidden_states)
        x += control.pop()


        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_xs = xs[-len(upsample_block.resnets):]
            xs = xs[: -len(upsample_block.resnets)]

            add_res = ()
            if control is not None:
                for item in res_xs[::-1]:
                    temp = item + control.pop()
                    add_res += (temp,)
                add_res = add_res[::-1]
            else:
                add_res = res_xs


            if not is_final_block and forward_upsample_size:
                upsample_size = xs[-1].shape[2:]

            if hasattr(upsample_block, "attentions") and upsample_block.attentions is not None:
                x = upsample_block(
                    hidden_states=x,
                    temb=emb,
                    res_hidden_states_tuple=add_res,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                )
            else:
                x = upsample_block(
                    hidden_states=x, temb=emb, res_hidden_states_tuple=add_res, upsample_size=upsample_size
                )
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)

        if not return_dict:
            return (x,)

        return UNet2DConditionOutput(sample=x)


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """
    def __init__(self, img_size=[64, 256], patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class VisionTransformerEncoder(nn.Module):
    """
    Pretrain Vision Transformer backbone.
    """
    def __init__(self, img_size=[64, 256], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask):
        x = self.patch_embed(x)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        B, _, C = x.shape
        if mask == None:
            x_vis = x.reshape(B, -1, C)
        else:
            x_vis = x[~mask].reshape(B, -1, C) 

        for blk in self.blocks:
            x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, mask):
        x = self.forward_features(x, mask)
        x = self.head(x)
        return x


class StylePyramidNet(nn.Module):
    def __init__(
            self,
            image_size=[64, 256],
            patch_size=16,
            in_channels=3,
            embed_dim=768,
            model_channels=320,
            channel_mult=(1, 2, 4, 4),
            pyramid_sizes=[ [8, 32], [4, 16], [2, 8], [1, 4]],
            use_checkpoint=True,
            use_new_attention_order=False,
            use_scale_shift_norm=False,
            dims=2,
            dropout=0,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.model_channels = model_channels

        self.vit = VisionTransformerEncoder(img_size=image_size, patch_size=patch_size, in_chans=in_channels,
                                            embed_dim=embed_dim, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                                            qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                                            norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0,
                                            use_learnable_pos_emb=False, )

        self.stages = nn.ModuleList([self._make_stage(embed_dim, size)
                                     for i, size in enumerate(pyramid_sizes)])

        self.zero_convs = nn.ModuleList([
            self.make_zero_conv(embed_dim, model_channels * channel_mult[0]),
            self.make_zero_conv(embed_dim, model_channels * channel_mult[0]),
            self.make_zero_conv(embed_dim, model_channels * channel_mult[0]),
            self.make_zero_conv(embed_dim, model_channels * channel_mult[0]),
            self.make_zero_conv(embed_dim, model_channels * channel_mult[1]),
            self.make_zero_conv(embed_dim, model_channels * channel_mult[1]),
            self.make_zero_conv(embed_dim, model_channels * channel_mult[1]),
            self.make_zero_conv(embed_dim, model_channels * channel_mult[2]),
            self.make_zero_conv(embed_dim, model_channels * channel_mult[2]),
            self.make_zero_conv(embed_dim, model_channels * channel_mult[2]),
            self.make_zero_conv(embed_dim, model_channels * channel_mult[3]),
            self.make_zero_conv(embed_dim, model_channels * channel_mult[3]),
        ])

        self.middle_block = nn.ModuleList([
            ResnetBlock2D(in_channels=embed_dim, down=True),
            ResnetBlock2D(in_channels=embed_dim, down=True),
        ])
        self.middle_block_out = self.make_zero_conv(embed_dim, model_channels*channel_mult[-1])
    def make_zero_conv(self, in_channels, out_channels):
            return zero_module(nn.Conv2d(in_channels, out_channels, 1, padding=0))

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size[0], size[1]))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, hint, sample=None, timesteps=None, encoder_hidden_states=None):

        h = self.vit(hint, mask=None)
        h_in = rearrange(h, 'b (h w) c -> b c h w', h=self.image_size[0]//self.patch_size, w=self.image_size[1]//self.patch_size)

        outs = []
        for i, stage in enumerate(self.stages):
            h_out = stage(h_in)
            for zero_conv in self.zero_convs[3*i: 3*i+3]:
                outs.append(zero_conv(h_out))

        hmid = h_in
        for block in self.middle_block:
            hmid = block(input_tensor=hmid, temb=None)
        outs.append(self.middle_block_out(hmid))

        return outs


import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class LabelEncoder(nn.Module):
    def __init__(
        self,
        max_len,
        character_path,
        d_model=768,
        residual_dropout_rate=0.0,
        scale_embedding=True,
        ckpt_path=None,
    ):
        super(LabelEncoder, self).__init__()
        self.max_len = max_len
        assert character_path is not None, "character_path is none"
        self.character = []
        with open(character_path, 'rb') as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip('\n').strip('\r\n')
                self.character.append(line)
        self.character.append(' ')
        vocab = len(self.character) + 3

        # 1. Embedding
        self.embedding = Embeddings(
            d_model=d_model,
            vocab=vocab,
            padding_idx=len(self.character)+2,
            scale_embedding=scale_embedding,
        )
        # 2. Positional Encoding
        self.positional_encoding = PositionalEncoding(
            dropout=residual_dropout_rate, dim=d_model, max_len=self.max_len+2,
        )

        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)  

    def get_index(self, labels):
        indexes = []
        for label in labels:
            if len(label) > self.max_len:
                label = label[:self.max_len]
            index = [(self.character.index(c) + 1) if c in self.character else 0 for c in label]
            index = [len(self.character)+1] + index + [0] + [len(self.character)+2] * (self.max_len - len(index))
            indexes.append(index)
        return torch.tensor(indexes, device=next(self.parameters()).device)

    def forward(self, x):
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x = self.positional_encoding(x)

        return x

    def encode(self, labels):
        text = self.get_index(labels)
        return self(text)


class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the
    tokens in the sequence. The positional encodings have the same dimension as
    the embeddings, so that the two can be summed. Here, we use sine and cosine
    functions of different frequencies.

    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, dropout, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros([max_len, dim])
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = torch.unsqueeze(pe, 0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)  # .permute([1, 0, 2])


class Embeddings(nn.Module):

    def __init__(self, d_model, vocab, padding_idx=None, scale_embedding=True):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        self.embedding.weight.data.normal_(mean=0.0, std=d_model**-0.5)
        self.d_model = d_model
        self.scale_embedding = scale_embedding

    def forward(self, x):
        if self.scale_embedding:
            x = self.embedding(x)
            return x * math.sqrt(self.d_model)
        return self.embedding(x)