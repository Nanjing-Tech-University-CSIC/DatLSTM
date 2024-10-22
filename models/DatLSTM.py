import torch
import torch.nn as nn
from einops import rearrange, einops
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
import numpy as np



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# 将特征图划分成窗口
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.
        patch_size (int): Patch token size.
        in_chans (int): Number of input image channels.
        embed_dim (int): Number of linear projection output channels.
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super(PatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        '''

        :param x:B,C,H,W
        :return: (B, num_patches, embed_dim) num_patches = H//p*W//p
        '''
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

# 将patch图还原成原始图片
class PatchInflated(nn.Module):
    r""" Tensor to Patch Inflating

    Args:
        in_chans (int): Number of input image channels.
        embed_dim (int): Number of linear projection output channels.
        input_resolution (tuple[int]): Input resulotion.
    """

    def __init__(self, in_chans, embed_dim, input_resolution, stride=2, padding=1, output_padding=1):
        super(PatchInflated, self).__init__()

        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        output_padding = to_2tuple(output_padding)
        self.input_resolution = input_resolution

        self.Conv = nn.ConvTranspose2d(in_channels=embed_dim, out_channels=in_chans, kernel_size=(3, 3),
                                       stride=stride, padding=padding, output_padding=output_padding)

    def forward(self, x):
        '''

        :param x: [B,L,C] L = pH*pW ,B patch的数量
        :return: tensor [B,C,H,W]
        '''
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        x = self.Conv(x)
        # x B C H W
        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchExpanding(nn.Module):
    r""" Patch Expanding Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super(PatchExpanding, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x

#  LN模块
class LayerNormProxy(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')

# Dat Attention
class DAttentionBaseline(nn.Module):

    def __init__(
            self, q_size, num_heads, n_head_channels, n_groups,
            attn_drop, proj_drop, stride,
            offset_range_factor, use_pe, dwc_pe,
            no_off, fixed_pe, ksize, log_cpb
    ):

        super().__init__()
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = num_heads
        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
        self.nc = n_head_channels * num_heads           # nc = n_heads * n_head_channels
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups # nc = groups*n_group_channels
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.ksize = ksize
        self.log_cpb = log_cpb
        self.stride = stride
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )
        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe and not self.no_off:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(
                    self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w)
                )
                trunc_normal_(self.rpe_table, std=0.01)
            elif self.log_cpb:

                self.rpe_table = nn.Sequential(
                    nn.Linear(2, 32, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, self.n_group_heads, bias=False)
                )
            else:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * 2 - 1, self.q_w * 2 - 1)
                )
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref

    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, x):
        # print("n_group",self.n_groups)
        # print('num_heads',self.n_heads)
        # print('n_group_channels',self.n_group_channels)
        # print("embed_dim",)
        # print("q",self.q_h)
        # print("up_sample",x.shape) 第一次upSample [8,512,8,8]
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off).contiguous()  # B * g 2 Hg Wg
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.no_off:
            offset = offset.fill_(0.0)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).clamp(-1., +1.)

        if self.no_off:
            x_sampled = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
            assert x_sampled.size(2) == Hk and x_sampled.size(3) == Wk, f"Size is {x_sampled.size()}"
        else:
            x_sampled = F.grid_sample(
                input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
                grid=pos[..., (1, 0)],  # y, x -> x, y
                mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)

        if self.use_pe and (not self.no_off):

            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_heads, self.n_head_channels,
                                                                              H * W)
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_heads, H * W, n_sample)
            elif self.log_cpb:
                q_grid = self._get_q_grid(H, W, B, dtype, device)
                displacement = (
                            q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups,
                                                                                                   n_sample,
                                                                                                   2).unsqueeze(1)).mul(
                    4.0)  # d_y, d_x [-8, +8]
                displacement = torch.sign(displacement) * torch.log2(torch.abs(displacement) + 1.0) / np.log2(8.0)
                attn_bias = self.rpe_table(displacement)  # B * g, H * W, n_sample, h_g
                attn = attn + einops.rearrange(attn_bias, 'b m n h -> (b h) m n', h=self.n_group_heads)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                q_grid = self._get_q_grid(H, W, B, dtype, device)
                displacement = (
                            q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups,
                                                                                                   n_sample,
                                                                                                   2).unsqueeze(1)).mul(
                    0.5)
                attn_bias = F.grid_sample(
                    input=einops.rearrange(rpe_bias, 'b (g c) h w -> (b g) c h w', c=self.n_group_heads,
                                           g=self.n_groups),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True)  # B * g, h_g, HW, Ns

                attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)
                attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)

        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe
        out = out.reshape(B, C, H, W)

        y = self.proj_drop(self.proj_out(out))

        return y, pos.reshape(B, self.n_groups, Hk, Wk, 2), reference.reshape(B, self.n_groups, Hk, Wk, 2)

# 局部窗口注意力模块


class LocalAttention(nn.Module):

    def __init__(self, dim, heads, window_size, attn_drop, proj_drop):
        super().__init__()

        window_size = to_2tuple(window_size)

        self.proj_qkv = nn.Linear(dim, 3 * dim)
        self.heads = heads
        assert dim % heads == 0
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.proj_out = nn.Linear(dim, dim)
        self.window_size = window_size
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        Wh, Ww = self.window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * Wh - 1) * (2 * Ww - 1), heads)
        )
        trunc_normal_(self.relative_position_bias_table, std=0.01)

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, mask=None):
        B, C, H, W = x.size()
        r1, r2 = H // self.window_size[0], W // self.window_size[1]

        x_total = einops.rearrange(x, 'b c (r1 h1) (r2 w1) -> b (r1 r2) (h1 w1) c', h1=self.window_size[0],
                                   w1=self.window_size[1])  # B x Nr x Ws x C

        x_total = einops.rearrange(x_total, 'b m n c -> (b m) n c')

        qkv = self.proj_qkv(x_total)  # B' x N x 3C
        q, k, v = torch.chunk(qkv, 3, dim=2)

        q = q * self.scale
        q, k, v = [einops.rearrange(t, 'b n (h c1) -> b h n c1', h=self.heads) for t in [q, k, v]]
        attn = torch.einsum('b h m c, b h n c -> b h m n', q, k)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn_bias = relative_position_bias
        attn = attn + attn_bias.unsqueeze(0)

        if mask is not None:
            # attn : (b * nW) h w w
            # mask : nW ww ww
            nW, ww, _ = mask.size()
            attn = einops.rearrange(attn, '(b n) h w1 w2 -> b n h w1 w2', n=nW, h=self.heads, w1=ww,
                                    w2=ww) + mask.reshape(1, nW, 1, ww, ww)
            attn = einops.rearrange(attn, 'b n h w1 w2 -> (b n) h w1 w2')
        attn = self.attn_drop(attn.softmax(dim=3))

        x = torch.einsum('b h m n, b h n c -> b h m c', attn, v)
        x = einops.rearrange(x, 'b h n c1 -> b n (h c1)')
        x = self.proj_drop(self.proj_out(x))  # B' x N x C
        x = einops.rearrange(x, '(b r1 r2) (h1 w1) c -> b c (r1 h1) (r2 w1)', r1=r1, r2=r2, h1=self.window_size[0],
                             w1=self.window_size[1])  # B x C x H x W

        return x, None, None


class DatTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

       Args:
           dim (int): Number of input channels.
           input_resolution (tuple[int]): Input resulotion.
           num_heads (int): Number of attention heads.
           window_size (int): Window size.
           mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
           qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
           qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
           drop (float, optional): Dropout rate. Default: 0.0
           attn_drop (float, optional): Attention dropout rate. Default: 0.0
           drop_path (float, optional): Stochastic depth rate. Default: 0.0
           act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
           norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm

       """
    def __init__(self, dim, num_heads, n_head_channels, n_groups,input_resolution,
                stride,offset_range_factor, use_pe, dwc_pe,
                 no_off, fixed_pe, ksize, log_cpb, window_size=2,
                 mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.,proj_drop=0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        q_size = to_2tuple(input_resolution)
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        # window attn
        self.window_attn = LocalAttention(
            dim, window_size=window_size, heads=num_heads,attn_drop=attn_drop, proj_drop=drop)
        # dat attn
        self.dat_attn = DAttentionBaseline(q_size=q_size, num_heads=num_heads, n_head_channels=n_head_channels,
                               n_groups=n_groups,attn_drop=attn_drop, proj_drop=proj_drop, stride=stride,
                               offset_range_factor=offset_range_factor, use_pe=use_pe, dwc_pe=dwc_pe,
                               no_off=no_off, fixed_pe=fixed_pe, ksize=ksize, log_cpb=log_cpb)


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.red = nn.Linear(2 * dim, dim)

    def forward(self, x, hx=None):
        B, L, C = x.shape
        H, W = self.input_resolution
        assert L == H * W, "input feature has wrong size"

        shortcut = x   # B L C
        x = self.norm1(x)
        # hx 不是None说明进行local attention
        if hx is not None:
            hx = self.norm1(hx)
            x = torch.cat((x, hx), -1) # 拼接导致拼接后的维度变为2dim
            x = self.red(x) # 将x和hx拼接并将维度由2dim改为dim
            x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            x,_,_ = self.window_attn(x, mask=None)
        else:
            x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

            x,_,_ = self.dat_attn(x)
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class DatTransformer(nn.Module):
    '''
        depth: DatTransformerBlock 的数量
        dim：embed_dim
        window_size: 进行LocalAttention的窗口大小
        input_resolution: 输入到transformer中的图像的分辨率，为img_size//patch_size
    '''

    def __init__(self, dim,num_heads, input_resolution, depth, window_size,
                 n_head_channels, n_groups,
                 proj_drop, stride, offset_range_factor, use_pe, dwc_pe,
                 no_off, fixed_pe, ksize, log_cpb,mlp_ratio=4.,
                  attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, flag=None,
                 ):
        super(DatTransformer, self).__init__()

        self.layers = nn.ModuleList([
            DatTransformerBlock(dim=dim, input_resolution=input_resolution,
                                num_heads=num_heads, window_size=window_size,
                                mlp_ratio=mlp_ratio,
                                attn_drop=attn_drop,
                                drop_path=drop_path[depth - i - 1] if (flag == 0) else drop_path[i],
                                norm_layer=norm_layer,
                                n_head_channels=n_head_channels, n_groups=n_groups,
                                proj_drop=proj_drop, stride=stride, offset_range_factor=offset_range_factor,
                                use_pe=use_pe, dwc_pe=dwc_pe, no_off=no_off, fixed_pe=fixed_pe,
                                ksize=ksize, log_cpb=log_cpb)
            for i in range(depth)])

    def forward(self, xt, hx):
        outputs = []

        for index, layer in enumerate(self.layers):
            if index == 0:
                x = layer(xt, hx)
                outputs.append(x)

            else:
                if index % 2 == 0:
                    x = layer(outputs[-1], xt)
                    outputs.append(x)

                if index % 2 == 1:
                    x = layer(outputs[-1], None)
                    outputs.append(x)

        return outputs[-1]

class DatLSTMCell(nn.Module):

    def __init__(self, n_head_channels, n_groups,input_resolution,
            stride,offset_range_factor, use_pe, dwc_pe,
            no_off, fixed_pe, ksize, log_cpb,dim,
                 num_heads, window_size, depth,
                 mlp_ratio=4., attn_drop=0.,proj_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, flag=None):
        super(DatLSTMCell, self).__init__()
        self.Dat = DatTransformer(num_heads=num_heads, n_head_channels=n_head_channels,
                                  n_groups=n_groups, proj_drop=proj_drop, stride=stride,
                                  offset_range_factor=offset_range_factor, use_pe=use_pe, dwc_pe=dwc_pe,
                                  no_off=no_off, fixed_pe=fixed_pe, ksize=ksize, log_cpb=log_cpb,
                                  dim=dim, input_resolution=input_resolution, depth=depth,
                                  window_size=window_size, mlp_ratio=mlp_ratio,
                                  attn_drop=attn_drop,
                                  drop_path=drop_path, norm_layer=norm_layer, flag=flag
        )
    def forward(self, xt, hidden_states):
        if hidden_states is None:
            B, L, C = xt.shape
            hx = torch.zeros(B, L, C).to(xt.device)
            cx = torch.zeros(B, L, C).to(xt.device)

        else:
            hx, cx = hidden_states

        Ft = self.Dat(xt, hx)

        gate = torch.sigmoid(Ft)
        cell = torch.tanh(Ft)

        cy = gate * (cx + cell)
        hy = gate * torch.tanh(cy)
        hx = hy
        cx = cy

        return hx, (hx, cx)
class DownSample(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, depths_downsample, num_heads, window_size,
                 n_head_channels, n_groups,
                 proj_drop, stride,
                 offset_range_factor, use_pe, dwc_pe,
                 no_off, fixed_pe, ksize, log_cpb,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm):
        super(DownSample, self).__init__()
        # self.aaa = depths_downsample  [2,6]
        self.num_layers = len(depths_downsample)  # 下采样的层数
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        patches_resolution = self.patch_embed.patches_resolution  # [img_H//patch_size,img_W//patch_size]

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_downsample))]

        self.layers = nn.ModuleList()
        self.downsample = nn.ModuleList()

        for i_layer in range(self.num_layers):
            downsample = PatchMerging(input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                        patches_resolution[1] // (2 ** i_layer)),
                                      dim=int(embed_dim * 2 ** i_layer))

            layer = DatLSTMCell(dim=int(embed_dim * 2 ** i_layer),
                                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                  patches_resolution[1] // (2 ** i_layer)),
                                depth=depths_downsample[i_layer],  # [2,6]
                                num_heads=num_heads[i_layer],
                                n_groups=n_groups[i_layer],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths_downsample[:i_layer]):sum(depths_downsample[:i_layer + 1])],
                                norm_layer=norm_layer,
                                n_head_channels=n_head_channels,
                                proj_drop=proj_drop, stride=stride,
                                offset_range_factor=offset_range_factor, use_pe=use_pe, dwc_pe=dwc_pe,
                                no_off=no_off, fixed_pe=fixed_pe, ksize=ksize,log_cpb=log_cpb
                                )

            self.layers.append(layer)
            self.downsample.append(downsample)

    def forward(self, x, y):

        # x[8, 1, 64, 64]
        x = self.patch_embed(x)
        # x[8, 1024, 128]
        hidden_states_down = []
        # 两个downSample块
        for index, layer in enumerate(self.layers):
            # 先进入一个DatLSTM Cell
            # x[8,1024,128]
            x, hidden_state = layer(x, y[index])
            # 再进行一次下采样
            x = self.downsample[index](x)
            # [8,1024,128]  第0次
            # [8, 256, 256] 第一次
            # [8, 64, 512]  第二次
            hidden_states_down.append(hidden_state)

        return hidden_states_down, x


class UpSample(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, depths_upsample, num_heads, window_size,
                 n_head_channels, n_groups,
                 proj_drop, stride,
                 offset_range_factor, use_pe, dwc_pe,
                 no_off, fixed_pe, ksize, log_cpb, mlp_ratio=4.,
                 attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, flag=0):
        super(UpSample, self).__init__()

        self.img_size = img_size
        self.num_layers = len(depths_upsample)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        patches_resolution = self.patch_embed.patches_resolution
        self.Unembed = PatchInflated(in_chans=in_chans, embed_dim=embed_dim, input_resolution=patches_resolution)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_upsample))]

        self.layers = nn.ModuleList()
        self.upsample = nn.ModuleList()

        for i_layer in range(self.num_layers):
            resolution1 = (patches_resolution[0] // (2 ** (self.num_layers - i_layer)))
            resolution2 = (patches_resolution[1] // (2 ** (self.num_layers - i_layer)))

            dimension = int(embed_dim * 2 ** (self.num_layers - i_layer))
            upsample = PatchExpanding(input_resolution=(resolution1, resolution2), dim=dimension)

            layer = DatLSTMCell(dim=dimension, input_resolution=(resolution1, resolution2),
                                depth=depths_upsample[(self.num_layers - 1 - i_layer)],
                                num_heads=num_heads[(self.num_layers + i_layer)],
                                n_groups=n_groups[(self.num_layers + i_layer)],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths_upsample[:(self.num_layers - 1 - i_layer)]):
                                              sum(depths_upsample[:(self.num_layers - 1 - i_layer) + 1])],
                                norm_layer=norm_layer, flag=flag,
                                n_head_channels=n_head_channels,
                                proj_drop=proj_drop, stride=stride,
                                offset_range_factor=offset_range_factor, use_pe=use_pe, dwc_pe=dwc_pe,
                                no_off=no_off, fixed_pe=fixed_pe, ksize=ksize, log_cpb=log_cpb
                                )

            self.layers.append(layer)
            self.upsample.append(upsample)

    def forward(self, x, y):
        # print("xxxddd",x.shape)
        hidden_states_up = []

        for index, layer in enumerate(self.layers):
            x, hidden_state = layer(x, y[index])
            x = self.upsample[index](x)
            hidden_states_up.append(hidden_state)

        x = torch.sigmoid(self.Unembed(x))

        return hidden_states_up, x

class DatLSTM(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, depths_downsample, depths_upsample, num_heads,
                 window_size,n_head_channels, n_groups,
                 proj_drop, stride,
                 offset_range_factor, use_pe, dwc_pe,
                 no_off, fixed_pe, ksize, log_cpb,):
        super(DatLSTM, self).__init__()

        self.Downsample = DownSample(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                     embed_dim=embed_dim, depths_downsample=depths_downsample,
                                     num_heads=num_heads, window_size=window_size,
                                     n_head_channels=n_head_channels, n_groups=n_groups,
                                     proj_drop=proj_drop, stride=stride,
                                     offset_range_factor=offset_range_factor, use_pe=use_pe, dwc_pe=dwc_pe,
                                     no_off=no_off, fixed_pe=fixed_pe, ksize=ksize, log_cpb=log_cpb,
                                    )

        self.Upsample = UpSample(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                 embed_dim=embed_dim, depths_upsample=depths_upsample,
                                 num_heads=num_heads, window_size=window_size,
                                 n_head_channels=n_head_channels, n_groups=n_groups,
                                 proj_drop=proj_drop, stride=stride,
                                 offset_range_factor=offset_range_factor, use_pe=use_pe, dwc_pe=dwc_pe,
                                 no_off=no_off, fixed_pe=fixed_pe, ksize=ksize, log_cpb=log_cpb,
                                 )

    def forward(self, input, states_down, states_up):
        # input [8,1,64,64]
        # print("input",input.shape)

        states_down, x = self.Downsample(input, states_down)
        # x[8,64,512]
        states_up, output = self.Upsample(x, states_up)
        return output, states_down, states_up


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_head_channels = 32
    n_groups = [2,4,8,4]
    num_heads = [4,8,16,8]
    attn_drop = 0.0
    proj_drop = 0.0
    stride = 2
    offset_range_factor = 2
    use_pe = True
    dwc_pe = False
    no_off = False
    fixed_pe = False
    ksize = 3
    log_cpb = False
    img_size = 64
    patch_size = 2
    device = 'cuda:0'
    model = DatLSTM(img_size=img_size, patch_size=patch_size,
                     in_chans=1, embed_dim=128,
                     depths_downsample=[2,6], depths_upsample=[6,2],
                     num_heads=num_heads, window_size=4,
                     n_head_channels= n_head_channels, n_groups=n_groups,
                     proj_drop=proj_drop, stride=stride,
                     offset_range_factor=offset_range_factor, use_pe=use_pe, dwc_pe=dwc_pe,
                     no_off=no_off, fixed_pe=fixed_pe, ksize=ksize, log_cpb=log_cpb,
                     ).to(device)
    inputs = torch.randn((8, 10, 1, 64, 64)).float().to(device)
    x1 = inputs[:, 0]
    states_down = [None] * len([2,6])
    states_up = [None] * len([6,2])
    output,states_down, states_up = model(x1, states_down,states_up)
    print("output",output.shape)
    print("main")