import torch
import torch.nn as nn
import math
import numpy as np


class Block(nn.Module):
    def __init__(self, dim, dim_out, kernel_size=3, groups=4):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size, padding ='same')
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, kernel_size = 3,time_emb_dim=None, groups=8):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out)
        ) if time_emb_dim is not None else None

        self.block1 = Block(dim, dim_out, kernel_size = kernel_size, groups=groups)
        self.block2 = Block(dim_out, dim_out, kernel_size = kernel_size, groups=groups)

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):


        # The time embedding here is a simplyfied version compared to the original version
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            # print(time_emb.shape)
            time_emb = time_emb[..., None, None]
            # print('timeshape')
            # print(time_emb.shape)
            x_t = time_emb + x  # time embedding as addition
        else:
            x_t = x.clone()

        h = self.block1(x_t)
        h = self.block2(h)

        return h + self.res_conv(x)


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, dim_out if dim_out is not None else dim, 2, stride=2)


# Didn't use convtranspose since the number of parameters is already very large.
def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear'),
        nn.Conv2d(dim, dim_out, 3, padding='same')
    )

# Decided to add attention layer because we some how need more global information
# Later on I find that dot product attention is too expensive for either memory or computation for unshrinked image

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_heads = 32):
        super().__init__()

        self.scale = dim_heads ** -0.5 # for numerically stability
        self.heads = heads
        self.dim_heads = dim_heads

        hidden_dim = heads * dim_heads
        self.to_q = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.to_k = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.to_v = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        q = self.to_q(x).view(b, self.heads, self.dim_heads, h*w) # B (h c) h w
        k = self.to_k(x).view(b, self.heads, self.dim_heads, h*w)
        v = self.to_v(x).view(b, self.heads, self.dim_heads, h*w)

        q = q*self.scale

        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)
        out = out.permute((0,1,3,2)).reshape((b,self.heads*self.dim_heads, h, w ))

        return self.to_out(out)


class CrossAttention(nn.Module):

    '''
    perform cross attention on two input 3d tensor (c1,h1,h1) and (c2,h2,h2)
    crossatten with residual connection
    '''
    def __init__(self, dim1, dim2, residual=True, heads=4, dim_heads=32):
        super().__init__()

        self.scale = dim_heads ** -0.5  # for numerically stability
        self.heads = heads
        self.dim_heads = dim_heads
        self.residual = residual

        hidden_dim = heads * dim_heads

        self.seq1to_q = nn.Conv2d(dim1, hidden_dim, 1, bias=False)
        self.seq1to_k = nn.Conv2d(dim1, hidden_dim, 1, bias=False)
        self.seq1to_v = nn.Conv2d(dim1, hidden_dim, 1, bias=False)
        self.to_out1 = nn.Conv2d(hidden_dim, dim1, 1)

        self.seq2to_q = nn.Conv2d(dim2, hidden_dim, 1, bias=False)
        self.seq2to_k = nn.Conv2d(dim2, hidden_dim, 1, bias=False)
        self.seq2to_v = nn.Conv2d(dim2, hidden_dim, 1, bias=False)
        self.to_out2 = nn.Conv2d(hidden_dim, dim2, 1)

    def forward(self, seq1, seq2):
        b1, c1, h1, w1 = seq1.shape
        b2, c2, h2, w2 = seq2.shape
        if self.residual:
            seq1c = seq1.clone()
            seq2c = seq2.clone()

        q1 = self.seq1to_q(seq1).view(b1, self.heads, self.dim_heads, h1 * w1)  # B (h c) h w
        k1 = self.seq1to_k(seq1).view(b1, self.heads, self.dim_heads, h1 * w1)
        v1 = self.seq1to_v(seq1).view(b1, self.heads, self.dim_heads, h1 * w1)

        q1 = q1 * self.scale

        q2 = self.seq2to_q(seq2).view(b2, self.heads, self.dim_heads, h2 * w2)  # B (h c) h w
        k2 = self.seq2to_k(seq2).view(b2, self.heads, self.dim_heads, h2 * w2)
        v2 = self.seq2to_v(seq2).view(b2, self.heads, self.dim_heads, h2 * w2)

        q2 = q2 * self.scale



        sim12 = torch.einsum('b h d i, b h d j -> b h i j', q1, k2)
        sim21 = torch.einsum('b h d i, b h d j -> b h i j', q2, k1)
        attn12 = sim12.softmax(dim=-1)
        attn21 = sim21.softmax(dim=-1)



        out2 = torch.einsum('b h i j, b h d j -> b h i d', attn21, v1)
        out2 = out2.permute((0, 1, 3, 2)).reshape((b2, self.heads * self.dim_heads, h2, w2))

        out1 = torch.einsum('b h i j, b h d j -> b h i d', attn12, v2)
        out1 = out1.permute((0, 1, 3, 2)).reshape((b1, self.heads * self.dim_heads, h1, w1))

        return self.to_out1(out1)+seq1c, self.to_out2(out2)+seq2c


class PsvCrossAttention(nn.Module):

    '''
    perform cross attention on two input 3d tensor (c1,h1,h1) and (c2,h2,h2)
    '''
    def __init__(self, dim1, dim2, heads=4, dim_heads=32):
        super().__init__()

        self.scale = dim_heads ** -0.5  # for numerically stability
        self.heads = heads
        self.dim_heads = dim_heads

        hidden_dim = heads * dim_heads

        self.seq1to_q = nn.Conv2d(dim1, hidden_dim, 1, bias=False)
        self.seq1to_k = nn.Conv2d(dim1, hidden_dim, 1, bias=False)
        self.seq1to_v = nn.Conv2d(dim1, hidden_dim, 1, bias=False)
        self.to_out1 = nn.Conv2d(hidden_dim, dim1, 1)

        self.seq2to_q = nn.Conv2d(dim2, hidden_dim, 1, bias=False)
        self.seq2to_k = nn.Conv2d(dim2, hidden_dim, 1, bias=False)
        self.seq2to_v = nn.Conv2d(dim2, hidden_dim, 1, bias=False)
        self.to_out2 = nn.Conv2d(hidden_dim, dim2, 1)

    def forward(self, seq1, seq2):
        b1, c1, h1, w1 = seq1.shape
        b2, c2, h2, w2 = seq2.shape

        q1 = self.seq1to_q(seq1).view(b1, self.heads, self.dim_heads, h1 * w1)  # B (h c) h w
        k1 = self.seq1to_k(seq1).view(b1, self.heads, self.dim_heads, h1 * w1)
        v1 = self.seq1to_v(seq1).view(b1, self.heads, self.dim_heads, h1 * w1)

        q1 = q1 * self.scale

        q2 = self.seq2to_q(seq2).view(b2, self.heads, self.dim_heads, h2 * w2)  # B (h c) h w
        k2 = self.seq2to_k(seq2).view(b2, self.heads, self.dim_heads, h2 * w2)
        v2 = self.seq2to_v(seq2).view(b2, self.heads, self.dim_heads, h2 * w2)

        q2 = q2 * self.scale



        sim12 = torch.einsum('b h d i, b h d j -> b h i j', q1, k2)
        sim21 = torch.einsum('b h d i, b h d j -> b h i j', q2, k1)
        attn12 = sim12.softmax(dim=-1)
        attn21 = sim21.softmax(dim=-1)



        out1 = torch.einsum('b h i j, b h d j -> b h i d', attn12, v1)
        out1 = out1.permute((0, 1, 3, 2)).reshape((b2, self.heads * self.dim_heads, h2, w2))

        out2 = torch.einsum('b h i j, b h d j -> b h i d', attn21, v2)
        out2 = out2.permute((0, 1, 3, 2)).reshape((b1, self.heads * self.dim_heads, h1, w1))

        return self.to_out1(out1), self.to_out2(out2)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_heads=32):
        super().__init__()
        self.scale = dim_heads ** -0.5
        self.heads = heads
        self.dim_heads = dim_heads
        hidden_dim = dim_heads * heads
        self.to_q = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.to_k = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.to_v = nn.Conv2d(dim, hidden_dim, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        q = self.to_q(x).view(b, self.heads, self.dim_heads, h * w)  # B (h c) h w
        k = self.to_k(x).view(b, self.heads, self.dim_heads, h * w)
        v = self.to_v(x).view(b, self.heads, self.dim_heads, h * w)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = out.view((b, self.heads * self.dim_heads, h, w))
        return self.to_out(out)


# group norm and residule function
# These two codes are just copy and paste, from https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main/denoising_diffusion_pytorch

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class Params():
    def __init__(self,
                 input_size = 128,
                 input_channel = 3,
                 output_channel = 3,
                 initial_dim = 16,
                 map_neckdim = 32,
                 fine_neckdim = 256,
                 coarse_neckdim = 256,
                 coarse_kernel_size = (9,7,5,3),
                 encode_attention = True,
                 expanding = (1,2,4,8),
                 crossattn = True,
                 use_conv = True,
                 conv_neckdim = 16,
                 output_var = 1e-4
                 ):


        """

        :param input_size:
        :param input_channel:
        :param output_channel:
        :param initial_dim:
        :param map_neckdim:
        :param fine_neckdim:
        :param coarse_neckdim:
        :param coarse_kernel_size:
        :param encode_attention:
        :param expanding:
        :param crossattn:
        :param use_conv:
        :param conv_neckdim:
        :param output_var: variance of output, set to 1e-4 by default, corresponds to std=0.01
        """
        self.input_size = input_size
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.initial_dim = initial_dim
        self.map_neckdim = map_neckdim
        self.fine_neckdim = fine_neckdim
        self.coarse_neckdim = coarse_neckdim
        self.coarse_kernel_size = coarse_kernel_size
        self.encode_attention = encode_attention
        self.expanding = expanding
        self.crossattn = crossattn
        self.useconv = use_conv
        self.conv_neckdim = conv_neckdim
        self.output_var = output_var

        self.dims = [self.initial_dim, *map(lambda m: self.initial_dim * m, self.expanding)]
        self.minimap_size = int((self.input_size / 2 ** len(self.expanding)))
        self.input_features = self.map_neckdim * self.minimap_size ** 2 // 2


class SepVAEEncoder(nn.Module):
    def __init__(self,
                 params: Params,
                 device = torch.device('cpu')
                 ):
        super(SepVAEEncoder, self).__init__()
        self.params = params

        self.init_conv = nn.Conv2d(self.params.input_channel, self.params.initial_dim, 3, padding='same')  # after this B*init*256*256
        dims = self.params.dims

        # print(dims)
        self.fine_encoder = nn.ModuleList([])
        self.coarse_encoder = nn.ModuleList([])
        self.device = device

        for index in range(len(dims) - 1):
            dim_in = dims[index]
            dim_out = dims[index + 1]
            self.fine_encoder.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in),
                # ResnetBlock(dim_in, dim_in),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if self.params.encode_attention else nn.Identity(),  # use linear attention because it is cheap
                Downsample(dim_in, dim_out)
            ]))

        for index in range(len(dims) - 1):
            dim_in = dims[index]
            dim_out = dims[index + 1]
            self.coarse_encoder.append(nn.ModuleList([
                # ResnetBlock(dim_in, dim_in, kernel_size = self.params.coarse_kernel_size[index]),
                ResnetBlock(dim_in, dim_in, kernel_size = self.params.coarse_kernel_size[index]),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if self.params.encode_attention else nn.Identity(),  # use linear attention because it is cheap
                Downsample(dim_in, dim_out)
            ]))

        # all the model up outputs the two feature maps of B x initial_dim*expanding[-1] * mini_fine(coarse) * mini_fine(coarse)

        self.crossattn = CrossAttention(dims[-1],dims[-1]) # crossattn with residual connection

        self.fine_compress = nn.Sequential(nn.Conv2d(dims[-1], self.params.map_neckdim, 1), nn.LeakyReLU())
        self.coarse_compress = nn.Sequential(nn.Conv2d(dims[-1], self.params.map_neckdim, 1), nn.LeakyReLU())

        if not self.params.useconv:

            self.minimap_size = int((self.params.input_size/2**len(self.params.expanding)))
            self.input_features = self.params.map_neckdim * self.minimap_size**2
            self.input_features = int(self.input_features/2)
            self.to_finemean = nn.Sequential(nn.BatchNorm1d(self.input_features),nn.Linear(self.input_features, self.params.fine_neckdim),nn.LeakyReLU())
            self.to_finelogvar = nn.Sequential(nn.BatchNorm1d(self.input_features),nn.Linear(self.input_features, self.params.fine_neckdim),nn.LeakyReLU())

            self.to_coarsemean = nn.Sequential(nn.BatchNorm1d(self.input_features),nn.Linear(self.input_features, self.params.coarse_neckdim),nn.LeakyReLU())
            self.to_coarselogvar = nn.Sequential(nn.BatchNorm1d(self.input_features),nn.Linear(self.input_features, self.params.coarse_neckdim),nn.LeakyReLU())

        else:
            self.to_finemean = nn.Conv2d(self.params.map_neckdim, self.params.conv_neckdim,1)
            self.to_finelogvar = nn.Conv2d(self.params.map_neckdim, self.params.conv_neckdim,1)
            self.to_coarsemean = nn.Conv2d(self.params.map_neckdim, self.params.conv_neckdim,1)
            self.to_coarselogvar = nn.Conv2d(self.params.map_neckdim, self.params.conv_neckdim,1)

    def forward(self,x):
        x = self.init_conv(x)
        # print(x.shape,'initial')

        xr = x.clone()


        for block2, attn, downsample in self.fine_encoder:
            x = block2(x)
            x = attn(x)
            # print(x.shape)
            x = downsample(x)
            # print(x.shape)

        for block2, attn, downsample in self.coarse_encoder:
            xr = block2(xr)
            xr = attn(xr)
            xr = downsample(xr)

        x, xr = self.crossattn(x, xr)

        x = self.fine_compress(x)
        xr = self.coarse_compress(xr)
        # print(x.shape, xr.shape)

        if not self.params.useconv:

            finemean_pt, finevar_pt = torch.chunk(x, 2, 1)
            coarsemean_pt, coarsevar_pt = torch.chunk(xr, 2, 1)

            fine_pos_mean = self.to_finemean(torch.flatten(finemean_pt, 1))
            fine_pos_var = self.to_finelogvar(torch.flatten(finevar_pt, 1)).exp()

            coarse_pos_mean = self.to_coarsemean(torch.flatten(coarsemean_pt, 1))
            coarse_pos_var = self.to_coarselogvar(torch.flatten(coarsevar_pt, 1)).exp()

        else:
            fine_pos_mean = self.to_finemean(x)
            fine_pos_var = self.to_finelogvar(x).exp()

            coarse_pos_mean = self.to_coarsemean(xr)
            coarse_pos_var = self.to_coarselogvar(xr).exp()

        return fine_pos_mean, fine_pos_var, coarse_pos_mean, coarse_pos_var



class SepVAEDecoder(nn.Module):
    def __init__(self,
                 params: Params,
                 device = torch.device('cpu')
                 ):
        super(SepVAEDecoder, self).__init__()

        self.params = params
        self.device = device
        if not self.params.useconv:
        # first from 1d vec to 3d tensor
            self.fine_sample_to_minimap = nn.Sequential(
                nn.Linear(self.params.fine_neckdim, self.params.input_features*2),
                nn.GELU()
            )
            self.coarse_sample_to_minimap = nn.Sequential(
                nn.Linear(self.params.coarse_neckdim, self.params.input_features*2),
                nn.GELU()
            )

        # detail expander
            self.fineup = nn.Sequential(nn.Conv2d(self.params.map_neckdim, self.params.dims[-1],  1), nn.LeakyReLU())
            self.coarseup = nn.Sequential(nn.Conv2d(self.params.map_neckdim, self.params.dims[-1],  1), nn.LeakyReLU())
        else:
            self.fineup = nn.Sequential(nn.Conv2d(self.params.conv_neckdim, self.params.dims[-1],  1), nn.LeakyReLU())
            self.coarseup = nn.Sequential(nn.Conv2d(self.params.conv_neckdim, self.params.dims[-1],  1), nn.LeakyReLU())

        self.fineUpsample = nn.ModuleList([])
        self.coarseUpsample = nn.ModuleList([])

        r_dims = self.params.dims.copy()
        r_dims.reverse()
        for index in range(len(r_dims) - 1):
            dim_out = r_dims[index]
            dim_in = r_dims[index + 1]

            self.fineUpsample.append(nn.ModuleList([
                ResnetBlock(dim_out, dim_out),
                # ResnetBlock(dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in)
            ]))

        for index in range(len(r_dims) - 1):
            dim_out = r_dims[index]
            dim_in = r_dims[index + 1]

            self.coarseUpsample.append(nn.ModuleList([
                ResnetBlock(dim_out + dim_out, dim_out),
                # ResnetBlock(dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out , LinearAttention(dim_out))),
                Upsample(dim_out, dim_in)
            ]))

        self.output = nn.Conv2d(self.params.initial_dim, self.params.output_channel, 1)

    def forward(self, fine_sample, coarse_sample):

        h = []
        if not self.params.useconv:
            finemap = self.fine_sample_to_minimap(fine_sample).view(-1, self.params.map_neckdim, self.params.minimap_size,
                                                                    self.params.minimap_size)


            coarsemap = self.coarse_sample_to_minimap(coarse_sample).view(-1, self.params.map_neckdim, self.params.minimap_size,
                                                                        self.params.minimap_size)
            finemap = self.fineup(finemap)
            coarsemap = self.coarseup(coarsemap)
        else:
            finemap = self.fineup(fine_sample)
            coarsemap = self.coarseup(coarse_sample)

        for block, attn, upsample in self.fineUpsample:
            finemap = block(finemap)
            h.append(finemap)
            finemap = attn(finemap)
            finemap = upsample(finemap)

        h.reverse()
        for block, attn, upsample in self.coarseUpsample:
            coarsemap = torch.cat((coarsemap, h.pop()), dim=1)
            coarsemap = block(coarsemap)
            coarsemap = attn(coarsemap)
            coarsemap = upsample(coarsemap)

        output = self.output(coarsemap)

        return output




class SepVAE(nn.Module):
    def __init__(self,
                 params: Params,
                 device = torch.device('cpu')
                 ):
        super(SepVAE,self).__init__()

        self.encoder = SepVAEEncoder(params, device)
        self.decoder = SepVAEDecoder(params, device)
        self.params = params
        self.device = device

    def forward(self, x):

        fine_pos_mean, fine_pos_var, coarse_pos_mean, coarse_pos_var = self.encoder(x)

        # sample

        fine_sample = torch.randn_like(fine_pos_mean).to(self.device) * fine_pos_var.sqrt() + fine_pos_mean
        coarse_sample = torch.randn_like(coarse_pos_mean).to(self.device) * coarse_pos_var.sqrt() + coarse_pos_mean

        output = self.decoder(fine_sample, coarse_sample)

        return fine_pos_mean, fine_pos_var, coarse_pos_mean, coarse_pos_var, fine_sample, coarse_sample, output



    def elbo(self, fine_pos_mean, fine_pos_var, coarse_pos_mean, coarse_pos_var, output, x):

        def gaussian_log(x, mu, var):
            sig = torch.sqrt(var)
            return - torch.log(sig) - 0.5*np.log(2*np.pi) -  0.5*((x - mu)/sig)**2

        def kl_divergence(mu1, var1):
            # kl divergence of standard normal
            mu2 = 0.
            var2 = 1.

            return 0.5*torch.log(var2/var1) + (var1 + (mu1-mu2)**2)/var2*0.5 - 0.5

        kl_fine = kl_divergence(fine_pos_mean,fine_pos_var).mean(dim=0)
        kl_coarse = kl_divergence(coarse_pos_mean, coarse_pos_var).mean(dim=0)
        loglike = gaussian_log(x,output,torch.ones_like(output)*self.params.output_var).mean(dim=0)


        return  kl_fine.sum() + kl_coarse.sum()  - loglike.sum()



















class SepVAE_legacy(nn.Module):
    def __init__(self,
                 params: Params,
                 device = torch.device('cpu')
                 ):
        super(SepVAE_legacy, self).__init__()


        self.params = params

        self.init_conv = nn.Conv2d(self.params.input_channel, self.params.initial_dim, 3, padding='same')  # after this B*init*256*256
        dims = [self.params.initial_dim, *map(lambda m: self.params.initial_dim * m, self.params.expanding)]
        self.dims = dims
        # print(dims)
        self.fine_encoder = nn.ModuleList([])
        self.coarse_encoder = nn.ModuleList([])
        self.device = device

        for index in range(len(dims) - 1):
            dim_in = dims[index]
            dim_out = dims[index + 1]
            self.fine_encoder.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in),
                # ResnetBlock(dim_in, dim_in),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if self.params.encode_attention else nn.Identity(),  # use linear attention because it is cheap
                Downsample(dim_in, dim_out)
            ]))

        for index in range(len(dims) - 1):
            dim_in = dims[index]
            dim_out = dims[index + 1]
            self.coarse_encoder.append(nn.ModuleList([
                # ResnetBlock(dim_in, dim_in, kernel_size = self.params.coarse_kernel_size[index]),
                ResnetBlock(dim_in, dim_in, kernel_size = self.params.coarse_kernel_size[index]),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if self.params.encode_attention else nn.Identity(),  # use linear attention because it is cheap
                Downsample(dim_in, dim_out)
            ]))

        # all the model up outputs the two feature maps of B x initial_dim*expanding[-1] * mini_fine(coarse) * mini_fine(coarse)

        self.crossattn = CrossAttention(self.dims[-1],self.dims[-1]) # crossattn with residual connection

        self.fine_compress = nn.Sequential(nn.Conv2d(self.dims[-1], self.params.map_neckdim, 1), nn.LeakyReLU())
        self.coarse_compress = nn.Sequential(nn.Conv2d(self.dims[-1], self.params.map_neckdim, 1), nn.LeakyReLU())


        self.minimap_size = int((self.params.input_size/2**len(self.params.expanding)))
        self.input_features = self.params.map_neckdim * self.minimap_size**2
        self.input_features = int(self.input_features/2)
        self.to_finemean = nn.Sequential(nn.BatchNorm1d(self.input_features),nn.Linear(self.input_features, self.params.fine_neckdim),nn.LeakyReLU())
        self.to_finelogvar = nn.Sequential(nn.BatchNorm1d(self.input_features),nn.Linear(self.input_features, self.params.fine_neckdim),nn.LeakyReLU())

        self.to_coarsemean = nn.Sequential(nn.BatchNorm1d(self.input_features),nn.Linear(self.input_features, self.params.coarse_neckdim),nn.LeakyReLU())
        self.to_coarselogvar = nn.Sequential(nn.BatchNorm1d(self.input_features),nn.Linear(self.input_features, self.params.coarse_neckdim),nn.LeakyReLU())

        # below is the reconstruction part

        # first from 1d vec to 3d tensor
        self.fine_sample_to_minimap = nn.Sequential(
            nn.Linear(self.params.fine_neckdim, self.input_features*2),
            nn.GELU()
        )
        self.coarse_sample_to_minimap = nn.Sequential(
            nn.Linear(self.params.coarse_neckdim, self.input_features*2),
            nn.GELU()
        )

        # detail expander
        self.fineup = nn.Sequential(nn.Conv2d(self.params.map_neckdim, self.dims[-1],  1), nn.LeakyReLU())
        self.coarseup = nn.Sequential(nn.Conv2d(self.params.map_neckdim, self.dims[-1],  1), nn.LeakyReLU())

        self.fineUpsample = nn.ModuleList([])
        self.coarseUpsample = nn.ModuleList([])

        r_dims = dims.copy()
        r_dims.reverse()
        for index in range(len(dims) - 1):
            dim_out = r_dims[index]
            dim_in = r_dims[index + 1]

            self.fineUpsample.append(nn.ModuleList([
                ResnetBlock(dim_out, dim_out),
                # ResnetBlock(dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in)
            ]))

        for index in range(len(dims) - 1):
            dim_out = r_dims[index]
            dim_in = r_dims[index + 1]

            self.coarseUpsample.append(nn.ModuleList([
                ResnetBlock(dim_out + dim_out, dim_out),
                # ResnetBlock(dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out , LinearAttention(dim_out))),
                Upsample(dim_out, dim_in)
            ]))

        self.output = nn.Conv2d(self.params.initial_dim, self.params.output_channel, 1)







    def forward(self, x):

        x = self.init_conv(x)
        # print(x.shape,'initial')

        xr = x.clone()

        # for block1, block2, attn, downsample in self.fine_encoder:
        #     x = block1(x)
        #     # h.append(x)
        #     # print(x.shape)
        #
        #     x = block2(x)
        #     x = attn(x)
        #     # h.append(x)
        #     # print(x.shape)
        #
        #     x = downsample(x)
        #     # print(x.shape)
        # # print('end fine')
        #
        # for block1, block2, attn, downsample in self.coarse_encoder:
        #     xr = block1(xr)
        #     # h.append(x)
        #     # print(x.shape)
        #
        #     xr = block2(xr)
        #     xr = attn(xr)
        #     # h.append(x)
        #     # print(x.shape)
        #
        #     xr = downsample(xr)
        #     # print(xr.shape)

        for block2, attn, downsample in self.fine_encoder:

            x = block2(x)
            x = attn(x)
            # print(x.shape)
            x = downsample(x)
            # print(x.shape)

        for block2, attn, downsample in self.coarse_encoder:

            xr = block2(xr)
            xr = attn(xr)
            xr = downsample(xr)

        x, xr = self.crossattn(x,xr)

        x = self.fine_compress(x)
        xr = self.coarse_compress(xr)
        # print(x.shape, xr.shape)

        finemean_pt, finevar_pt = torch.chunk(x, 2, 1)
        coarsemean_pt, coarsevar_pt = torch.chunk(xr, 2, 1)

        fine_pos_mean = self.to_finemean(torch.flatten(finemean_pt,1))
        fine_pos_var = self.to_finelogvar(torch.flatten(finevar_pt,1)).exp()

        coarse_pos_mean = self.to_coarsemean(torch.flatten(coarsemean_pt,1))
        coarse_pos_var = self.to_coarselogvar(torch.flatten(coarsevar_pt,1)).exp()

        # print(fine_pos_mean.shape, fine_pos_var.shape, coarse_pos_mean.shape, coarse_pos_var.shape)
        fine_sample = torch.randn_like(fine_pos_mean).to(self.device) * fine_pos_var.sqrt() + fine_pos_mean
        coarse_sample = torch.randn_like(coarse_pos_mean).to(self.device) * coarse_pos_var.sqrt() + coarse_pos_mean


        # reconstruction starting
        h = []
        finemap = self.fine_sample_to_minimap(fine_sample).view(-1, self.params.map_neckdim, self.minimap_size, self.minimap_size)
        finemap = self.fineup(finemap)

        coarsemap = self.coarse_sample_to_minimap(coarse_sample).view(-1, self.params.map_neckdim, self.minimap_size, self.minimap_size)
        coarsemap = self.coarseup(coarsemap)


        for block, attn, upsample in self.fineUpsample:
            finemap = block(finemap)
            h.append(finemap)
            finemap = attn(finemap)
            finemap = upsample(finemap)

        h.reverse()
        for block, attn, upsample in self.coarseUpsample:
            coarsemap = torch.cat((coarsemap, h.pop()), dim = 1)
            coarsemap = block(coarsemap)
            coarsemap = attn(coarsemap)
            coarsemap = upsample(coarsemap)

        output = self.output(coarsemap)

        return fine_pos_mean, fine_pos_var, coarse_pos_mean, coarse_pos_var, fine_sample, coarse_sample, output



















