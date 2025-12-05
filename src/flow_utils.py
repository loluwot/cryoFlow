import torch
from torch import optim, nn
from einops import rearrange
from torch.func import grad, jacrev
import torch.nn.functional as F
import numpy as np

def r6d_to_matrix(v):
    V = rearrange(v, '... (h w) -> ... h w', h=2)
    firsts = V[..., 0, :]
    seconds = V[..., 1, :]
    firsts = firsts / firsts.norm(dim=-1, keepdim=True)
    seconds = seconds - (firsts * seconds).sum(axis=-1, keepdims=True) * firsts
    seconds = seconds / seconds.norm(dim=-1, keepdim=True)
    thirds = torch.cross(firsts, seconds, dim=-1)
    return torch.stack([firsts, seconds, thirds], axis=-1)

mask = torch.tensor([[[0, 0, 0], [0, 0, -1], [0, 1, 0]],
                  [[0, 0, 1], [0, 0, 0], [-1, 0, 0]],
                  [[0, -1, 0], [1, 0, 0], [0, 0, 0]]])
def quat_to_matrix(v):
    s, xyz = v[..., -1], v[..., :3]
    M = 2 * xyz[..., None] * xyz[..., None, :]
    M = M + torch.eye(3, device=v.device) * (1 - 2 * xyz.square().sum(axis=-1))[..., None, None]
    M = M + 2 * (xyz[..., None, None] * mask.to(v)).sum(axis=-3) * s[..., None, None]
    return M

def f2d(f):
    f = torch.fft.ifftshift(f, dim=(-2, -1))
    return torch.fft.fftshift(torch.fft.ifftn(f, s=(f.shape[-2], f.shape[-1]), dim=(-2, -1)), dim=(-2, -1))

class Encoder(nn.Module):
    def __init__(self, config, full_config, out_features=6):
        super().__init__()
        additional_channels = 0
        if config.gaussian_filter:
            self.gaussian_pyramid = GaussianPyramid(
                    kernel_size=11,
                    kernel_variance=0.01,
                    num_octaves=config.num_octaves,
                    octave_scaling=10
            )
            additional_channels = config.num_octaves
        else:
            self.gaussian_pyramid = nn.Identity()
        
        self.M = CNNEncoderVGG16(1 + additional_channels, batch_norm=config.enc_batch_norm, high_res=(config.im_size > 128))
        v = self.M.get_out_shape(config.im_size, config.im_size).numel()
        self.oe = FCBlock(in_features=v,
                    out_features=out_features,
                    features=config.enc_linear_layers,
                    nonlinearity='relu',
                    last_nonlinearity=None,
                    batch_norm=config.enc_batch_norm)
    
    def forward(self, x):
        x = self.gaussian_pyramid(x)
        return self.oe(self.M(x).flatten(start_dim=1))

class AE(nn.Module):
    def __init__(self, config, full_config):
        super().__init__()
        self.encoder = Encoder(config, full_config)
        self.decoder = ImplicitFourierVolume(config.im_size, {'type': config.dec_type, 'force_symmetry': config.force_symmetry}, None).cuda()

    def forward(self, x, **kwargs):
        res = self.encoder(x)
        res = f2d(self.decoder(r6d_to_matrix(res))).real
        return self.loss(x, res), res

    def loss(self, x, res):
        return (x - res).square().mean(dim=(-1, -2)).sum()

proj = lambda u, v: ((u * v).sum(axis=-1) / (u.square().sum(axis=-1)))[..., None] * u
op = lambda p, q: (p - 2 * proj((p - q), p))

def reparam(x, sig=5):
    return 0.99 / (1 + x)

softsign = lambda x, b: (b + torch.cosh(x)).log()
softsign_d = lambda x, b: torch.sinh(x) / (b + torch.cosh(x))

def orthogonal(v): 
    """
    Returns an orthogonal basis to input vector. Assume input vector is unit norm.
    
    Parameters
    ----------
    v: torch.tensor
    """
    I = torch.eye(v.shape[-1]).to(v)
    M = v[..., None, :]
    
    # Choose unit vectors which are the "most" orthogonal (i.e. component is smallest)
    # Avoids issues in situations where vector has very small value in a component
    # E.g. v = (0.557, 0.557, 0.557, 1e-10); 
    # If naively use first three unit vectors, vector nearly lies in that space
    
    indices = v.abs().argsort(dim=-1)
    
    # Gram-Schmidt to orthogonalize basis 
    for i in range(v.shape[-1] - 1): 
        res = I[indices[..., i]] - (M[torch.arange(v.shape[0]), :, indices[..., i]][..., None] * M).sum(axis=-2)
        res = res / res.norm(dim=-1, keepdim=True)
        M = torch.cat([M, res[..., None, :]], dim=-2)
    return M[..., 1:, :]

def eval_kernel(x, center, k=20):
    # center (b, d)
    # x (n, b, d)
    M = k * (x[:, None] * torch.stack([center, -center], axis=0)).sum(axis=-1) # n 2 b
    res = (M - M.amax(dim=0, keepdim=True)).exp().sum(axis=1) # n b
    return res / res.amax(dim=0)

eps = torch.finfo(torch.float32).eps
class ConvexGradientLayer:
    def process_params(P):
        H = (P.shape[-1] - 1) // 6
        c = P[..., -1]
        W, u, b = rearrange(P[..., :-1], '... (n h) -> ... n h', h=H).split([4, 1, 1], dim=-2)
        u, b, c = F.softplus(u) + eps, F.softplus(b) + eps, F.softplus(c) + eps
        return W, u, b, c
    
    def transform_(p, W, u, b, c):
        p1 = softsign_d(p @ W, b[0])
        p2 = W @ (p1 * u[0]) + 2 * c * p
        v = p2 / p2.norm(dim=-1, keepdim=True)
        return v
    
    def transform(p, tup):
        res = torch.vmap(ConvexGradientLayer.transform_)(p, *tup)
        return res, -ConvexGradientLayer.log_det(p, tup, res)
    
    def log_det(p, tup, res, eps=eps):
        ep = orthogonal(p) # (B, 3, 4)
        eres = orthogonal(res) # (B, 3, 4)
        # assert not eres.isnan().any(), eres
        Js = torch.vmap(jacrev(ConvexGradientLayer.transform_))(p, *tup) # (B, 4, 4)
        d = torch.einsum('b l i, b i j, b k j -> b l k', ep, Js, eres)
        d = (torch.cross(d[..., 0], d[..., 1], dim=-1) * d[..., 2]).sum(axis=-1).abs()
        # assert (d > 0).all(), f'{p}, {ep}'
        return (d + eps).log()
    
    def n_features(config):
        return config.hidden_norm_size * 6 + 1

    # def sample_mode(params, num_samples=200, k=2):
    #     bs = params[0].shape[0]
    #     r = torch.randn(num_samples, bs, 4).to(params[0]) 
    #     r /= r.norm(dim=-1, keepdim=True)
    #     r_new, logp_abs = torch.vmap(ConvexGradientLayer.transform, in_dims=(0, None))(r, params)
    #     p_new = (logp_abs - logp_abs.amax(dim=0)).exp()
        
    #     with torch.no_grad():
    #         ## simple heuristic mode finding algorithm
    #         p_sort_idx = p_new.argsort(dim=0)        
    #         r_sorted = r_new[p_sort_idx, np.arange(bs)] # N B
    #         p_sorted = p_new[p_sort_idx, np.arange(bs)] # N B
    #         r_ = r_sorted.transpose(0, 1).cpu()
    #         D = torch.cdist(r_, r_, ).square() # B N N
    #         # print(D.shape)
    #         r, c = torch.tril_indices(*D.shape[1:])
    #         D[torch.arange(bs)[:, None], r, c] = D.amax(dim=(-1, -2))[:, None]
    #         heuristic = p_sorted.transpose(0, 1).cpu() * D.amin(dim=-1) # B N
    #         # 2k since flip symmetric, expect 2k modes 
    #         indices = (-heuristic).argsort(dim=-1)[:, :k*2:2] # B k
            
    #     # k B 4 -> (k B) 4
    #     return r_sorted[indices.T, torch.arange(bs)].reshape(-1, 4), logp_abs[p_sort_idx][indices.T, torch.arange(bs)].reshape(-1)

    # def sample_mode(params, num_samples=200, k=2):
    #     bs = params[0].shape[0]
    #     r = torch.randn(num_samples, bs, 4).to(params[0]) 
    #     r /= r.norm(dim=-1, keepdim=True)
        
    #     r_new, p_x = torch.vmap(ConvexGradientLayer.transform, in_dims=(0, None))(r, params)
    #     p_new = (p_x - p_x.amax(dim=0)).exp()
    #     ms = []
    #     for _ in range(k):
    #         ms.append(p_new.argmax(dim=0))
    #         p_new = p_new - eval_kernel(r_new, r_new[ms[-1], torch.arange(bs)], k=30) * p_new.amax(dim=0, keepdim=True)
    #     return r_new[torch.stack(ms), torch.arange(bs)].reshape(-1, 4), p_x[torch.stack(ms), torch.arange(bs)].reshape(-1)     

    def sample_mode(params, num_samples=200, k=2, bandwidth=50):
        bs = params[0].shape[0]
        r = torch.randn(num_samples // 2, bs, 4).to(params[0]) 
        r /= r.norm(dim=-1, keepdim=True)
        
        r_new, p_x = torch.vmap(ConvexGradientLayer.transform, in_dims=(0, None))(r, params)
        r_new, p_x = torch.cat([r_new, -r_new], dim=0), torch.cat([p_x, p_x], dim=0)
        
        p_new = (p_x - p_x.amax(dim=0)).exp().cpu()
        r_cpu = r_new.cpu()
        ms = []
        with torch.no_grad():
            for _ in range(k):
                ms.append(p_new.argmax(dim=0))
                p_new = p_new - eval_kernel(r_cpu, r_cpu[ms[-1], torch.arange(bs)], k=bandwidth) * p_new.amax(dim=0, keepdim=True)
        return r_new[torch.stack(ms), torch.arange(bs)].reshape(-1, 4), p_x[torch.stack(ms), torch.arange(bs)].reshape(-1)
            

class MoebiusLayer:
    def process_params(enc):
        qs = rearrange(enc, 'b (n d) -> b n d', d=4)
        qs = reparam(qs.norm(dim=-1, keepdim=True)) * qs
        return (qs,)
    
    def transform_(p, q):
        res = op(p, q) + op(p, -q)
        return res / res.norm(dim=-1, keepdim=True)
    
    def logdet(p, q, r2):
        # p (B, 4)
        qy2 = r2 - (p * q).sum(axis=-1).square()
        res = (1 - r2).log() + 3*(1 + r2).log() - 2 * (4 * qy2 + (1 - r2).square()).log()
        return res
    
    def transform(p, qs):
        qs, = qs
        # qs (..., N_l, D)
        r2s = qs.square().sum(axis=-1) # (..., N_l)
        logp = 0
        for q, r2 in zip(qs.unbind(-2), r2s.unbind(-1)):
            p, logp = MoebiusLayer.transform_(p, q), logp - MoebiusLayer.logdet(p, q, r2)
        return p, logp
    
    def n_features(config):
        return config.norm_layers * 4
