import torch
from torch import optim, nn
from einops import rearrange
from torch.func import grad, jacrev
import torch.nn.functional as F
import numpy as np

proj = lambda u, v: ((u * v).sum(axis=-1) / (u.square().sum(axis=-1)))[..., None] * u
op = lambda p, q: (p - 2 * proj((p - q), p))

def reparam(x, sig=5):
    return 0.99 / (1 + x)

softsign = lambda x, b: (b + torch.cosh(x)).log()
## numerically stable version of torch.sinh(x) / (b + torch.cosh(x))
softsign_d = lambda x, b: (1 - (-2*x.abs()).exp()) / (2 * b * (-x.abs()).exp() + 1 + (-2*x.abs()).exp()) * torch.tanh(1000 * x)
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
    # If naively use first three unit vectors, vector nearly lies in that space -> numerical instability
    
    indices = v.abs().argsort(dim=-1)
    
    # Gram-Schmidt to get orthogonal space
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
        Js = torch.vmap(jacrev(ConvexGradientLayer.transform_))(p, *tup) # (B, 4, 4)
        d = torch.einsum('b l i, b i j, b k j -> b l k', ep, Js, eres)
        d = (torch.cross(d[..., 0], d[..., 1], dim=-1) * d[..., 2]).sum(axis=-1).abs()
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

    def sample_mode(params, num_samples=200, k=2, bandwidth=50):
        bs = params[0].shape[0]
        r = torch.randn(num_samples, bs, 4).to(params[0]) 
        r /= r.norm(dim=-1, keepdim=True)   
        r_new, p_x = torch.vmap(ConvexGradientLayer.transform, in_dims=(0, None))(r, params)
        p_new = (p_x - p_x.amax(dim=0)).exp()#.cpu()
        return r_new[p_new.argmax(dim=0), torch.arange(bs)].reshape(-1, 4), p_x[p_new.argmax(dim=0), torch.arange(bs)].reshape(-1)
            
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
