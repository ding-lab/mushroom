import torch
import torch.nn.functional as F
from einops import rearrange


class ZinbReconstructor(torch.nn.Module):
    def __init__(self, in_dim, n_genes, n_metagenes=20):
        super().__init__()
        self.in_dim = in_dim
        self.n_genes = n_genes
        self.n_metagenes = n_metagenes

        self.metagenes = torch.nn.Parameter(torch.rand(self.n_metagenes, self.n_genes))
        self.scale_factors = torch.nn.Parameter(torch.rand(self.n_genes))
        self.p = torch.nn.Parameter(torch.rand(self.n_genes))

        self.to_meta = torch.nn.Linear(in_dim, self.n_metagenes)


    def zinb_reconstruction(self, emb):
        x = self.to_meta(emb) # (b n d) -> (b n m)
        
        r = x @ self.metagenes
        r = r * self.scale_factors
        r = F.softplus(r)
        
        p = torch.sigmoid(self.p)
        p = rearrange(p, 'g -> 1 1 g')
            
        r += 1e-8
            
        nb = torch.distributions.NegativeBinomial(r, p)
        
        return {
            'r': r,
            'p': p,
            'exp': nb.mean,
            'nb': nb,
            'metagene_activity': x # (b v m)
        }
    
    def forward(self, x):
        return self.zinb_reconstruction(x)