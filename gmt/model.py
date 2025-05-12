import torch.nn as nn
from .variants import define_variants

class TransformerBlock(nn.Module):
    def __init__(self,d_model,nhead,ff_dim,dropout):
        super().__init__()
        self.sa = nn.MultiheadAttention(d_model,nhead,dropout=dropout,batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model,ff_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim,d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self,x):
        res = x
        x,_ = self.sa(x,x,x)
        x    = self.norm1(x+res)
        res2 = x
        x    = self.ff(x)
        return self.norm2(x+res2)

class CrossTransformer(nn.Module):
    def __init__(self, tgt_dim, base_dim, variant='medium'):
        super().__init__()
        cfg = define_variants[variant]
        d_model,nl,ff_dim,drop = cfg['d_model'],cfg['num_layers'],cfg['ff_dim'],cfg['dropout']
        heads = cfg.get('nhead',8)

        self.proj_tgt  = nn.Linear(tgt_dim, d_model)
        self.proj_base = nn.Linear(base_dim,d_model)
        self.t_blocks  = nn.ModuleList([TransformerBlock(d_model,heads,ff_dim,drop) for _ in range(nl)])
        self.b_blocks  = nn.ModuleList([TransformerBlock(d_model,heads,ff_dim,drop) for _ in range(nl)])
        self.cross_sa  = nn.MultiheadAttention(d_model,heads,dropout=drop,batch_first=True)
        self.cross_ln  = nn.LayerNorm(d_model)
        self.head      = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model,ff_dim), nn.ReLU(),
            nn.Linear(ff_dim,2)
        )

    def forward(self, tgt, base):
        t = self.proj_tgt(tgt)
        b = self.proj_base(base)
        for blk in self.t_blocks: t = blk(t)
        for blk in self.b_blocks: b = blk(b)
        res,_ = self.cross_sa(t,b,b)
        t = self.cross_ln(res + t)
        return self.head(t)