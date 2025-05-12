import numpy as np
import torch
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from .utils import resample_curve, displace_curve, generate_sinusoid, generate_circle, generate_ellipse, generate_random_polyline
from .embeddings import compute_embeddings
from .utils import compute_curvature

generators = [generate_sinusoid, generate_circle, generate_ellipse, generate_random_polyline]

def generate_sample(_):
    gen = np.random.choice(generators)
    b = gen(200)
    t = displace_curve(b, 0.35)
    return b, t

class ThreadedRayDataset(Dataset):
    def __init__(self, num_samples=10_000, N=256, D=64, R_max=5.0, max_workers=8):
        self.N, self.D, self.R_max = N, D, R_max
        with ThreadPoolExecutor(max_workers=max_workers) as exec:
            baselines, targets = zip(*exec.map(generate_sample, range(num_samples)))

        ray_dim = 3*D; feat_dim = ray_dim + 4
        self.tgt_feats  = torch.zeros((num_samples, N, feat_dim))
        self.base_feats = torch.zeros((num_samples, N, 3))
        self.labels     = torch.zeros((num_samples, N, 2))
        self.b_res      = torch.zeros((num_samples, N, 2))
        self.t_res      = torch.zeros((num_samples, N, 2))
        idxs = torch.linspace(0,1,N).unsqueeze(1)

        for i,(b,t) in enumerate(zip(baselines,targets)):
            b_r = resample_curve(b,N); t_r = resample_curve(t,N)
            curv = compute_curvature(t_r); dif = b_r - t_r
            raw = np.zeros((N,ray_dim),dtype=np.float32)
            compute_embeddings(b_r[:,0],b_r[:,1],t_r[:,0],t_r[:,1],D,R_max,raw)

            self.tgt_feats[i]  = torch.cat([
                torch.from_numpy(raw), idxs, torch.from_numpy(curv).unsqueeze(1), torch.from_numpy(dif)
            ], dim=1)
            self.base_feats[i] = torch.cat([torch.from_numpy(b_r), idxs], dim=1)
            self.labels[i]     = torch.from_numpy(dif)
            self.b_res[i]      = torch.from_numpy(b_r)
            self.t_res[i]      = torch.from_numpy(t_r)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return (self.tgt_feats[idx], self.base_feats[idx],
                self.labels[idx], self.b_res[idx], self.t_res[idx])