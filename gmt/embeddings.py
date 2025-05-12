import math
import numpy as np
from numba import njit, prange
from .utils import compute_curvature, resample_curve

@njit(fastmath=True)
def ray_segment_intersect(px, py, dx, dy, x1, y1, x2, y2, R_max):
    v1x, v1y = px-x1, py-y1
    v2x, v2y = x2-x1, y2-y1
    v3x, v3y = -dy, dx
    denom = v2x*v3x + v2y*v3y
    if abs(denom) < 1e-6: return -1.0
    t1 = (v2x*v1y - v2y*v1x)/denom
    t2 = (v1x*v3x + v1y*v3y)/denom
    return t1 if (0.0 <= t1 <= R_max and 0.0 <= t2 <= 1.0) else -1.0

@njit(parallel=True, fastmath=True)
def compute_embeddings(bx, by, tx, ty, D, R_max, emb_out):
    N = tx.shape[0]
    for i in prange(N):
        px, py = tx[i], ty[i]
        for j in range(D):
            angle = 2*math.pi*j/D
            dx, dy = math.cos(angle), math.sin(angle)
            best = R_max
            for k in range(bx.shape[0]-1):
                d = ray_segment_intersect(px,py,dx,dy,bx[k],by[k],bx[k+1],by[k+1],R_max)
                if 0.0 <= d < best: best = d
            emb_out[i,3*j  ] = best
            emb_out[i,3*j+1] = angle
            emb_out[i,3*j+2] = 1.0 if best<R_max else 0.0
