import numpy as np
import math
from scipy.interpolate import interp1d

def compute_curvature(pts):
    dx = np.gradient(pts[:,0])
    dy = np.gradient(pts[:,1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    num = np.abs(dx * ddy - dy * ddx)
    den = (dx*dx + dy*dy)**1.5 + 1e-6
    return (num/den).astype(np.float32)

def generate_sinusoid(num_points=200, amplitude=None, frequency=None, x_range=(0,1)):
    amplitude = float(amplitude or np.random.uniform(0.3,1.0))
    frequency = float(frequency or np.random.uniform(0.5,3.0))
    xs = np.linspace(x_range[0], x_range[1], num_points, dtype=np.float32)
    ys = (amplitude * np.sin(2*math.pi*frequency*xs)).astype(np.float32)
    return np.stack((xs, ys), axis=1)

def generate_circle(num_points=200, radius=None, center=None):
    radius = float(radius or np.random.uniform(0.3,1.0))
    center = center or (np.random.uniform(0.5,4.5), np.random.uniform(0.5,4.5))
    angles = np.linspace(0,2*math.pi,num_points,endpoint=False,dtype=np.float32)
    xs = (center[0]+radius*np.cos(angles)).astype(np.float32)
    ys = (center[1]+radius*np.sin(angles)).astype(np.float32)
    return np.stack((xs, ys), axis=1)

def generate_ellipse(num_points=200, a=None, b=None, center=None):
    a = float(a or np.random.uniform(0.3,1.2))
    b = float(b or np.random.uniform(0.3,1.2))
    center = center or (np.random.uniform(0.5,4.5), np.random.uniform(0.5,4.5))
    angles = np.linspace(0,2*math.pi,num_points,endpoint=False,dtype=np.float32)
    xs = (center[0]+a*np.cos(angles)).astype(np.float32)
    ys = (center[1]+b*np.sin(angles)).astype(np.float32)
    return np.stack((xs, ys), axis=1)

def generate_random_polyline(num_vertices=5, x_range=(0,5), y_range=(0,5)):
    xs = np.random.uniform(*x_range, size=num_vertices).astype(np.float32)
    ys = np.random.uniform(*y_range, size=num_vertices).astype(np.float32)
    order = np.argsort(xs)
    return np.stack((xs[order], ys[order]), axis=1)

def displace_curve(pts, sigma=0.3):
    return (pts + np.random.normal(scale=sigma, size=pts.shape)).astype(np.float32)

def resample_curve(pts, N=256):
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1).astype(np.float32)
    cum = np.concatenate(([0.0], np.cumsum(d))).astype(np.float32)
    ts = np.linspace(0.0, cum[-1], N, dtype=np.float32)
    fx = interp1d(cum, pts[:,0].astype(np.float32), kind='cubic')
    fy = interp1d(cum, pts[:,1].astype(np.float32), kind='cubic')
    xs, ys = fx(ts).astype(np.float32), fy(ts).astype(np.float32)
    return np.stack((xs, ys), axis=1)