import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from gmt.dataset import ThreadedRayDataset
from gmt.model import ComplexCrossTransformer
from gmt.variants import define_variants, MODEL_NAME


def generate_mappings(preds, baselines, targets):
    """
    For each sample, compute the aligned points and return a list of mappings.
    Each mapping is a dict with keys: 'baseline', 'target', 'aligned'.
    """
    mappings = []
    for pred, b, t in zip(preds, baselines, targets):
        aligned = t + pred
        mappings.append({
            'baseline': b,      # shape (N,2)
            'target': t,        # shape (N,2)
            'aligned': aligned  # shape (N,2)
        })
    return mappings


class ExternalGeometryDataset(Dataset):
    """
    Load external geometries from NumPy .npz files containing arrays
    'baseline' and 'target', both of shape (N,2).
    """
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.baselines = data['baseline']  # shape (M, N, 2)
        self.targets   = data['target']    # shape (M, N, 2)
        assert self.baselines.shape == self.targets.shape, \
            "baseline and target arrays must match shapes"

    def __len__(self):
        return len(self.baselines)

    def __getitem__(self, idx):
        b_r = self.baselines[idx]
        t_r = self.targets[idx]
        return b_r, t_r


def load_model(variant, device, feat_dim, checkpoint_path=None):
    model = ComplexCrossTransformer(
        tgt_dim=feat_dim,
        base_dim=3,
        variant=variant
    ).to(device)
    ckpt = checkpoint_path or f"{MODEL_NAME}_{variant}.pth"
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def infer(model, data_loader, device):
    preds = []
    baselines = []
    targets = []
    with torch.no_grad():
        for item in data_loader:
            if isinstance(item, tuple) and len(item)==5:
                tgt, base, lab, b_r, t_r = item
            else:
                b_r, t_r = item
                from gmt.dataset import prepare_infer_features
                tgt, base = prepare_infer_features(b_r, t_r)
                tgt = torch.from_numpy(tgt).unsqueeze(0)
                base = torch.from_numpy(base).unsqueeze(0)
                lab = None
            tgt = tgt.to(device)
            base = base.to(device)
            pred = model(tgt, base).cpu().numpy()  # (B, N, 2)
            preds.extend(pred)
            baselines.extend(b_r if isinstance(b_r, list) else b_r.numpy() if isinstance(b_r, torch.Tensor) else b_r)
            targets.extend(t_r if isinstance(t_r, list) else t_r.numpy() if isinstance(t_r, torch.Tensor) else t_r)
    return preds, baselines, targets


def plot_sample(mapping, variant, idx):
    b = mapping['baseline']
    t = mapping['target']
    a = mapping['aligned']

    plt.figure(figsize=(6,6))
    plt.plot(b[:,0], b[:,1], '-o', label='Baseline')
    plt.plot(t[:,0], t[:,1], '--x', alpha=0.5, label='Original')
    plt.plot(a[:,0], a[:,1], '-s', alpha=0.7, label='Aligned')
    plt.axis('equal')
    plt.legend()
    plt.title(f"{MODEL_NAME}-{variant} Sample {idx} Mapping")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Run inference and extract geometry mappings with GMT model.")
    parser.add_argument('--variant', choices=define_variants.keys(), default='medium')
    parser.add_argument('--samples', type=int, default=1, help='Number of samples to infer and plot')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--ckpt', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--save', action='store_true', help='Save mappings to .npz file')
    parser.add_argument('--external', type=str, default=None,
                        help='Path to external .npz with baseline/target arrays')
    args = parser.parse_args()

    device = torch.device('mps') if torch.backends.mps.is_available() else \
             torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    if args.external:
        ds = ExternalGeometryDataset(args.external)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        from gmt.dataset import prepare_infer_features
        dummy_b, dummy_t = ds[0]
        dummy_tgt, dummy_base = prepare_infer_features(dummy_b, dummy_t)
        feat_dim = dummy_tgt.shape[-1]
    else:
        ds = ThreadedRayDataset(num_samples=100, max_workers=4)
        feat_dim = ds.tgt_feats.shape[-1]
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = load_model(args.variant, device, feat_dim, args.ckpt)

    preds, baselines, targets = infer(model, loader, device)

    mappings = generate_mappings(preds, baselines, targets)

    if args.save:
        aligned_arr = np.stack([m['aligned'] for m in mappings])
        baseline_arr = np.stack([m['baseline'] for m in mappings])
        target_arr   = np.stack([m['target'] for m in mappings])
        np.savez(f"mappings_{args.variant}.npz",
                 baseline=baseline_arr,
                 target=target_arr,
                 aligned=aligned_arr)
        print(f"Mappings saved to mappings_{args.variant}.npz")

    for i in range(min(args.samples, len(mappings))):
        plot_sample(mappings[i], args.variant, i)

if __name__ == '__main__':
    main()
