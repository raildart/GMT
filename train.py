import argparse
import torch
from torch.utils.data import DataLoader
from gmt.dataset import ThreadedRayDataset
from gmt.model import CrossTransformer
from gmt.trainer import train as train_model
from gmt.variants import define_variants


def main():
    parser = argparse.ArgumentParser(description="Train GMT variants on synthetic or external dataset.")
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of dataset samples to generate')
    parser.add_argument('--max_workers', type=int, default=16, help='Threads for data generation')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--variants', nargs='+', default=list(define_variants.keys()),
                        help='List of variants to train')
    args = parser.parse_args()

    ds = ThreadedRayDataset(num_samples=args.num_samples,
                             max_workers=args.max_workers)
    feat_dim = ds.tgt_feats.shape[-1]

    device = torch.device('mps') if torch.backends.mps.is_available() else (
             torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Using device: {device}")

    for var in args.variants:
        if var not in define_variants:
            print(f"Warning: variant '{var}' not recognized. Skipping.")
            continue
        print(f"=== Training {var} ===")
        model = CrossTransformer(tgt_dim=feat_dim,
                                 base_dim=3,
                                 variant=var)
        train_model(ds, model,
                    variant=var,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr)

if __name__ == '__main__':
    main()
