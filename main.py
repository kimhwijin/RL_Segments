import argparse

from utils import plot

parser = argparse.ArgumentParser()
parser.add_argument('--seg_dist', type=str)
parser.add_argument('--weights', type=str)
parser.add_argument('--backbone', type=str)
parser.add_argument('--loss', type=str)
parser.add_argument('--dataset', type=str)


args = parser.parse_args()
import torch
weights = list(map(float, args.weights.split(',')))
print(weights)

if args.loss == 'reinforce':
    import reinforce
    reinforce.main(
        backbone=args.backbone, 
        weights=weights, 
        seg_dist=args.seg_dist,
        dataset=args.dataset,
    )
elif args.loss == 'ppo':
    import ppo
    ppo.main(
        backbone=args.backbone, 
        weights=weights, 
        seg_dist=args.seg_dist,
        dataset=args.dataset,
    )


plot.result_plots(
    loss=args.loss,
    backbone=args.backbone, 
    weights=weights, 
    seg_dist=args.seg_dist,
    dataset=args.dataset,
)

plot.result_plots(
    loss=args.loss,
    backbone=args.backbone, 
    weights=weights, 
    seg_dist=args.seg_dist,
    dataset=args.dataset,
)

plot.test_sample_plots(
    loss=args.loss,
    backbone=args.backbone, 
    weights=weights, 
    seg_dist=args.seg_dist,
    dataset=args.dataset,
)
