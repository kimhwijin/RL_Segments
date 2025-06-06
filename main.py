import argparse

from utils import plot

parser = argparse.ArgumentParser()
parser.add_argument('--seg_dist', type=str, choices=['cat_cat', 'cat_nb', 'nb_nb'])
parser.add_argument('--weights', type=str)
parser.add_argument('--backbone', type=str, choices=['tcn', 'rnn'])
parser.add_argument('--loss', type=str, choices=['reinforce', 'ppo'])
parser.add_argument('--dataset', type=str, choices=['one', 'onetwo'])
parser.add_argument('--target_type', type=str, choices=['blackbox', 'y'])
parser.add_argument('--predictor_type', type=str, choices=['predictor', 'blackbox'])
parser.add_argument('--mask_type', type=str, choices=['seq', 'zero'])
parser.add_argument("--predictor_pretrain", type=int)

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
        target_type=args.target_type,
        predictor_type=args.predictor_type,
        predictor_pretrain=args.predictor_pretrain,
        mask_type=args.mask_type,
    )

plot.result_plots(
    loss=args.loss,
    backbone=args.backbone, 
    weights=weights, 
    seg_dist=args.seg_dist,
    dataset=args.dataset,
    target_type=args.target_type,
    predictor_type=args.predictor_type,
    predictor_pretrain=args.predictor_pretrain,
    mask_type=args.mask_type,
)

plot.test_sample_plots(
    loss=args.loss,
    backbone=args.backbone, 
    weights=weights, 
    seg_dist=args.seg_dist,
    dataset=args.dataset,
    target_type=args.target_type,
    predictor_type=args.predictor_type,
    predictor_pretrain=args.predictor_pretrain,
    mask_type=args.mask_type,
)
