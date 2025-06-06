from functools import partial
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F


from tensordict import TensorDict
from tensordict.nn import TensorDictModule, InteractionType

from torchrl.data import BoundedTensorSpec
from torchrl.modules import ProbabilisticActor, ValueOperator

from torchrl.objectives import ClipPPOLoss, ReinforceLoss, A2CLoss
from torchrl.objectives.value import GAE

from dataloader import SeqComb
from models import Policy, Value, Predictor
from distributions import *
from utils import env
from rewards import Reward
from utils import masking, get_exp_name

import os

def main(
    backbone, 
    weights, 
    seg_dist,
    dataset,
):
    exp_name = get_exp_name('reinforce', backbone, weights, seg_dist, dataset)
    exp_dir = f'./checkpoints/{exp_name}'
    os.makedirs(exp_dir, exist_ok=True)

    if seg_dist == 'cat_nb':
        d_start, d_end = 100, 2
        SegmentDistribution = CategoricalToNegativeBinomial
    elif seg_dist == 'cat_cat':
        d_start, d_end = 100, 100
        SegmentDistribution = CategoricalToCategorical
    elif seg_dist == 'nb_nb':
        d_start, d_end = 2, 2
        SegmentDistribution = NegativeBinomialToNegativeBinomial

    d_in = 1
    d_model = 128
    d_out, average = SeqComb.get_num_classes(dataset)

    seq_len = 100

    batch_size = 256
    epochs = 250
    device = 'cuda:4'
    train_set = SeqComb.get_SeqComv(dataset, 'TRAIN')
    valid_set = SeqComb.get_SeqComv(dataset, 'VALID')
    test_set = SeqComb.get_SeqComv(dataset, 'TEST')


    policy_net = Policy.PolicyNetwork(
        d_in = d_in+1,
        d_model = d_model,
        d_start = d_start,
        d_end = d_end,
        seq_len = seq_len,
        backbone=backbone
    )

    policy_module = TensorDictModule(
        policy_net, 
        in_keys=['x', 'curr_mask'], 
        out_keys=['start_logits', 'end_logits']
    )

    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=BoundedTensorSpec(low=0, high=seq_len-1, dtype=int, shape=(2,)),
        in_keys=["start_logits", "end_logits"],
        distribution_class=SegmentDistribution,
        distribution_kwargs={"seq_len": seq_len},
        return_log_prob=True,
        default_interaction_type=InteractionType.RANDOM
    )

    value_net = Value.ValueNetwork(
        d_in = d_in+1, 
        d_model = d_model,
        d_out = 1,
        seq_len = seq_len,
        backbone=backbone
    )

    value_module = ValueOperator(
        module=value_net,
        in_keys=["x", 'curr_mask'],
    )
    policy_module = policy_module.to(device)
    value_module = value_module.to(device)
    advantage_module = GAE(
        value_network=None,
        gamma = 0.99, lmbda=0.95,average_gae=True
    )
    loss_module = A2CLoss(policy_module, value_module)

    policy_optim = torch.optim.Adam(policy_module.parameters(), lr=1e-4)
    value_optim = torch.optim.Adam(value_module.parameters(), lr=3e-4)
    policy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(policy_optim, epochs)
    value_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(value_optim, epochs)


    blackbox_model = Predictor.PredictorNetwork(d_in=1, d_model=128, d_out=2, seq_len=seq_len, backbone='tcn')
    blackbox_model.load_state_dict(torch.load('./blackbox/best_tcn.pth')['model_state'])
    blackbox_model = blackbox_model.to(device)
    blackbox_model = blackbox_model.eval()

    mask_fn = masking.SeqCombMask()

    ce_reward_fn = partial(Reward.exp_minus_cross_entropy_reward, mask_fn=mask_fn, predictor=blackbox_model)
    length_reward_fn = Reward.length_reward

    # weights = [0.7, 0.3]
    reward_fns = [ce_reward_fn, length_reward_fn]
    reward_fn = partial(Reward.compose_reward, reward_fns=reward_fns, weights=weights)


    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    logger = open(f"{exp_dir}/{exp_name}.log", 'w')

    best_reward = 0.

    for epoch in range(epochs):
        msg = f"Epoch {epoch:02d}/{epochs} "
        # Train --------------------------------------------
        epoch_total = 0.0
        avg_length = 0.
        avg_reward = 0.
        avg_actor_loss = 0.
        avg_critic_loss = 0.
        
        policy_module.train()
        for batch in train_loader:
            x = batch['x'].to(device)
            B = x.shape[0]
            with torch.no_grad():
                y_blackbox = blackbox_model(x).softmax(-1)
            tensordict = TensorDict(
                {
                    "x": x,
                    "y": y_blackbox,
                    "curr_mask": torch.zeros_like(x, dtype=bool)
                }, 
                batch_size=(B,), device=device)
            
            
            env.step(tensordict, policy_module, reward_fn)

            value_module(tensordict)
            value_module(tensordict['next'])
            advantage_module(tensordict)

            loss_td = loss_module(tensordict)
            loss = loss_td['loss_objective'] + loss_td['loss_critic']
            
            policy_optim.zero_grad()
            value_optim.zero_grad()
            loss.backward()
            policy_optim.step()
            value_optim.step()

            epoch_total += B
            avg_length += tensordict['next', 'curr_mask'].sum([1, 2], dtype=float).sum().item()
            avg_reward += tensordict['next', 'reward'].sum().item()
            avg_actor_loss += loss_td['loss_objective'] * B
            avg_critic_loss += loss_td['loss_critic'] * B

        policy_scheduler.step()
        value_scheduler.step()

        avg_length /= epoch_total
        avg_reward /= epoch_total
        avg_actor_loss /= epoch_total
        avg_critic_loss /= epoch_total
        msg += f"\n\t| Avg Actor Loss: {avg_actor_loss:.4f} " \
            + f"| Avg Critic Loss: {avg_critic_loss:.4f} " \
            + f"\n\t| Avg Length: {avg_length:.4f} " \
            + f"| Avg Reward: {avg_reward:.4f} "

        # # Valid --------------------------------------------
        epoch_total = 0.0
        avg_length = 0.
        avg_reward = 0.
        targets, og_preds, masked_preds = [], [], []

        policy_module.eval()
        with torch.no_grad():
            for batch in test_loader:
                x = batch['x'].to(device)
                B = x.shape[0]
                y_blackbox = blackbox_model(x).softmax(-1)
                tensordict = TensorDict(
                    {
                        "x": x,
                        "y": y_blackbox,
                        "curr_mask": torch.zeros_like(x, dtype=bool)
                    }, 
                    batch_size=(B,), device=device)
                env.step(tensordict, policy_module, reward_fn, mode=True)
                
                epoch_total  += B
                avg_length   += tensordict["next", "curr_mask"].sum([1, 2], dtype=float).sum().item()
                avg_reward   += tensordict["next", "reward"].sum().item()
                
                x = tensordict['x']
                curr_mask = tensordict['next', 'curr_mask']

                x_masked = mask_fn(x, curr_mask)
                y_masked = blackbox_model(x_masked).softmax(-1)
                targets.append(batch['y'].cpu())
                og_preds.append(y_blackbox.argmax(-1).cpu())
                masked_preds.append(y_masked.argmax(-1).cpu())

            avg_length   /= epoch_total
            avg_reward   /= epoch_total

            targets = torch.cat(targets).numpy()
            og_preds = torch.cat(og_preds).numpy()
            masked_preds = torch.cat(masked_preds).numpy()
        
        og_acc = accuracy_score(targets, og_preds)
        masked_acc = accuracy_score(targets, masked_preds)
        og_f1 = f1_score(targets, og_preds, average=average)
        masked_f1 = f1_score(targets, masked_preds, average=average)
        
        if avg_reward >= best_reward:
            print(f"Best : {avg_reward:.2f}")
            best_reward = avg_reward
            torch.save({
                "epoch": epoch,
                'acc': masked_acc,
                "f1": masked_f1,
                "reward": avg_reward,
                "length": avg_length,
                "policy_state": policy_module.state_dict(),
                "value_state": value_module.state_dict(),
            }, f'{exp_dir}/{exp_name}.pth')

        msg+= f"\n\t| Avg Length: {avg_length:.4f}" \
            + f" | Avg Reward: {avg_reward:.4f}" \
            + f"\n\t| OG  Acc: {og_acc:.2f}  | Masked Acc: {masked_acc:.2f}" \
            + f" | OG  F1: {og_f1:.2f} | Masked F1: {masked_f1:.2f}"
        print(msg)
        logger.write(msg+"\n")
        
    logger.close()