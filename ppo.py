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
from torchrl.data.replay_buffers import (
    TensorDictReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
)

from dataloader import SeqComb
from models import Policy, Value, Predictor
from distributions import CategoricalToNegativeBinomial, CategoricalToCategorical, NegativeBinomialToNegativeBinomial
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
    exp_name = get_exp_name('ppo', backbone, weights, seg_dist, dataset)
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
        
    device          = 'cuda:4'
    epochs          = 250
    rollout_len     = 4096+2048
    ppo_epochs      = 4
    batch_size      = 256
    clip_eps        = 0.2
    entropy_coef    = 0.01
    value_coef      = 0.5

    d_in = 1
    d_model = 128
    d_out, average = SeqComb.get_num_classes(dataset)
    seq_len = 100

    train_set = SeqComb.get_SeqComv(dataset, 'TRAIN')
    valid_set = SeqComb.get_SeqComv(dataset, 'VALID')
    test_set = SeqComb.get_SeqComv(dataset, 'TEST')


    policy_net = Policy.PolicyNetwork(
        d_in = d_in+1,
        d_model = d_model,
        d_start = d_start,
        d_end = d_end,
        seq_len = seq_len,
        backbone = backbone
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
        backbone = backbone
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

    loss_module = ClipPPOLoss(
        actor=policy_module,
        critic=value_module,
        clip_epsilon=clip_eps,
        entropy_bonus=bool(entropy_coef),
        entropy_coef=entropy_coef,
        critic_coef=1.0,
        loss_critic_type="smooth_l1",

    )

    policy_optim = torch.optim.Adam(policy_module.parameters(), lr=1e-4)
    value_optim = torch.optim.Adam(value_module.parameters(), lr=3e-4)
    policy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(policy_optim, epochs)
    value_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(value_optim, epochs)

    storage = LazyTensorStorage(rollout_len, device=device)
    sampler = SamplerWithoutReplacement()
    replay_buffer = TensorDictReplayBuffer(storage=storage, sampler=sampler, batch_size=batch_size)
    replay_buffer.empty()
    collected = 0

    blackbox_model = Predictor.PredictorNetwork(d_in=1, d_model=64, d_out=d_out, seq_len=seq_len, backbone='tcn')
    blackbox_model.load_state_dict(torch.load(f'./blackbox/best_{dataset}_tcn.pth')['model_state'])
    blackbox_model = blackbox_model.to(device)
    blackbox_model = blackbox_model.eval()


    mask_fn = masking.SeqCombMask()

    ce_reward_fn = partial(Reward.exp_minus_cross_entropy_reward, mask_fn=mask_fn, predictor=blackbox_model)
    length_reward_fn = Reward.length_reward

    reward_fns = [ce_reward_fn, length_reward_fn]
    reward_fn = partial(Reward.compose_reward, reward_fns=reward_fns, weights=weights)



    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)


    logger = open(f"{exp_dir}/{exp_name}.log", 'w')
    num_update = 0

    best_f1 = 0.
    best_reward = 0.
    for epoch in range(epochs):
        # Collect    
        for batch in train_loader:
            x = batch['x'].to(device)
            B = x.size(0)
            # ---------------------------------------------------------------
            with torch.no_grad():
                y_blackbox = blackbox_model(x).softmax(-1)
                td = TensorDict(
                    {
                        "x": x,
                        "y": y_blackbox,
                        "curr_mask": torch.zeros_like(x, dtype=bool)
                    }, 
                    batch_size=(B,), device=device)
                env.step(td, policy_module, reward_fn)
                value_module(td);
                value_module(td['next'])
                advantage_module(td)

            # ---------------------------------------------------------------
            replay_buffer.extend(td.view(-1).detach().cpu())

            collected += B
            if collected >= rollout_len:

                num_update += 1
                epoch_total = 0.0
                avg_length = 0.
                avg_reward = 0.
                avg_actor_loss = 0.
                avg_critic_loss = 0.

                policy_module.train()
                value_module.train()

                for _ in range(ppo_epochs):
                    for mb in replay_buffer:                    
                        loss_td = loss_module(mb.to(device))
                        loss = (
                            loss_td["loss_objective"]
                            + loss_td["loss_critic"]
                            + loss_td["loss_entropy"]
                        )
                        policy_optim.zero_grad()
                        value_optim.zero_grad()
                        loss.backward()
                        policy_optim.step()
                        value_optim.step()

                        epoch_total += B
                        avg_length += td['next', 'curr_mask'].sum([1, 2], dtype=float).sum().item()
                        avg_reward += td['next', 'reward'].sum().item()
                        avg_actor_loss += loss_td['loss_objective'] * B
                        avg_critic_loss += loss_td['loss_critic'] * B
                        
                policy_scheduler.step()
                value_scheduler.step()
                replay_buffer.empty()
                collected = 0
                torch.cuda.empty_cache()


                avg_length /= epoch_total
                avg_reward /= epoch_total
                avg_actor_loss /= epoch_total
                avg_critic_loss /= epoch_total
                msg = f"Epoch {num_update:02d}" \
                    + f"\n\t| Avg Actor Loss: {avg_actor_loss:.4f} " \
                    + f"| Avg Critic Loss: {avg_critic_loss:.4f} " \
                    + f"\n\t| Avg Length: {avg_length:.4f} " \
                    + f"| Avg Reward: {avg_reward:.4f} "
                print(msg)
                logger.write(msg+"\n")

                # # Valid --------------------------------------------
                policy_module.eval()
                value_module.eval()

                epoch_total  = 0
                avg_length   = 0.0
                avg_reward   = 0.0
                targets, og_preds, masked_preds = [], [], []

                with torch.no_grad():
                    for batch in test_loader:
                        x = batch["x"].to(device)
                        B = x.size(0)
                        y_blackbox = blackbox_model(x).softmax(-1)

                        td = TensorDict(
                            {
                                "x": x,
                                "y": y_blackbox,
                                "curr_mask": torch.zeros_like(x, dtype=bool),
                            },
                            batch_size=(B,), device=device,
                        )

                        # deterministic rollout ────────────────────────────────────────
                        env.step(td, policy_module, reward_fn, mode=True)

                        # 집계용 변수 ­─────────────────────────────────────────────────
                        epoch_total  += B
                        avg_length   += td["next", "curr_mask"].sum([1, 2], dtype=float).sum().item()
                        avg_reward   += td["next", "reward"].sum().item()

                        x_masked  = mask_fn(td["x"], td["next", "curr_mask"])
                        y_masked  = blackbox_model(x_masked).softmax(-1)

                        targets.append(batch["y"].cpu())
                        og_preds.append(y_blackbox.argmax(-1).cpu())
                        masked_preds.append(y_masked.argmax(-1).cpu())

                # --- 스칼라 메트릭 계산 ------------------------------------------------
                avg_length   /= epoch_total
                avg_reward   /= epoch_total

                targets      = torch.cat(targets).numpy()
                og_preds     = torch.cat(og_preds).numpy()
                masked_preds = torch.cat(masked_preds).numpy()

                og_acc     = accuracy_score(targets, og_preds)
                masked_acc = accuracy_score(targets, masked_preds)
                og_f1      = f1_score(targets, og_preds, average=average)
                masked_f1  = f1_score(targets, masked_preds, average=average)


                if avg_reward >= best_reward:
                    print(f"Best : {avg_reward:.2f}")
                    best_reward = avg_reward
                    torch.save({
                        "epoch": epoch,
                        'acc': masked_acc,
                        "f1": masked_f1,
                        "length": avg_length,
                        "reward": avg_reward,
                        "policy_state": policy_module.state_dict(),
                        "value_state": value_module.state_dict(),
                    }, f'{exp_dir}/{exp_name}.pth')

                msg = f"\t| Avg Length: {avg_length:.4f}"\
                    + f" | Avg Reward: {avg_reward:.4f}"\
                    + f"\n\t| OG  Acc: {og_acc:.2f}  | Masked Acc: {masked_acc:.2f}"\
                    + f" | OG  F1: {og_f1:.2f} | Masked F1: {masked_f1:.2f}"
                print(msg)
                logger.write(msg+"\n")
    logger.close()