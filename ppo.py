from functools import partial
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


from tensordict import TensorDict
from tensordict.nn import TensorDictModule, InteractionType

from torchrl.objectives import ClipPPOLoss, ReinforceLoss, A2CLoss
from torchrl.objectives.value import GAE
from torchrl.data.replay_buffers import (
    TensorDictReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
)

from dataloader import SeqComb
from models import Policy, Value, Predictor
from rewards import Reward
from utils import env
from utils import masking
from utils import get_exp_name
from utils import evals

import os

def main(
    backbone,
    weights,
    seg_dist,
    dataset,
    target_type, # blackbox, y
    predictor_type, # blackbox, predictor
    predictor_pretrain, # blackbox, predictor
    mask_type, # seq, zero, normal,
    epochs,
    device,
):
    exp_name = get_exp_name('ppo', backbone, weights, seg_dist, dataset, target_type, predictor_type, predictor_pretrain, mask_type)
    exp_dir = f'./checkpoints/{exp_name}'
    os.makedirs(exp_dir, exist_ok=True)

    # Logger ------------------------------------------------------------------------------------
    logger_dir = f"{exp_dir}/log.log"
    with open(logger_dir, 'w') as f:
        pass

    # Setting ------------------------------------------------------------------------------------
    predictor_epochs    = 1
        
    rollout_len         = 4096+2048
    ppo_epochs          = 4
    batch_size          = 256

    d_in                = 1
    d_model             = 128
    seq_len             = 100
    
    # Dataset Setting ------------------------------------------------------------------------------------
    d_out, average = SeqComb.get_num_classes(dataset)

    train_set = SeqComb.get_SeqComv(dataset, 'TRAIN')
    valid_set = SeqComb.get_SeqComv(dataset, 'VALID')
    test_set = SeqComb.get_SeqComv(dataset, 'TEST')

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Policy, Value Network ------------------------------------------------------------------------------------
    policy_module = Policy.init_policy_module(
        d_in = d_in,
        d_model = d_model,
        seq_len = seq_len,
        backbone = backbone,
        seg_dist = seg_dist
    )
    value_module, advantage_module = Value.init_value_module(
        d_in = d_in,
        d_model = d_model,
        seq_len = seq_len,
        backbone = backbone,
    )
    policy_module = policy_module.to(device)
    policy_module.eval()
    value_module = value_module.to(device)
    value_module.eval()

    # Loss Module ------------------------------------------------------------------------------------
    loss_module = ClipPPOLoss(
        actor=policy_module,
        critic=value_module,
        clip_epsilon=0.2,
        entropy_bonus=True,
        entropy_coef=0.01,
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
    )

    # Optimizer Setting ------------------------------------------------------------------------------------
    policy_optim = torch.optim.Adam(policy_module.parameters(), lr=1e-4)
    value_optim = torch.optim.Adam(value_module.parameters(), lr=3e-4)
    policy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(policy_optim, epochs)
    value_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(value_optim, epochs)
    
    # Buffer Setting ------------------------------------------------------------------------------------
    storage = LazyTensorStorage(rollout_len, device=device)
    sampler = SamplerWithoutReplacement()
    replay_buffer = TensorDictReplayBuffer(storage=storage, sampler=sampler, batch_size=batch_size)
    replay_buffer.empty()
    collected = 0

    # Masking Type ------------------------------------------------------------------------------------
    mask_fn = masking.MaskingFunction(mask_type)

    # BlackBox Setting ------------------------------------------------------------------------------------
    blackbox_model = Predictor.PredictorNetwork(d_in=d_in, d_model=64, d_out=d_out, seq_len=seq_len, backbone='tcn')
    blackbox_model.load_state_dict(torch.load(f'./blackbox/best_{dataset}_tcn.pth')['model_state'])
    blackbox_model = blackbox_model.to(device)
    blackbox_model = blackbox_model.eval()

    # Predictor Setting ------------------------------------------------------------------------------------
    if predictor_type == 'blackbox':
        predictor = blackbox_model
    elif predictor_type == 'predictor':
        predictor = Predictor.PredictorNetwork(d_in=d_in, d_model=d_model, d_out=d_out, seq_len=seq_len, backbone=backbone)
        predictor = predictor.to(device)
        pred_optim = torch.optim.Adam(predictor.parameters(), lr=1e-3 if 'rnn' in backbone.lower() else 1e-4)

        if bool(predictor_pretrain):
            predictor.train()
            predictor_random_train(
                pre_train_epochs = predictor_pretrain,
                loader = train_loader,
                predictor = predictor,
                pred_optim = pred_optim,
                seq_len = seq_len,
                mask_fn = mask_fn,
                device = device,
            )
            predictor.eval()

    # Reward Setting ------------------------------------------------------------------------------------
    ce_reward_fn = partial(Reward.exp_minus_cross_entropy_reward, mask_fn=mask_fn, predictor=predictor)
    length_reward_fn = Reward.length_reward

    reward_fns = [ce_reward_fn, length_reward_fn]
    reward_fn = partial(Reward.compose_reward, reward_fns=reward_fns, weights=weights)

    # Init ------------------------------------------------------------------------------------
    history = defaultdict(list)
    num_update      = 0
    best_f1         = 0.
    best_reward     = 0.
    # Train Start ------------------------------------------------------------------------------------
    for epoch in range(epochs):
        msg = f"Epoch : {epoch}"
        print(msg)
        with open(logger_dir, 'a') as f:
            f.write(msg + '\n')
        
        # Collect ------------------------------------------------------------------------------------
        replay_buffer = evals.collect_buffer_with_old_policy( # torch no grad func
            replay_buffer = replay_buffer, 
            loader = train_loader, 
            policy_module = policy_module, 
            value_module = value_module,
            advantage_module = advantage_module,
            target_type = target_type, 
            blackbox_model = blackbox_model, 
            rollout_len = rollout_len, 
            reward_fn = reward_fn,
            device = device
        )
        # Predictor Train ------------------------------------------------------------------------------------
        if predictor_type == 'predictor':
            predictor.train()
            history = predictor_update(
                predictor_epochs = predictor_epochs,
                replay_buffer = replay_buffer,
                predictor = predictor,
                pred_optim = pred_optim,
                mask_fn = mask_fn,
                history = history,
                logger_dir = logger_dir,
                device = device,
            )
            predictor.eval()

        # PPO Train ------------------------------------------------------------------------------------
        policy_module.train(); value_module.train()
        history = ppo_update(
            ppo_epochs = ppo_epochs,
            replay_buffer = replay_buffer,
            policy_module = policy_module,
            value_module = value_module,
            policy_optim = policy_optim,
            policy_scheduler = policy_scheduler,
            value_optim = value_optim,
            value_scheduler = value_scheduler,
            history = history,
            logger_dir = logger_dir,
            device = device,
        )
        replay_buffer.empty()
        policy_module.eval(); value_module.eval()

        # Valid Step ------------------------------------------------------------------------------------
        history = evals.valid_step(
                epoch = epoch,
                loader = test_loader,
                policy_module = policy_module,
                value_module = value_module,
                predictor = predictor,
                mask_fn = mask_fn,
                reward_fn = reward_fn,
                target_type = target_type,
                blackbox_model = blackbox_model,
                num_classes = d_out,
                exp_dir = exp_dir,
                best_reward = best_reward,
                logger_dir = logger_dir,
                history = history,
                device = device,
        )
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    torch.save(history, f"{exp_dir}/history.pth")





def predictor_random_train(
    pre_train_epochs,
    loader,
    predictor,
    pred_optim,
    seq_len,
    mask_fn,
    device,
):  
    pbar = tqdm(range(pre_train_epochs))
    for _ in pbar:
        total_loss = 0.
        total = 0.
        for batch in loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            B = x.size(0)

            # Select Two index uniformly
            indices, _ = torch.multinomial(
                torch.tensor([1/seq_len]).repeat(B, seq_len), 2, replacement=True
            ).sort()

            start, end = indices[:, :1], indices[:, 1:]
            arange = torch.arange(seq_len).unsqueeze(0)
            mask = torch.logical_and(start <= arange, arange <= end).unsqueeze(-1).to(device)
            segments = mask_fn(x, mask)
            logits = predictor(segments)
            loss = F.cross_entropy(logits, y)

            pred_optim.zero_grad()
            loss.backward()
            pred_optim.step()

            total_loss += B * loss.item()
            total += B
        pbar.set_postfix(loss = round(total_loss / total, 4))

def predictor_update(
    predictor_epochs,
    replay_buffer,
    predictor,
    pred_optim,
    mask_fn,
    history,
    logger_dir,
    device
):
    epoch_total = 0.
    avg_pred_loss = 0.

    for _ in range(predictor_epochs):
        for mb in replay_buffer:
            x = mb['x'].to(device)
            y = mb['y'].to(device)
            B = x.size(0)
            with torch.no_grad():
                mask = torch.logical_or(mb['curr_mask'], mb['next', 'curr_mask'])
                masked_x = mask_fn(x, mask)
            logits = predictor(masked_x)

            loss = F.cross_entropy(logits, y)
            pred_optim.zero_grad()
            loss.backward()
            pred_optim.step()
        
            epoch_total += B
            avg_pred_loss += loss.item() * B
        avg_pred_loss /= epoch_total
    msg = f"\t| Avg Predictor Loss: {avg_pred_loss:.4f}"
    print(msg)
    with open(logger_dir, 'a') as f:
        f.write(msg + '\n')
    history['pred_loss'].append(avg_pred_loss)
    return history


def ppo_update(
    ppo_epochs,
    replay_buffer,
    policy_module,
    value_module,
    policy_optim,
    policy_scheduler,
    value_optim,
    value_scheduler,
    history,
    logger_dir,
    device,
):
    epoch_total = 0.
    avg_length = 0.
    avg_reward = 0.
    avg_actor_loss = 0.
    avg_critic_loss = 0.

    loss_module = ClipPPOLoss(
        actor=policy_module,
        critic=value_module,
        clip_epsilon=0.2,
        entropy_bonus=True,
        entropy_coef=0.01,
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
    )

    for _ in range(ppo_epochs):
        for mb in replay_buffer:
            B = mb.shape[0]
            loss_td = loss_module(mb.to(device))
            loss = loss_td["loss_objective"] + loss_td["loss_critic"] + loss_td["loss_entropy"]

            policy_optim.zero_grad()
            value_optim.zero_grad()
            loss.backward()
            policy_optim.step()
            value_optim.step()

            epoch_total += B
            avg_length += mb['next', 'curr_mask'].sum([1, 2], dtype=float).sum().item()
            avg_reward += mb['next', 'reward'].sum().item()
            avg_actor_loss += loss_td['loss_objective'] * B
            avg_critic_loss += loss_td['loss_critic'] * B
    policy_scheduler.step()
    value_scheduler.step()
    torch.cuda.empty_cache()

    avg_length /= epoch_total
    avg_reward /= epoch_total
    avg_actor_loss /= epoch_total
    avg_critic_loss /= epoch_total

    msg = f"\t| Avg Actor Loss: {avg_actor_loss:.4f} " \
        + f"| Avg Critic Loss: {avg_critic_loss:.4f} " \
        + f"\n\t| Avg Length: {avg_length:.4f} " \
        + f"| Avg Reward: {avg_reward:.4f}\n"
    print(msg, end='')
    with open(logger_dir, 'a') as f:
        f.write(msg)
    history['actor_loss'].append(avg_actor_loss)
    history['critic_loss'].append(avg_critic_loss)
    history['train_length'].append(avg_length)
    history['train_reward'].append(avg_reward)
    
    return history
