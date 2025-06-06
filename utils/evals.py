import sys
sys.path.append("/home/hjkim/RL_TimeSegment")

from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from utils import env

@torch.no_grad()
def valid_step(
    epoch,
    loader,
    policy_module,
    value_module,
    predictor,
    mask_fn,
    reward_fn,
    target_type,
    blackbox_model,
    num_classes,
    exp_dir,
    best_reward,
    logger,
    history,
    device,
):
    epoch_total  = 0
    avg_length   = 0.0
    avg_reward   = 0.0
    targets, trues, masked_preds = [], [], []

    for batch in loader:
        x = batch['x'].to(device)
        B = x.size(0)
        if target_type == 'blackbox':
            y_target = blackbox_model(x).softmax(-1)
        elif target_type == 'y':
            y_target = F.one_hot(batch['y'], num_classes=num_classes).float().to(device)

        y_true = batch['y'].to(device)
        
        td = TensorDict(
            {
                "x": x,
                "y": y_target,
                "curr_mask": torch.zeros_like(x, dtype=bool),
            },
            batch_size=(B,), device=device,
        )
        env.step(td, policy_module, reward_fn, mode=True)

        epoch_total  += B
        avg_length   += td["next", "curr_mask"].sum([1, 2], dtype=float).sum().item()
        avg_reward   += td["next", "reward"].sum().item()
        
        x_masked  = mask_fn(td["x"], td["next", "curr_mask"])
        y_masked  = predictor(x_masked).softmax(-1)
        y_pred = y_masked.argmax(-1)

        trues.append(y_true.cpu())
        targets.append(y_target.argmax(-1).cpu())
        masked_preds.append(y_pred.cpu())

    avg_length   /= epoch_total
    avg_reward   /= epoch_total

    trues        = torch.cat(trues).numpy()
    targets      = torch.cat(targets).numpy()
    masked_preds = torch.cat(masked_preds).numpy()

    og_acc     = accuracy_score(trues, targets)
    masked_acc = accuracy_score(targets, masked_preds)

    og_f1      = f1_score(trues, targets, average='binary' if num_classes == 2 else 'macro')
    masked_f1  = f1_score(targets, masked_preds, average='binary' if num_classes == 2 else 'macro')

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
            "predictor_state": predictor.state_dict(),
        }, f'{exp_dir}/checkpoints.pth')

    msg = f"\t| Avg Length: {avg_length:.4f}"\
        + f" | Avg Reward: {avg_reward:.4f}"\
        + f"\n\t| OG  Acc: {og_acc:.2f}  | Masked Acc: {masked_acc:.2f}"\
        + f" | OG  F1: {og_f1:.2f} | Masked F1: {masked_f1:.2f}"
    print(msg)
    logger.write(msg+"\n")
    history['valid_length'].append(avg_length)
    history['valid_reward'].append(avg_reward)
    history['og_acc'].append(og_acc)
    history['masked_acc'].append(masked_acc)
    history['og_f1'].append(og_f1)
    history['masked_f1'].append(masked_f1)

    return history


@torch.no_grad()
def collect_buffer_with_old_policy(
    replay_buffer,
    loader, 
    policy_module,
    value_module,
    advantage_module,
    target_type, 
    blackbox_model,
    rollout_len, 
    reward_fn,
    device
):
    collected = 0
    while True:
        for batch in loader:
            x = batch['x'].to(device)
            B = x.size(0)
            with torch.no_grad():
                if target_type == 'blackbox':
                    y_target = blackbox_model(x).softmax(-1)
                elif target_type == 'y':
                    y_target = batch['y'].to(device)
                td = TensorDict(
                    {
                        "x": x,
                        "y": y_target,
                        "curr_mask": torch.zeros_like(x, dtype=bool)
                    }, 
                    batch_size=(B,), device=device)

                env.step(td, policy_module, reward_fn)
                value_module(td);
                value_module(td['next'])
                advantage_module(td)
            replay_buffer.extend(td.view(-1).detach().cpu())
            collected += B
            if collected >= rollout_len:
                return replay_buffer