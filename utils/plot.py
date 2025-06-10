import sys
sys.path.append("/home/hjkim/RL_TimeSegment")

import re
import matplotlib.pyplot as plt


import re
import matplotlib.pyplot as plt
from functools import partial
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import scipy.special as S

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensordict import TensorDict
from tensordict.nn import TensorDictModule, InteractionType

from torchrl.data import BoundedTensorSpec
from torchrl.modules import ProbabilisticActor, ValueOperator


from dataloader import SeqComb
from models import Policy, Value, Predictor
from distributions import CategoricalToNegativeBinomial, CategoricalToCategorical, NegativeBinomialToNegativeBinomial
from utils import get_exp_name, masking, env
from rewards import Reward




def test_sample_plots(
    loss,
    backbone,
    weights,
    seg_dist,
    dataset,
    target_type,
    predictor_type,
    predictor_pretrain,
    mask_type,
):  
    device = 'cpu'
    d_in        = 1
    d_model     = 128
    seq_len     = 100
    d_out, average = SeqComb.get_num_classes(dataset)

    test_set = SeqComb.get_SeqComv(dataset, 'TEST')

    exp_name = get_exp_name(loss, backbone, weights, seg_dist, dataset, target_type, predictor_type, predictor_pretrain, mask_type)
    exp_dir = f'./checkpoints/{exp_name}'
    
    checkpoints = torch.load(f'{exp_dir}/checkpoints.pth')

    policy_module = Policy.init_policy_module(
        d_in = d_in,
        d_model = d_model,
        seq_len = seq_len,
        backbone = backbone,
        seg_dist = seg_dist
    )
    policy_module = policy_module.to(device)
    policy_module.load_state_dict(checkpoints['policy_state'])
    policy_module.eval()
    class_samples = {}
    for c in range(d_out):
        class_samples[c] = []

    for batch in test_set:
        x = batch['x']
        y = batch['y']
        if len(class_samples[y.item()]) < 20:
            class_samples[y.item()].append([x.squeeze(0), y])

        tot = 0
        for k, v in class_samples.items():
            tot += len(v)
        if tot == 20 * d_out:
            break
    

    blackbox_model = Predictor.PredictorNetwork(d_in=1, d_model=64, d_out=d_out, seq_len=seq_len, backbone='tcn')
    blackbox_model.load_state_dict(torch.load(f'./blackbox/best_{dataset}_tcn.pth')['model_state'])
    blackbox_model = blackbox_model.to(device)
    blackbox_model = blackbox_model.eval()

    if predictor_type == 'blackbox':
        predictor = blackbox_model
    elif predictor_type == 'predictor':
        predictor = Predictor.PredictorNetwork(d_in=d_in, d_model=d_model, d_out=d_out, seq_len=seq_len, backbone=backbone)
        predictor = predictor.to(device)
        predictor.load_state_dict(checkpoints['predictor_state'])
        predictor.eval()
    
    mask_fn = masking.MaskingFunction(mask_type)

    ce_reward_fn = partial(Reward.exp_minus_cross_entropy_reward, mask_fn=mask_fn, predictor=predictor)
    length_reward_fn = Reward.length_reward

    reward_fns = [ce_reward_fn, length_reward_fn]
    reward_fn = partial(Reward.compose_reward, reward_fns=reward_fns, weights=weights)


    n_cols = 5
    rows_per_class = 20 // n_cols  # 4
    n_rows = rows_per_class * d_out    # 클래스 2개

    fig = plt.figure(figsize=(20, 4 * n_rows))
    outer_gs = fig.add_gridspec(n_rows, n_cols) #, wspace=0.3, hspace=0.4)

    # fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows), sharey=True)
    for cls in range(d_out):
        for idx, (x_sample, y_sample) in enumerate(class_samples[cls]):
            row = cls * rows_per_class + idx // n_cols
            col = idx % n_cols

            inner_gs = outer_gs[row, col].subgridspec(
                4, 1, height_ratios=[2, 0.5, 0.5, 0.5], hspace=0.03
            )
            ax_top  = fig.add_subplot(inner_gs[0])
            ax_mask  = fig.add_subplot(inner_gs[1], sharex=ax_top)
            ax_start  = fig.add_subplot(inner_gs[2], sharex=ax_top)
            ax_end  = fig.add_subplot(inner_gs[3], sharex=ax_top)
            # ------------------------------------------------------
            x_tensor = x_sample.unsqueeze(0).to(device)  # [1, T, C]
            if target_type == 'blackbox':
                y_target = blackbox_model(x_tensor)
            elif target_type == 'y':
                y_target = y_sample.unsqueeze(0).to(device)

            with torch.no_grad():
                td = TensorDict(
                    {"x": x_tensor, 'y':y_target ,"curr_mask": torch.zeros_like(x_tensor, dtype=torch.bool)},
                    batch_size=(1,), device=device
                )
                dist = policy_module.get_dist(td)
                mask_probs = dist.calculate_marginal_mask().cpu().numpy()
                env.step(td, policy_module, reward_fn, mode=True)

            x_masked = mask_fn(x_tensor, td["next", "curr_mask"])
        
            with torch.no_grad():
                if target_type == 'blackbox':
                    probs = blackbox_model(x_masked).softmax(-1)
                    prob_true = probs[0, cls].item()
                elif target_type == 'y':
                    prob_true = 1.

            x_np = x_sample.cpu().squeeze().numpy()
            x_masked = x_masked.squeeze().cpu().numpy()
            mask = td["next", "curr_mask"][0].cpu().squeeze().numpy()    # [T]
            

            ax_top.plot(x_np,     label="Original")
            ax_top.plot(x_masked, label="Masked")
            ax_top.set_title(f"Class {cls} Sample {idx+1}", fontsize=9)
            ax_top.tick_params(labelsize=7)
            if col == 0:
                ax_top.set_ylabel("Value", fontsize=8)

            # 마스크된 구간 강조 (axvspan)
            mi = mask.astype(int)
            d  = np.diff(mi)
            starts = np.where(d == 1)[0] + 1
            ends   = np.where(d == -1)[0] + 1
            if mask[0]:
                starts = np.r_[0, starts]
            if mask[-1]:
                ends = np.r_[ends, mask.size]
            for s, e in zip(starts, ends):
                ax_top.axvspan(s, e, alpha=0.25)


            ax_mask.imshow(mask_probs.reshape(-1)[np.newaxis, :],
                aspect='auto', cmap='RdBu_r',
                interpolation='nearest',
                extent=[0, len(mask)-1, 0, 1]
            )
            ax_mask.set_yticks([])
            ax_mask.set_xlim(0, len(mask)-1)
            if col == 0:
                ax_mask.set_ylabel("mask", fontsize=8)
            ax_mask.set_xlabel("t", fontsize=8)

            if seg_dist == 'cat_cat':
                start_p = dist.cat_start.probs.detach().cpu().numpy().reshape(-1)
                ax_start.bar(range(seq_len), start_p, color='tab:blue', label='P(start=s)')
                ax_start.legend(fontsize=5)

                end_p = dist.end_logits.softmax(dim=-1).detach().cpu().numpy().reshape(-1)
                ax_end.bar(range(seq_len), end_p, color='tab:orange', label='P(end=e)')
                ax_end.legend(fontsize=5)

            elif seg_dist == 'cat_nb':
                start_p = dist.cat.probs.detach().cpu().numpy().reshape(-1)
                ax_start.bar(range(seq_len), start_p, color='tab:blue', label='P(start=s)')
                ax_start.legend(fontsize=5)

                end_p = dist.nb.log_prob(torch.arange(seq_len, device=dist.nb.probs.device)).exp().detach().cpu().numpy().reshape(-1)
                ax_end.bar(range(seq_len), end_p, color='tab:orange', label=f'End r={dist.nb.total_count.item():.2f}, p={dist.nb.probs.item():.2f}')
                ax_end.legend(fontsize=5)
            
            elif seg_dist == 'nb_nb':
                start_p = dist.nb_start.log_prob(torch.arange(seq_len, device=dist.nb_start.probs.device)).exp().detach().cpu().numpy().reshape(-1)
                ax_start.bar(range(seq_len), start_p, color='tab:blue', label=f'Start r={dist.nb_start.total_count.item():.2f}, p={dist.nb_start.probs.item():.2f}')
                ax_start.legend(fontsize=5)

                end_p = dist.nb_end.log_prob(torch.arange(seq_len, device=dist.nb_end.probs.device)).exp().detach().cpu().numpy().reshape(-1)
                ax_end.bar(range(seq_len), end_p, color='tab:orange', label=f'End r={dist.nb_end.total_count.item():.2f}, p={dist.nb_end.probs.item():.2f}')
                ax_end.legend(fontsize=5)

    # 범례 한 번만 표시
    handles, labels = ax_top.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.suptitle("Test Samples: Original vs Masked (with 1-D Heat-map)", fontsize=16)
    fig.savefig(f"{exp_dir}/sample.png", dpi=200)
    plt.close() 


# def test_sample_plots(
#     loss,
#     backbone,
#     weights,
#     seg_dist,
#     dataset,
#     target_type,
#     predictor_type,
#     predictor_pretrain,
#     mask_type,
# ):  
#     device = 'cuda:4'
#     d_in        = 1
#     d_model     = 128
#     seq_len     = 100
#     d_out, average = SeqComb.get_num_classes(dataset)

#     test_set = SeqComb.get_SeqComv(dataset, 'TEST')

#     exp_name = get_exp_name(loss, backbone, weights, seg_dist, dataset, target_type, predictor_type, predictor_pretrain, mask_type)
#     exp_dir = f'./checkpoints/{exp_name}'
    
#     checkpoints = torch.load(f'{exp_dir}/checkpoints.pth')

#     policy_module = Policy.init_policy_module(
#         d_in = d_in,
#         d_model = d_model,
#         seq_len = seq_len,
#         backbone = backbone,
#         seg_dist = seg_dist
#     )
#     policy_module = policy_module.to(device)
#     policy_module.load_state_dict(checkpoints['policy_state'])
#     policy_module.eval()
#     class_samples = {}
#     for c in range(d_out):
#         class_samples[c] = []

#     for batch in test_set:
#         x = batch['x']
#         y = batch['y']
#         if len(class_samples[y.item()]) < 20:
#             class_samples[y.item()].append([x.squeeze(0), y])

#         tot = 0
#         for k, v in class_samples.items():
#             tot += len(v)
#         if tot == 20 * d_out:
#             break
    

#     blackbox_model = Predictor.PredictorNetwork(d_in=1, d_model=64, d_out=d_out, seq_len=seq_len, backbone='tcn')
#     blackbox_model.load_state_dict(torch.load(f'./blackbox/best_{dataset}_tcn.pth')['model_state'])
#     blackbox_model = blackbox_model.to(device)
#     blackbox_model = blackbox_model.eval()

#     if predictor_type == 'blackbox':
#         predictor = blackbox_model
#     elif predictor_type == 'predictor':
#         predictor = Predictor.PredictorNetwork(d_in=d_in, d_model=d_model, d_out=d_out, seq_len=seq_len, backbone=backbone)
#         predictor = predictor.to(device)
#         predictor.load_state_dict(checkpoints['predictor_state'])
#         predictor.eval()
    
#     mask_fn = masking.MaskingFunction(mask_type)

#     ce_reward_fn = partial(Reward.exp_minus_cross_entropy_reward, mask_fn=mask_fn, predictor=predictor)
#     length_reward_fn = Reward.length_reward

#     reward_fns = [ce_reward_fn, length_reward_fn]
#     reward_fn = partial(Reward.compose_reward, reward_fns=reward_fns, weights=weights)


#     n_cols = 5
#     rows_per_class = 20 // n_cols  # 4
#     n_rows = rows_per_class * d_out    # 클래스 2개

#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows), sharey=True)
#     for cls in range(d_out):
#         for idx, (x_sample, y_sample) in enumerate(class_samples[cls]):
#             row = cls * rows_per_class + idx // n_cols
#             col = idx % n_cols

#             x_tensor = x_sample.unsqueeze(0).to(device)  # [1, T, C]
#             if target_type == 'blackbox':
#                 y_target = blackbox_model(x_tensor)
#             elif target_type == 'y':
#                 y_target = y_sample.unsqueeze(0).to(device)

#             with torch.no_grad():
#                 td = TensorDict(
#                     {"x": x_tensor, 'y':y_target ,"curr_mask": torch.zeros_like(x_tensor, dtype=torch.bool)},
#                     batch_size=(1,), device=device
#                 )
#                 env.step(td, policy_module, reward_fn, mode=True)


#             x_masked = mask_fn(x_tensor, td["next", "curr_mask"])
        
#             with torch.no_grad():
#                 if target_type == 'blackbox':
#                     probs = blackbox_model(x_masked).softmax(-1)
#                     prob_true = probs[0, cls].item()
#                 elif target_type == 'y':
#                     prob_true = 1.

#             x_np = x_sample.cpu().squeeze().numpy()
#             x_masked = x_masked.squeeze().cpu().numpy()
#             mask = td["next", "curr_mask"][0].cpu().squeeze().numpy()    # [T]
            

#             ax = axes[row][col]
#             ax.plot(x_np,    label="Original")
#             ax.plot(x_masked, label="Masked")
#             ax.text(0.95, 0.95, f"P(class={cls})={prob_true:.2f}",
#                     ha='right', va='top', transform=ax.transAxes,
#                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7))

#             # 마스크된 구간 강조 (axvspan)
#             mi = mask.astype(int)
#             d = np.diff(mi)
#             starts = np.where(d == 1)[0] + 1
#             ends   = np.where(d == -1)[0] + 1
#             if mask[0]:
#                 starts = np.r_[0, starts]
#             if mask[-1]:
#                 ends = np.r_[ends, mask.size]
#             for s, e in zip(starts, ends):
#                 ax.axvspan(s, e, alpha=0.3)

#             ax.set_title(f"Class {cls} Sample {idx+1}")
#             if col == 0:
#                 ax.set_ylabel("Value")

#     # 범례 한 번만 표시
#     axes[0][0].legend(loc="lower right")
#     fig.suptitle("Test Samples: Original vs Masked", fontsize=16)
#     fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.savefig(f"{exp_dir}/sample.png")
#     plt.close()

    

def result_plots(
    loss,
    backbone,
    weights,
    seg_dist,
    dataset,
    target_type,
    predictor_type,
    predictor_pretrain,
    mask_type,
):
    exp_name = get_exp_name(loss, backbone, weights, seg_dist, dataset, target_type, predictor_type, predictor_pretrain, mask_type)
    exp_dir = f'./checkpoints/{exp_name}'
    
    # checkpoint = torch.load(f"{exp_dir}/checkpoints.pth")
    history = torch.load(f"{exp_dir}/history.pth")

    avg_lengths = history['valid_length']
    avg_lengths_tr = history['train_length']
    avg_rewards = history['valid_reward']
    avg_rewards_tr = history['train_reward']
    masked_accs = history['masked_acc']
    masked_f1s = history['masked_f1']
    epochs = list(range(1, len(avg_rewards)+1))

    best_idx = avg_rewards.index(max(avg_rewards))
    best_epoch = best_idx+1

    fig, axs = plt.subplots(4, 1, figsize=(8, 14), sharex=True)

    # Avg Length
    axs[0].plot(epochs, avg_lengths_tr, label='Train')
    axs[0].plot(epochs, avg_lengths, label='Valid')
    axs[0].legend()
    axs[0].set_ylabel('Avg Length')
    axs[0].set_title('Train & Valid Avg Length')
    axs[0].plot(best_epoch, avg_lengths[best_idx], 'ro')
    axs[0].annotate(f'Length {avg_lengths[best_idx]:.2f}',
                    xy=(best_epoch, avg_lengths[best_idx]),
                    xytext=(best_epoch, avg_lengths[best_idx]*1.05),
                    arrowprops=dict(arrowstyle='->'))

    # Avg Reward
    axs[1].plot(epochs, avg_rewards_tr, label='Train')
    axs[1].plot(epochs, avg_rewards, label='Valid')
    axs[1].legend()
    axs[1].set_ylabel('Avg Reward')
    axs[1].set_title('Train & Valid Avg Reward')
    axs[1].plot(best_epoch, avg_rewards[best_idx], 'ro')
    axs[1].annotate(f'Reward {avg_rewards[best_idx]:.2f}',
                    xy=(best_epoch, avg_rewards[best_idx]),
                    xytext=(best_epoch, avg_rewards[best_idx]*1.05),
                    arrowprops=dict(arrowstyle='->'))

    # Masked Accuracy
    axs[2].plot(epochs, masked_accs)
    axs[2].set_ylabel('Masked Acc')
    axs[2].set_title('Validation Masked Accuracy')
    axs[2].plot(best_epoch, masked_accs[best_idx], 'ro')
    axs[2].annotate(f'Acc {masked_accs[best_idx]:.2f}',
                    xy=(best_epoch, masked_accs[best_idx]),
                    xytext=(best_epoch, masked_accs[best_idx]+0.02),
                    arrowprops=dict(arrowstyle='->'))

    # Masked F1
    axs[3].plot(epochs, masked_f1s)
    axs[3].set_xlabel('Epoch')
    axs[3].set_ylabel('Masked F1')
    axs[3].set_title('Validation Masked F1 Score')
    axs[3].plot(best_epoch, masked_f1s[best_idx], 'ro')
    axs[3].annotate(f'F1 {masked_f1s[best_idx]:.2f}',
                    xy=(best_epoch, masked_f1s[best_idx]),
                    xytext=(best_epoch, masked_f1s[best_idx]+0.02),
                    arrowprops=dict(arrowstyle='->'))

    fig.tight_layout()
    plt.savefig(f'{exp_dir}/result_plots.png')
    plt.close()