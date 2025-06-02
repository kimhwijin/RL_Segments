import sys
sys.path.append("/home/hjkim/RL_TimeSegment")

import re
import matplotlib.pyplot as plt


import re
import matplotlib.pyplot as plt
from functools import partial
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

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
):  
    d_in = 1
    d_model = 128
    test_set = SeqComb.get_SeqComv(dataset, 'TEST')
    d_out, average = SeqComb.get_num_classes(dataset)
    seq_len = 100
    device = 'cuda:4'

    exp_name = get_exp_name(loss, backbone, weights, seg_dist, dataset)
    exp_dir = f'./checkpoints/{exp_name}'

    if seg_dist == 'cat_nb':
        d_start, d_end = 100, 2
        SegmentDistribution = CategoricalToNegativeBinomial
    elif seg_dist == 'cat_cat':
        d_start, d_end = 100, 100
        SegmentDistribution = CategoricalToCategorical
    elif seg_dist == 'nb_nb':
        d_start, d_end = 2, 2
        SegmentDistribution = NegativeBinomialToNegativeBinomial

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
    policy_module = policy_module.to(device)

    class_samples = {}
    for c in range(d_out):
        class_samples[c] = []

    for batch in test_set:
        x = batch['x']      # shape [1, T, C]
        y = batch['y']
        if len(class_samples[y.item()]) < 20:
            class_samples[y.item()].append([x.squeeze(0), y])  # [T, C] 혹은 [T]

        tot = 0
        for k, v in class_samples.items():
            tot += len(v)
        if tot == 20 * d_out:
            break

    blackbox_model = Predictor.PredictorNetwork(d_in=1, d_model=64, d_out=d_out, seq_len=seq_len, backbone='tcn')
    blackbox_model.load_state_dict(torch.load(f'./blackbox/best_{dataset}_tcn.pth')['model_state'])
    blackbox_model = blackbox_model.to(device)
    blackbox_model = blackbox_model.eval()

    results = torch.load(f'{exp_dir}/{exp_name}.pth')
    best_epoch = results['epoch']
    best_acc = results['acc']
    best_f1 = results['f1']
    best_length = results['length']
    best_reward = results['reward']
    policy_module.load_state_dict(results['policy_state'])

    mask_fn = masking.SeqCombMask()
    ce_reward_fn = partial(Reward.exp_minus_cross_entropy_reward, mask_fn=mask_fn, predictor=blackbox_model)
    length_reward_fn = Reward.length_reward

    reward_fns = [ce_reward_fn, length_reward_fn]
    reward_fn = partial(Reward.compose_reward, reward_fns=reward_fns, weights=weights)



    n_cols = 5
    rows_per_class = 20 // n_cols  # 4
    n_rows = rows_per_class * d_out    # 클래스 2개

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows), sharey=True)
    for cls in range(d_out):
        for idx, (x_sample, y_sample) in enumerate(class_samples[cls]):
            row = cls * rows_per_class + idx // n_cols
            col = idx % n_cols

            x_tensor = x_sample.unsqueeze(0).to(device)  # [1, T, C]
            y_tensor = y_sample.unsqueeze(0).to(device)  # [1, T, C]
            B = 1
            with torch.no_grad():
                td = TensorDict(
                    {"x": x_tensor, 'y':y_tensor ,"curr_mask": torch.zeros_like(x_tensor, dtype=torch.bool)},
                    batch_size=(B,), device=device
                )
                # 결정론적 모드로 한 번만 env.step() → mask 생성
                env.step(td, policy_module, reward_fn, mode=True)

            # 마스크와 원본/마스킹 시퀀스 추출
            mask = td["next", "curr_mask"][0].cpu().squeeze().numpy()    # [T]
            x_np = x_sample.cpu().squeeze().numpy()
            x_masked = masking.SeqCombMask()(x_tensor, td["next", "curr_mask"])
            x_masked = x_masked.squeeze().cpu().numpy()

            x_masked_td = masking.SeqCombMask()(x_tensor, td["next", "curr_mask"])
            
            with torch.no_grad():
                probs = blackbox_model(x_masked_td).softmax(-1)
            prob_true = probs[0, cls].item()

            ax = axes[row][col]
            ax.plot(x_np,    label="Original")
            ax.plot(x_masked, label="Masked")
            ax.text(0.95, 0.95, f"P(class={cls})={prob_true:.2f}",
                    ha='right', va='top', transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7))

            # 마스크된 구간 강조 (axvspan)
            mi = mask.astype(int)
            d = np.diff(mi)
            starts = np.where(d == 1)[0] + 1
            ends   = np.where(d == -1)[0] + 1
            if mask[0]:
                starts = np.r_[0, starts]
            if mask[-1]:
                ends = np.r_[ends, mask.size]
            for s, e in zip(starts, ends):
                ax.axvspan(s, e, alpha=0.3)

            ax.set_title(f"Class {cls} Sample {idx+1}")
            if col == 0:
                ax.set_ylabel("Value")

    # 범례 한 번만 표시
    axes[0][0].legend(loc="lower right")
    fig.suptitle("Test Samples: Original vs Masked", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{exp_dir}/sample.png")
    plt.close()

    

def result_plots(
    loss,
    backbone,
    weights,
    seg_dist,
    dataset,
):
    exp_name = get_exp_name(loss, backbone, weights, seg_dist, dataset)
    exp_dir = f'./checkpoints/{exp_name}'
    
    with open(f"{exp_dir}/{exp_name}.log", 'r') as f:
        line = f.readlines()
        line = "".join(line)

    epoch_iter = re.finditer(
        r'Epoch\s+(\d+)(.*?)(?=Epoch\s+\d+|\Z)',
        line,
        re.DOTALL
    )
    epochs = []
    avg_lengths = []
    avg_rewards = []
    masked_accs = []
    masked_f1s = []

    for m in epoch_iter:
        e = int(m.group(1))
        blk = m.group(2)

        L = re.findall(r'Avg Length:\s*([\d\.]+)', blk)
        R = re.findall(r'Avg Reward:\s*([\d\.]+)', blk)
        if len(L) < 2 or len(R) < 2:
            continue
        epochs.append(e)
        avg_lengths.append(float(L[1]))
        avg_rewards.append(float(R[1]))

        ac = re.search(r'Masked Acc:\s*([\d\.]+)', blk)
        f1 = re.search(r'Masked F1:\s*([\d\.]+)', blk)
        if ac and f1:
            masked_accs.append(float(ac.group(1)))
            masked_f1s.append(float(f1.group(1)))



    best_idx = avg_rewards.index(max(avg_rewards))
    best_epoch = best_idx+1

    fig, axs = plt.subplots(4, 1, figsize=(8, 14), sharex=True)

    # Avg Length
    axs[0].plot(epochs, avg_lengths)
    axs[0].set_ylabel('Avg Length')
    axs[0].set_title('Validation Avg Length')
    axs[0].plot(best_epoch, avg_lengths[best_idx], 'ro')
    axs[0].annotate(f'Length {avg_lengths[best_idx]:.2f}',
                    xy=(best_epoch, avg_lengths[best_idx]),
                    xytext=(best_epoch, avg_lengths[best_idx]*1.05),
                    arrowprops=dict(arrowstyle='->'))

    # Avg Reward
    axs[1].plot(epochs, avg_rewards)
    axs[1].set_ylabel('Avg Reward')
    axs[1].set_title('Validation Avg Reward')
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