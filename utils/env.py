import torch
from tensordict import TensorDict

def step(
    tensordict,
    policy_module,
    reward_fn,
    # termimated_fn,
    mode=False,
):  
    # Action
    if not mode:
        policy_module(tensordict)
    else:
        dist = policy_module.get_dist(tensordict)
        action = dist.deterministic_sample
        tensordict['action'] = action

    x = tensordict["x"]
    y = tensordict["y"]
    action = tensordict["action"]
    curr_mask = tensordict["curr_mask"]

    arange = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
    start, end = action[:, :1], action[:, 1:]
    new_mask = torch.logical_and(start <= arange, arange <= end).unsqueeze(-1)
    next_mask = torch.logical_or(curr_mask, new_mask)
    
    if reward_fn is None:
        reward = torch.zeros(x.shape[0], 1, dtype=x.dtype)
    else:
        reward = reward_fn(x=x, y=y, curr_mask=curr_mask, new_mask=new_mask) # no grad reward func
    # is_done = terminated_fn(x, y, curr_mask, new_mask)

    next_tensordict = TensorDict(
        {
            'x': x,
            'y': y,
            'curr_mask': next_mask,
            'reward': reward,
            'done': torch.ones(x.shape[0], 1, dtype=bool)
        },
        batch_size=tensordict.shape,
        device=tensordict.device
    )
    tensordict['next'] = next_tensordict
    return tensordict
    

    