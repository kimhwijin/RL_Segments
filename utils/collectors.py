import torch



def collect_data(dataset, policy_module, sample_per_epoch):
    x = dataset.X
    
    policy_module()

    