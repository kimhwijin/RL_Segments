import torch
from pathlib import Path

f_dir = Path(__file__).parent.parent


def get_num_classes(dataset):
    if dataset.lower() == 'one':
        return 2, "binary"
    elif dataset.lower() == 'onetwo':
        return 3, "macro"

def get_SeqComv(dataset_name : str, phase : str):
    if dataset_name.lower() == 'one':
        return SeqCombOneUV(phase)
    elif dataset_name.lower() == 'onetwo':
        return SeqCombOneTwoUV(phase)


class SeqCombOneUV(torch.utils.data.Dataset):
    def __init__(self, split):
        path = f_dir / 'dataset' / 'SeqCombOneUV'
        tr = torch.load(path / f'{split.lower()}.pt', weights_only=False)
        self.X = torch.FloatTensor(tr['X'])
        self.times = torch.FloatTensor(tr['times'])
        self.gt_mask = torch.FloatTensor(tr['gt_mask'])
        self.y = torch.LongTensor(tr['y'])

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        x = self.X[idx]
        t = self.times[idx]
        y = self.y[idx]
        gt_mask = self.gt_mask[idx]
        return {
            "x": x, 
            "y": y, 
            "gt_mask": gt_mask,
        }

class SeqCombOneTwoUV(torch.utils.data.Dataset):
    def __init__(self, split):
        path = f_dir / 'dataset' / 'SeqCombOneTwoUV'
        tr = torch.load(path / f'{split.lower()}.pt', weights_only=False)
        self.X = torch.FloatTensor(tr['X'])
        self.times = torch.FloatTensor(tr['times'])
        self.gt_mask = torch.FloatTensor(tr['gt_mask'])
        self.y = torch.LongTensor(tr['y'])

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        x = self.X[idx]
        t = self.times[idx]
        y = self.y[idx]
        gt_mask = self.gt_mask[idx]
        return {
            "x": x, 
            "y": y, 
            "gt_mask": gt_mask,
        }
