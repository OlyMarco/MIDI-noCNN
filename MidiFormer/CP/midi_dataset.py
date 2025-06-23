from torch.utils.data import Dataset
import torch

class MidiDataset(Dataset):
    def __init__(self, X):
        """
        X: 包含(segments, pctm, nltm)的元组
        """
        self.segments, self.pctm, self.nltm = X
        
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        return {
            'segments': torch.LongTensor(self.segments[idx]),
            'pctm': torch.FloatTensor(self.pctm[idx]),
            'nltm': torch.FloatTensor(self.nltm[idx])
        }
