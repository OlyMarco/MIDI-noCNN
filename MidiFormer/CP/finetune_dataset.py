from torch.utils.data import Dataset
import torch

class FinetuneDataset(Dataset):
    """
    Expected data shape: (data_num, data_len)
    """
    def __init__(self, X, y, seq_class=False):
        self.segments, self.pctm, self.nltm = X
        self.label = y
        self.seq_class = seq_class  # 是否为序列分类任务

    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        if self.seq_class:
            # 序列分类任务 - 标签是单个值
            return {
                'segments': torch.LongTensor(self.segments[idx]),
                'pctm': torch.FloatTensor(self.pctm[idx]),
                'nltm': torch.FloatTensor(self.nltm[idx]),
                'labels': torch.LongTensor([self.label[idx]])  # 确保是标量，用列表包装
            }
        else:
            # 标记分类任务 - 标签是序列
            return {
                'segments': torch.LongTensor(self.segments[idx]),
                'pctm': torch.FloatTensor(self.pctm[idx]),
                'nltm': torch.FloatTensor(self.nltm[idx]),
                'labels': torch.LongTensor(self.label[idx])
            }


# from torch.utils.data import Dataset
# import torch

# class FinetuneDataset(Dataset):
#     """
#     Expected data shape: (data_num, data_len)
#     """
#     def __init__(self, X, y, seq_class=False):
#         self.data = X 
#         self.label = y

#     def __len__(self):
#         return(len(self.data))
    
#     def __getitem__(self, index):
#         return torch.tensor(self.data[index]), torch.tensor(self.label[index])