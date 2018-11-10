from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch.utils.data as D


class ExpDataSet(Dataset):
    def __init__(self, train=True, normalization=True):
        data_file = "./2/TrainSamples.csv"
        label_file = "./2/TrainLabels.csv"
        skiprows, nrows = (0, 14000) if train else (14000, None)
        print(skiprows, nrows)

        origin_data = pd.read_csv(data_file, dtype=np.float, skiprows=skiprows, nrows=nrows, header=None).values
        labels = pd.read_csv(label_file, dtype=np.float, skiprows=skiprows, nrows=nrows, header=None).values

        if normalization:
            self.data = (origin_data - np.mean(origin_data)) / np.std(origin_data, ddof=1)
        self.labels = labels
        assert len(self.data) == len(self.labels)

        print(len(origin_data), len(labels))

    def __getitem__(self, index):
        return (self.data[index], self.labels[index][0])

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    train_loader = D.DataLoader(ExpDataSet(train=False),
                                batch_size=2, shuffle=False, num_workers=32
                                )

    for i, data in enumerate(train_loader):
        print(data)
