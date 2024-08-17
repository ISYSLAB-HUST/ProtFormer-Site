"""
"""
from torch.utils.data import DataLoader, Dataset
import pandas as pd


class ProteinDataset(Dataset):
    def __init__(self, data_file=None, max_len=1000):
        super(ProteinDataset, self).__init__()
        temp_data = pd.read_csv(data_file)
        self.names = []
        self.sequences = []
        self.labels = []
        for _, row in temp_data.iterrows():
            sequence = row['sequence'][:max_len]
            label = row['label'][:max_len]
            self.names.append(row['name'])
            self.sequences.append(sequence)
            self.labels.append(label)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.names[index], self.sequences[index], self.labels[index]


if __name__ == "__main__":
    data_file = "./dataset/Task_DeepPPISP/Test_70.csv"

    dataset = ProteinDataset(data_file)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for batch in test_loader:
        print(batch)
        break
