"""
"""
from torch.utils.data import DataLoader, Dataset
import pandas as pd


class ThreeDProteinDataset(Dataset):
    def __init__(self, data_file=None, max_len=1000):
        super(ThreeDProteinDataset, self).__init__()
        temp_data = pd.read_csv(data_file)
        self.names = []
        self.sequences = []
        self.labels = []
        self.combined = []
        for _, row in temp_data.iterrows():
            sequence = row['seq'][:max_len]
            label = row['label'][:max_len]
            combined = row['combined_seq'][:max_len*2]
            self.names.append(row['name'])
            self.sequences.append(sequence)
            self.labels.append(label)
            self.combined.append(combined)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.names[index], self.sequences[index], self.labels[index], self.combined[index]


if __name__ == "__main__":
    data_file = "./Task_ThreeD_dataset/Task_DeepPPISP/Test_70.csv"

    dataset = ThreeDProteinDataset(data_file)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for batch in test_loader:
        print(batch)
        break
