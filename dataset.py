import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class RRTStarDataset(Dataset):
    def __init__(self, dataset_path):
        self.data = np.load(dataset_path, allow_pickle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tree_nodes, next_sample_point, env_map = self.data[idx]

        tree_nodes_tensor = torch.tensor(tree_nodes, dtype=torch.float32)  # shape: (N, 2)
        next_sample_point_tensor = torch.tensor(next_sample_point, dtype=torch.float32)  # shape: (2,)
        env_map_tensor = torch.tensor(env_map, dtype=torch.float32)  # shape: (height, width)

        return tree_nodes_tensor, next_sample_point_tensor, env_map_tensor

def collate_fn(batch):
    """collate function, process the different length data."""
    tree_nodes_batch = [item[0] for item in batch]
    next_sample_points_batch = [item[1] for item in batch]
    env_maps_batch = [item[2] for item in batch]

    # Get the same length
    padded_tree_nodes = pad_sequence(tree_nodes_batch, batch_first=True, padding_value=0.0)  # (batch_size, max_nodes, 2)

    next_sample_points = torch.stack(next_sample_points_batch)  # (batch_size, 2)
    env_maps = torch.stack(env_maps_batch)  # (batch_size, height, width)

    return padded_tree_nodes, next_sample_points, env_maps

def get_data_loader(dataset_path, batch_size=32, shuffle=True, validation_split=0.1):
    dataset = RRTStarDataset(dataset_path)
    print("DataSet Size: ", dataset.data.shape)
    
    # if validation_split > 0 and validation_split < 1:
    if validation_split > 0 and validation_split < 1:
        dataset_size = len(dataset)
        validation_size = int(dataset_size * validation_split)
        train_size = dataset_size - validation_size
        
        # randomly split the dataset
        train_dataset, val_dataset = random_split(
            dataset, [train_size, validation_size], 
            generator=torch.Generator().manual_seed(42)  # set random seed for reproducibility
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")
        
        return train_loader, val_loader
    else:
        # return the whole dataset as a single loader
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        return data_loader
