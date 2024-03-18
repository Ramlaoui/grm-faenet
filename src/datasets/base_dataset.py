from pathlib import Path
import lmdb
import subprocess
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
import pickle
from ..utils import pyg2_data_transform


class LMDBDataset(Dataset):
    def __init__(self, config, transform=None, fa_frames=None):
        super(LMDBDataset, self).__init__()
        self.config = config
        self.path = Path(self.config["src"])

        if not self.path.exists():
            try:
                print(f"Downloading {self.path}...")
                subprocess.run(
                    ["bash", str(Path("scripts/") / "download_data.sh"), "is2re"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                print(f"Downloaded {self.path}.")
            except Exception as e:
                print(e)
                raise FileNotFoundError(f"{self.path} does not exist.")

        self.env = lmdb.open(
            str(self.path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )

        self.num_samples = int(self.env.stat()["entries"])
        self._keys = [f"{i}".encode("ascii") for i in range(self.num_samples)]

        self.transform = transform
        self.fa_frames = fa_frames
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        datapoint_pickle = self.env.begin().get(self._keys[idx])
        data_object = pyg2_data_transform(pickle.loads(datapoint_pickle))

        if self.transform:
            data_object = self.transform(data_object)

        return data_object
    
    def close_db(self):
        self.env.close()

class ParallelCollater:
    def __init__(self, num_gpus, otf_graph=False):
        self.num_gpus = num_gpus
        self.otf_graph = otf_graph

    def __call__(self, data_list):
        if self.num_gpus in [0, 1]:  # adds cpu-only case
            batch = data_list_collater(data_list, otf_graph=self.otf_graph)
            return [batch]

        else:
            num_devices = min(self.num_gpus, len(data_list))

            count = torch.tensor([data.num_nodes for data in data_list])
            cumsum = count.cumsum(0)
            cumsum = torch.cat([cumsum.new_zeros(1), cumsum], dim=0)
            device_id = num_devices * cumsum.to(torch.float) / cumsum[-1].item()
            device_id = (device_id[:-1] + device_id[1:]) / 2.0
            device_id = device_id.to(torch.long)
            split = device_id.bincount().cumsum(0)
            split = torch.cat([split.new_zeros(1), split], dim=0)
            split = torch.unique(split, sorted=True)
            split = split.tolist()

            return [
                data_list_collater(data_list[split[i] : split[i + 1]])
                for i in range(len(split) - 1)
            ]

def data_list_collater(data_list, otf_graph=False):
    batch = Batch.from_data_list(data_list)

    if (
        not otf_graph
        and hasattr(data_list[0], "edge_index")
        and data_list[0].edge_index is not None
    ):
        try:
            n_neighbors = []
            for i, data in enumerate(data_list):
                n_index = data.edge_index[1, :]
                n_neighbors.append(n_index.shape[0])
            batch.neighbors = torch.tensor(n_neighbors)
        except NotImplementedError:
            pass

    return batch
