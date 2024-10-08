from pathlib import Path
import re
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import dense_to_sparse

from Utils import compute_KNN_graph


class ConnectivityData(InMemoryDataset):
    """ Dataset for the connectivity data."""

    def __init__(self,
                 root):
        super(ConnectivityData, self).__init__(root, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    #只读
    @property
    def raw_file_names(self):
        file_paths = list(Path(self.raw_dir).glob("*.csv"))

        def extract_number(filename):
            match = re.search(r'(\d+)', filename)
            return int(match.group()) if match else 0
        file_paths = sorted(file_paths, key=lambda x: extract_number(x.stem))

        return [str(file_path.name) for file_path in file_paths]


    #只读
    @property
    def processed_file_names(self):
        return 'data.pt'


    def set_new_indices(self):
        self.__indices__ = list(range(self.len()))

    def process(self):
        labels = np.genfromtxt(Path(self.raw_dir) / "SLabels.txt", encoding='utf-8', skip_header=1)


        data_list = []

        for filename, y in zip(self.raw_paths, labels):

            y = torch.tensor([y]).long()
            connectivity = np.genfromtxt(filename, delimiter=',')
            np.fill_diagonal(connectivity, 0)
            x = torch.from_numpy(connectivity).float()

            adj = compute_KNN_graph(connectivity)
            adj = torch.from_numpy(adj).float()
            edge_index, edge_attr = dense_to_sparse(adj)

            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
