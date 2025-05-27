import os
from typing import List, Callable

from sklearn.utils import shuffle

import torch
import pandas as pd
# from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from datasets.lattice import Structure
from utils import lattice_params_to_matrix_torch
from utils.lattice_utils import Topology, scale_to_cell
import pickle

from torch_geometric.datasets import MD17
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_tar,
    extract_zip,
)



class LatticeModulus(InMemoryDataset):
    downloadurl = ''

    file_names = {'LatticeModulus':'data.pkl'}
    name = 'LatticeModulus'
    def __init__(self, data_path: str,
                 file_name='data'):
        # define column names from data
        self.raw_data_keys = ['Name', 'Other name(s)', 'lengths', 'angles', 'Z_avg', 'Young', 'Shear', 'Poisson',
                              'Emax',
                              'Scaling constants', 'Scaling exponents', 'has overlapping bars', 'Nodal positions',
                              'Edge index']
        self.properties_names = ['Young', 'Shear', 'Poisson']

        self.data_path = data_path
        self.file_name = file_name

        super().__init__(root=data_path)

        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def processed_data_exist(self):
        return os.path.exists(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.file_name, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.file_name, 'processed')
    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']

    @property
    def raw_file_names(self) -> str:
        name = self.file_names[self.name]
        return name

    def download(self):
        url = self.downloadurl
        path = download_url(url, self.raw_dir)
        return path


    def process(self):
        print('Processing data...')
        with open(self.raw_paths[0], mode='rb') as f:
            raw_data = pickle.load(f)

        data_list = []
        #['Name', 'Other name(s)', 'lengths', 'angles', 'Z_avg', 'Young', 'Shear', 'Poisson', 'Emax',
         # 'Scaling constants', 'Scaling exponents', 'has overlapping bars', 'Nodal positions', 'Edge index']
        for i in tqdm(range(len(raw_data))):
            exported_lattice = raw_data[i]
            try:
                properties = torch.FloatTensor(list(exported_lattice['Young'])+list(exported_lattice['Shear'])+list(exported_lattice['Poisson']))
                density = torch.FloatTensor([exported_lattice['Density']])
                S1 = Structure(torch.FloatTensor([exported_lattice['lengths'],exported_lattice['angles']]),
                               torch.LongTensor(exported_lattice['Edge index']).squeeze(1),
                               torch.FloatTensor(exported_lattice['Nodal positions']).squeeze(1), is_cartesian=False,
                               properties=properties,
                               properties_names=self.properties_names)
            except KeyError:
                print('Error sample {}, skip it.'.format(i))
                continue
            edge_num = S1.edge_index.shape[1]
            node_feat = torch.zeros((S1.num_nodes, 1), dtype=torch.long)
            edge_feat = torch.zeros((edge_num, 1), dtype=torch.float32)
            edge_num = S1.num_edges
            #if S1.num_nodes != 14: continue
            data = Data(
                frac_coords=S1.frac_coords.to(torch.float32),
                cart_coords=S1.cart_coords.to(torch.float32),
                node_feat=node_feat,
                edge_feat=edge_feat,
                edge_index=S1.edge_index,
                num_nodes=S1.num_nodes,
                num_atoms=S1.num_nodes,
                num_edges=edge_num,
                lengths=S1.lattice_params[0].view(1, -1).to(torch.float32),
                angles=S1.lattice_params[1].view(1, -1).to(torch.float32),
                vector=S1.lattice_vector.view(1,-1).to(torch.float32),
                y=S1.properties.to(torch.float32).view(1, -1),
                to_jimages = S1.to_jimages,
                density = density
            )

            data_list.append(data)
        print('End preprocessing data.')
        print('Saving data...')
        torch.save(self.collate(data_list), self.processed_paths[0])
        print('Completed preprocessing data.')


    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = ids[:train_size], ids[train_size:train_size + valid_size], ids[train_size + valid_size:]
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict







class LatticeStiffness(InMemoryDataset):
    # define column names from data
    F1_features_names = ['relative_density', 'U1', 'U2', 'U3', 'lattice_type1', 'lattice_type2', 'lattice_type3',
                         'lattice_rep1', 'lattice_rep2', 'lattice_rep3']
    R1_names = ['R1_theta', 'R1_rot_ax1', 'R1_rot_ax2']
    V_names = ['V1', 'V2', 'V3']
    R2_names = ['R2_theta', 'R2_rot_ax1', 'R2_rot_ax2']
    C_ort_names = ['C11_ort', 'C12_ort', 'C13_ort', 'C22_ort', 'C23_ort', 'C33_ort', 'C44_ort', 'C55_ort',
                   'C66_ort']
    C_names = ['C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C22', 'C23', 'C24', 'C25', 'C26', 'C33', 'C34', 'C35',
               'C36', 'C44', 'C45', 'C46', 'C55', 'C56', 'C66']

    F1_features_scaling_strategy = 'none'
    V_scaling_strategy = 'none'
    C_ort_scaling_strategy = 'none'
    C_scaling_strategy = 'none'
    C_hat_scaling_strategy = 'none'

    downloadurl='https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/520254/training.csv?sequence=1&isAllowed=y'
    file_names = {'LatticeStiffness':'training.csv'}
    name = 'LatticeStiffness'
    def __init__(self, data_path: str,
                 file_name='training'):
        self.data_path = data_path
        self.file_name = file_name

        super().__init__(root=data_path)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_data_exist(self):
        return os.path.exists(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.file_name, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.file_name, 'processed')
    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']

    @property
    def raw_file_names(self) -> str:
        name = self.file_names[self.name]
        return name

    def download(self):
        url = self.downloadurl
        path = download_url(url, self.raw_dir)
        return path


    def process(self):
        print('Processing data...')

        df = pd.read_csv(self.raw_paths[0])
        print(len(df))
        data_list = []

        # for i in tqdm(range(len(df))):
        for i in tqdm(range(100)):
            dfi = df.iloc[i]
            exported_lattice = Topology(dfi)
            
            #if exported_lattice.coordinates.shape[0] != 14: continue
            coords = torch.from_numpy(exported_lattice.coordinates)
            # lattice_vector = Structure.find_lattice_vectors(coords)
            lattice_vector = torch.from_numpy(exported_lattice.lattice_vector)

            S1 = Structure(lattice_vector,
                           torch.from_numpy(exported_lattice.connectity),
                           coords, is_cartesian=True,
                           diameter=exported_lattice.diameter,
                           properties=torch.from_numpy(dfi[self.C_names].values),
                           properties_names=self.C_names)
            #if S1.num_nodes != 15: continue
            edge_num = S1.num_edges
            node_feat = torch.zeros((S1.num_nodes, 1), dtype=torch.long)
            edge_feat = torch.ones((edge_num, 1), dtype=torch.float32) * S1.diameter
            lattice_vector = S1.lattice_vector.view(1, -1)
            data = Data(
                frac_coords=S1.frac_coords.to(torch.float32),
                cart_coords=S1.cart_coords.to(torch.float32),
                node_feat=node_feat,
                edge_feat=edge_feat,
                edge_index=S1.edge_index,
                num_nodes=S1.num_nodes,
                num_atoms=S1.num_nodes,
                num_edges=edge_num,
                lengths=S1.lattice_params[0].view(1, -1).to(torch.float32),
                angles=S1.lattice_params[1].view(1, -1).to(torch.float32),
                vector=lattice_vector.to(torch.float32),
                y=S1.properties.to(torch.float32).view(1, -1),
                to_jimages = S1.to_jimages
            )
            data_list.append(data)
        print('End preprocessing data.')
        print('Saving data...')
        print('Sample amount: ' + str(len(data_list)))
        torch.save(self.collate(data_list), self.processed_paths[0])
        print('Completed preprocessing data.')


    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.LongTensor(ids[:train_size]), torch.tensor(ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict
    
    def find_corners_index(self,coords):
        corners = []  #index of 4 corners
        center = torch.mean(coords,dim=0)
        for i in range(coords.shape[0]):
            if torch.norm(coords[i,:] - center) < 1e-6:
                continue
            corners.append(i)
        selected_corners = [corners[0]] #4 selected corners to represent the lattice
        for i in range(1,8):
            for j in range(1,8):
                for k in range(1,8):
                    if torch.norm((coords[corners[0],:]-coords[corners[i],:]-coords[corners[j],:]+coords[corners[k],:])) < 1e-6:
                        if torch.norm((coords[corners[0],:]+coords[corners[i],:]+coords[corners[j],:]+coords[corners[k],:])-4*center) > 1e-6:
                            for h in range(1,8):
                                if torch.norm((coords[corners[k],:]+coords[corners[h],:])-2*center) < 1e-6:
                                    return corners, selected_corners + [corners[i],corners[j],corners[h]]
        return corners, [0,1,2,3]

    def shrink_coords(self,ini_coords):
            reshaped_coords = ini_coords.view(-1,3)
            num_node = reshaped_coords.shape[1]
            result = torch.zeros((5,3))
            for _ in range(1):
                corners, selected_corners = self.find_corners_index(reshaped_coords)
                #print(corners)
                #print(selected_corners)
                for j in range(4):
                    result[j,:] = reshaped_coords[selected_corners[j],:]
                position = 4
                for j in range(num_node):
                    if j not in corners:
                        result[position,:] = reshaped_coords[j,:]
                        position += 1
            return result.view(-1,3)



def main():
    from torch_geometric.loader import DataLoader
    from utils.lattice_utils import plot_lattice

    dataset = LatticeStiffness('D:\项目\Material design\code_data\data\LatticeStiffness', file_name='training')
    # dataset = LatticeStiffness('/home/jianpengc/datasets/metamaterial/LatticeStiffness', file_name='training')
    # dataset = LatticeModulus('/home/jianpengc/datasets/metamaterial/LatticeModulus', file_name='data_node_num32')
    # dataset = LatticeModulus('D:\项目\Material design\code_data\data\LatticeModulus', file_name='data')
    # dataset = LatticeStiffness('/home/jianpengc/datasets/metamaterial/LatticeStiffness', file_name='training_node_num47')
    # dataset = LatticeModulus('/home/jianpengc/datasets/metamaterial/LatticeModulus', file_name='data_node_num32')

    split_idx = dataset.get_idx_split(len(dataset), train_size=5, valid_size=5, seed=42)
    print(split_idx.keys())
    print(dataset[split_idx['train']])
    train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[
        split_idx['test']]
    train_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    print(dataset)
    data_batch = next(iter(train_loader))
    print('Property shape: ',data_batch.y.shape)
    node_num_per_lattice = dataset.slices['node_feat'][1:] - dataset.slices['node_feat'][:-1]
    edge_num_per_lattice = dataset.num_edges
    print('max node num', max(node_num_per_lattice))
    print('min node num', min(node_num_per_lattice))
    print('average node num', node_num_per_lattice.float().mean())

    print('max edge num', max(edge_num_per_lattice))
    print('average edge num', edge_num_per_lattice.float().mean())
    print('min edge num', min(edge_num_per_lattice))

    data = dataset[0]
    print(data.edge_index.dtype)
    print(data.y)
    print(data.cart_coords.dtype)

if __name__ == '__main__':
    main()
    # plot_lattice(data.cart_coords, data.edge_index.t())