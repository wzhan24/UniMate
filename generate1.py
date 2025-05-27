import os
import argparse

import numpy as np
import torch
import shutil
from runner import Runner
#from vis import plot_lattice_from_path
def clear_folder(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)
    else:
        print(f'The path {folder_path} does not exist or is not a directory.')

def reshape_edges(edge_index_list):
    edge_list = [[] for _ in range(gen_coords_list.shape[0])]
    for i in range(len(edge_index_list[0])):
        index = int((edge_index_list[0][i] + edge_index_list[1][i])/2/24)
        edge_list[index].append([edge_index_list[0][i].item()-24*index, edge_index_list[1][i].item()-24*index])
    #print(edge_list)
    return edge_list

# from utils import smact_validity, compute_elem_type_num_wdist, get_structure, compute_density_wdist, structure_validity
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./result_modulus_0122/model_39.pth', type=str, help='The directory for storing training outputs')
    # parser.add_argument('--dataset', type=str, default='perov_5', help='Dataset name, must be perov_5, carbon_24, or mp_20')
    parser.add_argument('--dataset', type=str, default='LatticeModulus', help='Dataset name, must be perov_5, carbon_24, or mp_20, LatticeModulus, LatticeStiffness')
    parser.add_argument('--data_path', type=str, default='/data/home/wzhan24/materialgen/material_data/', help='The directory for storing training outputs')
    parser.add_argument('--save_mat_path', type=str, default='generated_mat/0128', help='The directory for storing training outputs')

    parser.add_argument('--num_gen', type=int, default=100, help='Number of materials to generate')
    parser.add_argument('--device', type=str, default='cuda:1')

    args = parser.parse_args()


    assert args.dataset in ['perov_5', 'carbon_24', 'mp_20', 'LatticeModulus', 'LatticeStiffness'], "Not supported dataset"


    if args.dataset in ['perov_5', 'carbon_24', 'mp_20']:
        train_data_path = os.path.join('data', args.dataset, 'train.pt')
        if not os.path.isfile(train_data_path):
            train_data_path = os.path.join('data', args.dataset, 'train.csv')

        test_data_path = os.path.join('data', args.dataset, 'test.pt')
        if not os.path.isfile(test_data_path):
            train_data_path = os.path.join('data', args.dataset, 'test.csv')

        if args.dataset == 'perov_5':
            from config.perov_5_config_dict import conf
        elif args.dataset == 'carbon_24':
            from config.carbon_24_config_dict import conf
        else:
            from config.mp_20_config_dict import conf

        score_norm_path = os.path.join('data', args.dataset, 'score_norm.txt')


    elif args.dataset in ['LatticeModulus', 'LatticeStiffness']:
        data_path = os.path.join(args.data_path, args.dataset)
        if args.dataset == 'LatticeModulus':
            from config.LatticeModulus_config_dict import conf
        elif args.dataset == 'LatticeStiffness':
            from config.LatticeStiffness_config_dict import conf

        train_data_path, val_data_path = None, None
        score_norm_path = None
    
    print('loading model...')
    runner = Runner(conf, score_norm_path)
    runner.model.load_state_dict(torch.load(args.model_path))
    print('loading data...')
    runner.load_data(data_path, args.dataset, file_name='data')
    #runner.load_data(data_path, args.dataset)
    # codes to sample some conditions as input

    num_gen = args.num_gen
    # #coords = np.zeros((num_gen,24,3))
    # #print(runner.train_dataset[0]['y'])
    # ys = np.zeros((num_gen, len(runner.train_dataset[0]['y'][0])))
    # #ys = np.zeros((num_gen, runner.train_dataset[0]['y'].shape[0]))
    # for i in range(num_gen):
    #     ind = np.random.randint(0, len(runner.train_dataset))
    #     #ys[i] = np.array(runner.train_dataset[ind]['y'])*((np.random.random(ys[0].shape)-0.5)*0.1+1)
    #     ys[i] = np.array(runner.train_dataset[ind]['y'][0])
    #     #coords[i] = np.array(runner.train_dataset[ind]['cart_coords'])
    # np.savetxt('ys.csv', ys, delimiter=',')
    # print('finished saving y')
    # #ys = np.loadtxt('ys.csv', delimiter=',')
    # cond = torch.tensor(ys).to('cuda:0').float()
    # cond = torch.tensor(np.genfromtxt('ys.csv',delimiter=',')).to('cuda:0').float()
    # density = 0
    # #coords = torch.tensor(coords).to('cuda:0').float()

    #cond = torch.tensor([1.0,1.0,1.0,0.38,0.38,0.38,0.3,0.3,0.3,0.3,0.3,0.3]).repeat(num_gen,1).to(args.device).float()
    #density = torch.tensor([0.3]).repeat(num_gen,1).to(args.device).float()
    cond = torch.tensor([0.211,0.180,0.25,0.0747,0.0776,0.0956,0.3026,0.2578,0.2516,0.3506,0.3277,0.276]).repeat(num_gen,1).to(args.device).float()
    density = torch.tensor([0.192]).repeat(num_gen,1).to(args.device).float()
    gen_coords_list, edge_index_list = runner.generate(cond=cond, density=density, latent_dim=16)
    #print(gen_coords_list[0].shape)
    #print(edge_index_list[0].shape)
    gen_coords_list = gen_coords_list[0]
    edge_index_list = edge_index_list[0]
    edge_index_list = reshape_edges(edge_index_list)
    if not os.path.exists(args.save_mat_path):
        os.makedirs(args.save_mat_path)
    
    print('Saving lattice...')
    #if os.path.exists(args.save_mat_path):
    #    clear_folder(args.save_mat_path)
    for i in range(args.num_gen):
        lattice_name = os.path.join(args.save_mat_path, '{}_lattice_{}.npz'.format(args.dataset, i))
        print('Saving {}, atom_num {}'.format(lattice_name, gen_coords_list[i].shape[0]))
        #print(gen_frac_coords_list[i])
        #input()
        np.savez(lattice_name,
                frac_coords=gen_coords_list[i],
                edge_index=edge_index_list[i],
                # edge_index=np.array([]),
                )

    # print('Vis saving...')
    # path = args.save_mat_path
    # file_names = os.listdir(path)
    # save_path = os.path.join('./vis',args.save_mat_path)
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # else:
    #     clear_folder(save_path)
    # for file_name in file_names:
    #     save_dir = os.path.join(save_path, file_name[:-3] + 'png')
    #     plot_lattice_from_path(path, file_name, save_dir=save_path)

