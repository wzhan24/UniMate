import os
import argparse

import numpy as np

from runner import Runner

DEVICE_ID = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_ID

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, default='result_modulus_0122/', help='The directory for storing training outputs')
    parser.add_argument('--dataset', type=str, default='LatticeModulus', help='Dataset name, must be perov_5, carbon_24, or mp_20, LatticeModulus, LatticeStiffness')
    #parser.add_argument('--data_path', type=str, default='/data/home/wzhan24/materialgen/material_data/', help='Data path')
    parser.add_argument('--data_path', type=str, default='./dataset/', help='Data path')
    parser.add_argument('--load_model_path', type=str, default='result_modulus_32/model_4999.pth', help='load checkpoint')
    args = parser.parse_args()

    result_path = args.result_path
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    assert args.dataset in ['perov_5', 'carbon_24', 'mp_20', 'LatticeModulus', 'LatticeStiffness'], "Not supported dataset"


    if args.dataset in ['perov_5', 'carbon_24', 'mp_20']:
        train_data_path = os.path.join('data', args.dataset, 'train.pt')
        if not os.path.isfile(train_data_path):
            train_data_path = os.path.join('data', args.dataset, 'train.csv')

        val_data_path = os.path.join('data', args.dataset, 'val.pt')
        if not os.path.isfile(val_data_path):
            val_data_path = os.path.join('data', args.dataset, 'val.csv')

        if args.dataset == 'perov_5':
            from config.perov_5_config_dict import conf
        elif args.dataset == 'carbon_24':
            from config.carbon_24_config_dict import conf
        else:
            from config.mp_20_config_dict import conf

        score_norm_path = os.path.join('data', args.dataset, 'score_norm.txt')


    elif args.dataset in ['LatticeModulus', 'LatticeStiffness']:
        #data_path = os.path.join(args.data_path, args.dataset) #ini for data_path
        data_path = args.data_path
        if args.dataset == 'LatticeModulus':
            from config.LatticeModulus_config_dict import conf
        elif args.dataset == 'LatticeStiffness':
            from config.LatticeStiffness_config_dict import conf

        train_data_path, val_data_path = None, None
        score_norm_path = None

    #print(conf)
    np.save(result_path+'config.npy', conf)


    runner = Runner(conf, score_norm_path)
    dataset = runner.load_data(data_path, args.dataset, file_name='data')
    #dataset = runner.load_data(data_path, args.dataset)
    #runner.train(train_data_path, val_data_path, result_path, load_model_path=args.load_model_path)
    runner.train(train_data_path, val_data_path, result_path)