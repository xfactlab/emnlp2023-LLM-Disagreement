import collections
from collections import defaultdict
import json
import os
import random
import numpy as np
import argparse
from tqdm import tqdm
from datasets import load_dataset

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    """ args for """
    parser = argparse.ArgumentParser(description="Args for Dataset")

    parser.add_argument('--data_dir',
                    default = "./data/chaosNLI_v1.0/",
                    type = str,
    )
    parser.add_argument('--data_type',
                    default = "qnli",
                    choices = ['snli','mnli','alphanli','Pavlick','anli','qnli','disnli'],
                    type = str,
    )
    parser.add_argument('--sample',
                    default = False,
                    type = str2bool,
    )
    parser.add_argument('--sample_size',
                    default = 100,
                    type = int,
    )
    parser.add_argument('--seed',
                    default = 0,
                    type = int,
    )    
    parser.add_argument('--dump',
                    default = True,
                    type = str2bool,
    )    
    parser.add_argument('--file_name',
                    default = None,
                    type = str,
    )    
    parser.add_argument('--out_dir',
                    default = None,
                    type = str,
    )
    args = parser.parse_args()
    return args


def load_json(data_dir, data_type):
    """
    Load

    Path: Local directory
    """
    data_list = [cand for cand in os.listdir(data_dir) if cand[-5:] == 'jsonl']

    cnli_data_dict = {}
    for data in data_list:
        data_name = data[:-6].split("_")[1]

        with open(os.path.join(data_dir, data), "r", encoding="utf-8") as f:
            data_temp = [json.loads(line) for line in f]

        cnli_data_dict[data_name] = data_temp
        print(f'Data # of {len(data_temp)} for {data_name} Dataset')

    print(f'Total Data # of : {sum([len(v) for _,v in cnli_data_dict.items()])}')

    return cnli_data_dict


def load_data(data_type):
    """
    Load

    Path: Huggingface library
    """
    
    if data_type == "anli":
        dataset = load_dataset(data_type)
        dataset = dataset['test_r3']
        label_dict = {0:'e', 1:'n', 2:'c'}
    elif data_type == "qnli":
        dataset = load_dataset('glue', data_type)
        dataset = dataset['validation']
        label_dict = {0:'e', 1:'c'}
        
    print(f'Total Data # of : {len(dataset)}')

    return dataset, label_dict


def load_two_data(data_dir):
    """
    Load

    Path: Local directory
    """
    json_data = defaultdict(list)
    idx = 0
    
    uid_lst = []
    with open(data_dir, "r") as json_file:
        for line in json_file:
            sample = json.loads(line)
            if sample['a1'][0] in ['Lexical', 'Implicature', 'Presupposition',
                                   'Probabilistic', 'Imperfection', 'Coreference',
                                   'Temporal', 'Interrogative', 'Accomodating'
                                   'High Overlap']:  
                uid = sample['pairID']
                json_data[uid] = json.loads(line)
                uid_lst.append(uid)
                idx += 1

    print(f'Total Data # of : {len(json_data)}')

    return json_data, uid_lst


def generate_sample(data, data_type,
                    sample = True, sample_size=100, seed=0, dump = False, 
                    file_name = 'dict_sampled', out_dir = None, **kwargs):

    if data_type in ['snli','mnli','alphanli']:
        data = data[data_type]
    elif args.data_type in ['anli','qnli']:
        label_dict = kwargs['label_dict']
    elif data_type == 'disnli':
        uid_lst = kwargs['uid_lst']
        orig_data_dir = f'../data/inputs/dict_mnli.json'
        with open(orig_data_dir, "r") as json_file:
            orig_json_data = json.load(json_file)

    # if sample and data_type != 'disnli':
    #     np.random.seed(seed)
    #     rand_idx = np.random.choice(list(range(total_num)), 
    #                                     sample_size, replace=False)

    #     sampled_data = [data[i] for i in range(total_num) if i in rand_idx]
    # else:
    #     sampled_data = data
    print(len(data))
    
    dict_sampled_data = collections.defaultdict(dict)
    
    if data_type != 'disnli':
        for new_ind, sampl in enumerate(data):
            if data_type in ['snli','mnli','alphanli']:
                dict_sampled_data[new_ind]['uid']= sampl['uid']
                dict_sampled_data[new_ind]['label_counter']= sampl['label_counter']
                dict_sampled_data[new_ind]['label_counter']= sampl['label_counter']
                dict_sampled_data[new_ind]['old_label']= sampl['old_label']
                dict_sampled_data[new_ind]['majority_label']= sampl['majority_label']

            if data_type == "alphanli":
                dict_sampled_data[new_ind]['observation'] = [sampl['example']['obs1'],sampl['example']['obs2']]
                dict_sampled_data[new_ind]['hypothesis']= [sampl['example']['hyp1'],sampl['example']['hyp2']]

            elif data_type in ['snli','mnli']:
                dict_sampled_data[new_ind]['premise']= sampl['example']['premise']
                dict_sampled_data[new_ind]['hypothesis']= sampl['example']['hypothesis'] 

            elif data_type == "anli":
                dict_sampled_data[new_ind]['uid']= sampl['uid']
                dict_sampled_data[new_ind]['premise']= sampl['premise']
                dict_sampled_data[new_ind]['hypothesis']= sampl['hypothesis'] 
                dict_sampled_data[new_ind]['majority_label']= label_dict[sampl['label']]
                dict_sampled_data[new_ind]['reason']= sampl['reason']

            elif data_type == "qnli":
                dict_sampled_data[new_ind]['uid']= sampl['idx']
                dict_sampled_data[new_ind]['premise']= sampl['question']
                dict_sampled_data[new_ind]['hypothesis']= sampl['sentence']
                dict_sampled_data[new_ind]['majority_label']= label_dict[sampl['label']]
    else:
        for i in range(len(orig_json_data)):
            orig_sample = orig_json_data[str(i)]
            uid = orig_sample['uid']
            if uid in uid_lst:
                new_sample = data[str(uid)]
                temp_dic = {}
                for orig_key in orig_sample.keys():
                    temp_dic[orig_key] = orig_sample[orig_key]
                for new_key in new_sample.keys():
                    if new_key not in temp_dic:
                        temp_dic[new_key] = new_sample[new_key]
                dict_sampled_data[i] = temp_dic

    if dump:
        if not file_name:
            file_name = 'dict_' + data_type

        if not out_dir:
            out_dir = os.path.join(os.curdir, 'data')

        json_file_name = os.path.join(out_dir, f'{file_name}.json')

        with open(json_file_name, 'w') as f:
            json.dump(dict_sampled_data, f, indent=2)

        print(f'JSON File dumped as {file_name}.json')

    return dict_sampled_data


if __name__ == '__main__':
    if not os.path.isdir('data'):
        os.mkdir('data')

    args = get_args()
    
    if args.data_type in ['snli','mnli','alphanli']:
        data = load_json(args.data_dir, args.data_type)
        generate_sample(data, args.data_type,
                        args.sample, args.sample_size, args.seed, args.dump, 
                        args.file_name, args.out_dir)
        
    elif args.data_type in ['anli','qnli']:
        data, label_dict = load_data(args.data_type)
        generate_sample(data, args.data_type,
                        args.sample, args.sample_size, args.seed, args.dump, 
                        args.file_name, args.out_dir, label_dict=label_dict)
    
    elif args.data_type == 'disnli':
        data, uid_lst = load_two_data(args.data_dir)
        generate_sample(data, args.data_type, 
                        args.sample, args.sample_size, args.seed, args.dump, 
                        args.file_name, args.out_dir, uid_lst=uid_lst)
