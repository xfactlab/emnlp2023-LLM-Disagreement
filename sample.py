import json
import os
import argparse
import random
from scipy.stats import entropy


def get_args():
    """ args for """
    parser = argparse.ArgumentParser(description="Args for Generater")
    parser.add_argument('--data_dir',
                    default = "./data/inputs/dict_snli.json",
                    type = str,
                        )
    parser.add_argument('--file_name',
                        default=None,
                        type=str,
                        )
    parser.add_argument('--out_dir',
                        default=None,
                        type=str,
                        )
    parser.add_argument('--sample_size',
                        default=100,
                        type=int,
                        )
    parser.add_argument('--sample_strategy',
                        default='random',
                        choices=['random','entropy'],
                        type=str,
                        )
    parser.add_argument('--data_type',
                        default='alphanli',
                        choices=['alphanli','mnli', 'snli', 'Pavlick'],
                        type=str,
                        )
    parser.add_argument('--few_shot_k',
                        default=0,
                        type=int,
                        )
    args = parser.parse_args()
    return args


def random_sample(data, total_indices, sample_size, few_shot_k, seed=22):
    random.seed(seed)
    ran_indices = random.sample(total_indices, sample_size)
    sampled_data = {key:data[key] for key in ran_indices}
    
    if few_shot_k != 0:
        excluded_indices = [idx for idx in total_indices if idx not in ran_indices]
        random.seed(seed)
        few_shot_indices = random.sample(excluded_indices, few_shot_k)
        few_shot_data = {key:data[key] for key in few_shot_indices}        
        return sampled_data, few_shot_data
    else:
        return sampled_data

def dict_postprocess(instance, data_type):   
    if data_type == 'alphanli':
        cand = ['1','2']
    else:
        cand = ['e', 'c', 'n']
    
    if len(instance['label_counter'].keys()) < len(cand):
        for can in cand:
            if can not in instance['label_counter'].keys():
                instance['label_counter'][can] = 0

    orig_dist = {k : v/sum(list(instance['label_counter'].values())) for k,v in instance['label_counter'].items()}

    return orig_dist

def entropy_sample(data, process_fn, data_type, sample_size):
    for _, dict_ in data.items():
        dict_['human_ent'] = entropy(list(process_fn(dict_, data_type).values()))

    temp = sorted(data.items(), key=lambda x:-x[1]['human_ent'])[:sample_size]

    return {k:v for k,v in temp}


if __name__ == '__main__':
    args = get_args()

    with open(args.data_dir, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = {int(k):v for k,v in data.items()}
    total_indices = range(len(data))    

    if args.sample_strategy == 'random':
        if args.few_shot_k != 0:
            sampled_data, few_shot_data = random_sample(data, total_indices, args.sample_size, args.few_shot_k)
        else:
            sampled_data = random_sample(data, total_indices, args.sample_size, args.few_shot_k)

    elif args.sample_strategy == 'entropy':
        sampled_data = entropy_sample(data, dict_postprocess, args.data_type, args.sample_size)

    if not args.file_name:
        file_name = 'dict_sampled_' + args.data_type
    else:
        file_name = args.file_name

    if not args.out_dir:
        out_dir = os.path.join(os.curdir, 'data')
    else:
        out_dir = args.out_dir

    json_file = os.path.join(out_dir, f'{file_name}.json')

    with open(json_file, 'w') as f:
        json.dump(sampled_data, f, indent=2)
    print(f'Total of {len(sampled_data)} data saved to {os.path.basename(json_file)}')
    
    if args.few_shot_k != 0:
        fs_json_file = os.path.join(out_dir, f'{file_name}_few_shot.json')
        with open(fs_json_file, 'w') as f:
            json.dump(few_shot_data, f, indent=2)
        print(f'Total of {len(few_shot_data)} data saved to {os.path.basename(fs_json_file)}')