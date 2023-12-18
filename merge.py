import os
import sys
import json
import math
import argparse
from glob import glob
from pathlib import Path
import collections
from collections import defaultdict


def get_args():
    """ args for """
    parser = argparse.ArgumentParser(description="Args for Merger")

    parser.add_argument('--data_dirs',
                        default = "../data/old/postprocessed_outputs",
                        type = str,
    )
    parser.add_argument('--data_type',
                        default = "snli",
                        choices = ['snli','mnli','alphanli','Pavlick'],
                        type = str,
    )
    parser.add_argument('--model_type',
                    default = "all",
                    choices = ['notopenai','openai','all'],
                    type = str,
    )
    parser.add_argument('--out_dir',
                    default = "../data/old/merged_postprocessed_outputs",
                    type = str,
    )
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    
    dirs_lst = glob(args.data_dirs + f"/dict_{args.data_type}*")
    json_lst = [file for file in dirs_lst if file.endswith('.json')]
    
    merged_mc_data = {}
    for json_path in json_lst:
        with open(json_path, "r", encoding="utf-8") as f:
            orig_data = json.load(f)
        if json_path.split('_')[-1].split('.')[0] == "Nie":
            for model_type in orig_data.keys():
                model_data = orig_data[model_type]
                mc_data = {int(k):v for k,v in model_data.items()}
                merged_mc_data[model_type] = mc_data
                
        else:
            mc_data = {int(k):v for k,v in orig_data.items()}       
            json_file = os.path.basename(json_path)
            json_file = json_file.split('dist_')[-1]
            if 'p1' in json_file:
                model_type = json_file.split('_p1')[0]
            elif 'p2' in json_file:
                model_type = json_file.split('_p2')[0]
            merged_mc_data[model_type] = mc_data
     
    json_file = os.path.join(args.out_dir, f'dict_{args.data_type}_mc_dist_{args.model_type}.json')
    with open(json_file, 'w') as f:
        json.dump(merged_mc_data, f, indent=2)

    print(f'Merged JSON File dumped as {os.path.basename(json_file)}')
