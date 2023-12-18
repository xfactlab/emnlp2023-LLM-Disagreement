import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import collections
import argparse

from tqdm.auto import tqdm
from scipy.stats import entropy, wasserstein_distance
from scipy.spatial.distance import jensenshannon

from utils.metrics import kl_divergence, ent_ce, rank_cs, tvd
from utils.prompts import prompt_generator


def get_args():
    """ args for """
    parser = argparse.ArgumentParser(description="Args for Evaluator")

    parser.add_argument('--data_dir',
                    default = "./data/old/postprocessed_outputs/text-davinci-003/dict_alphanli_logprob_mc_dist_text-davinci-003_pp_prompt1.json",
                    type = str,
    )
    parser.add_argument('--data_type',
                    default = "alphanli",
                    choices = ['snli','mnli','alphanli','Pavlick','anli','qnli'],
                    type = str,
    )
    parser.add_argument('--gen_type',
                    default = "mce",
                    choices = ['mce','lpe'],
                    type = str,
    )
    parser.add_argument('--prompt_type',
                    default = 'p2',
                    choices = ['p1', 'p2'],
                    type = str,
    )
    parser.add_argument('--model',
                    default = "flan-t5-xxl",
                    type = str,
    )
    args = parser.parse_args()
    return args


def score_distribution(mc_data, data_type, gen_type):
    if data_type == 'alphanli':
        order = ['1', '2']
    elif data_type == 'qnli':
        order = ['y', 'n']
        conv_dic = {'e':'y', 'c':'n'}
    else:
        order = ['e', 'n', 'c']
        
    orig_dist, mc_dist = [],[]

    for idx,_ in mc_data.items():
        try:
            if gen_type == 'lpe':
                mc_dist.append([mc_data[idx]['lpe_dist'][ans] for ans in order])
            else:
                mc_dist.append([mc_data[idx]['mc_dist'][ans] for ans in order])
        except:
            mc_dist.append([1/len(order)]*len(order))
        try:
            orig_dist.append([mc_data[idx]['orig_dist'][ans] for ans in order])
        except:
            orig_dist.append([1/len(order)]*len(order))        

    orig_arr = np.array(orig_dist)
    mc_arr = np.array(mc_dist)
    assert(orig_arr.shape == mc_arr.shape)

    kl_div_wo_zero, kl_div, js_dis, js_div, old_acc, new_acc, ent_ce_val, rank_cs_val, tvd_val, wass_dist = [0]*10
    prompt_lengths, run_times, valid_ratios = [], [], []
    
    for (i, key) in enumerate(mc_data.keys()):
        if args.data_type in ['snli', 'mnli','Pavlick','anli']:
            if args.prompt_type == "p1":
                template_num = 1
            elif args.prompt_type == "p2":
                template_num = 4

        elif args.data_type == 'alphanli':
            if args.prompt_type == "p1":
                template_num = 2
            elif args.prompt_type == "p2":
                template_num = 5        
        
        elif args.data_type == 'qnli':
            if args.prompt_type == "p1":
                template_num = 3
            elif args.prompt_type == "p2":
                template_num = 6               
        
        input = prompt_generator(mc_data[key], args.model, args.data_type,
                                 template_type=template_num, shuffle=True)
        prompt_lengths.append(len(input))
        
        run_times.append(mc_data[key]['run_time'])
        valid_ratios.append(mc_data[key]['valid_ratio'])
        
        cur_js_dis = jensenshannon(orig_arr[i], mc_arr[i])
        js_dis += cur_js_dis
        js_div += cur_js_dis**2
        
        cur_kl_div = entropy(orig_arr[i], mc_arr[i])      
        if np.isinf(cur_kl_div): 
            temp2 = np.where(mc_arr[i] == 0, 1e-10, mc_arr[i])
            cur_kl_div = entropy(orig_arr[i], temp2)
        kl_div +=  cur_kl_div
        kl_div_wo_zero += kl_divergence(orig_arr[i], mc_arr[i], True)
        
        new_label = order[np.argmax(mc_arr[i])]

        if data_type == 'alphanli':
            new_label = int(new_label)
            
        if data_type in ['snli','mnli','alphanli']:
            old_acc += (new_label==mc_data[key]['old_label'])
        elif data_type == 'Pavlick':
            old_acc += (new_label==mc_data[key]['original-dataset-label'][0])
        else:
            old_acc += 0
        
        if data_type == 'qnli':
            new_acc += (new_label==conv_dic[mc_data[key]['majority_label']])
        else:
            new_acc += (new_label==mc_data[key]['majority_label'])

        ent_ce_val += ent_ce(orig_arr[i], mc_arr[i])
        rank_cs_val += rank_cs(orig_arr[i], mc_arr[i])
        tvd_val += tvd(orig_arr[i], mc_arr[i])
        wass_dist += wasserstein_distance(orig_arr[i], mc_arr[i])

        metric = {
            'Old_Acc' : old_acc / len(mc_data),
            'New_Acc' : new_acc / len(mc_data),
            'KLD_wo_Zeros' : kl_div_wo_zero / len(mc_data),
            'KLD': kl_div / len(mc_data),
            'JSDist' : js_dis / len(mc_data),
            'JSD': js_div / len(mc_data),
            'EntCE' : ent_ce_val / len(mc_data),
            'RankCS' : rank_cs_val / len(mc_data),
            'TVD' : tvd_val / len(mc_data),
            'WassDist' : wass_dist /len(mc_data)
        }
        
    print(f'Old Accuracy: {metric["Old_Acc"]*100:.2f}%')
    print(f'New Accuracy: {metric["New_Acc"]*100:.2f}')
    # print(f'KL Divergence wo/ Zeros: {metric["KLD_wo_Zeros"]:.4f}')
    print(f'KL Divergence: {metric["KLD"]:.4f}')
    print(f'JS Distance: {metric["JSDist"]:.4f}')
    print(f'JS Divergence: {metric["JSD"]:.4f}')
    print(f'Entropy Calibration Error: {metric["EntCE"]:.4f}')
    print(f'Ranking Calibration Score: {metric["RankCS"]:.4f}')
    print(f'Distribution Calibration Error: {metric["TVD"]:.4f}')
    print(f'Wasserstein Distance: {metric["WassDist"]:.4f}')
    print(f'Run Time: {sum(run_times)/len(run_times):.2f} sec. (SD: {np.std(run_times):.2f})')
    print(f'Valid Ratio: {sum(valid_ratios)/len(valid_ratios)}% (SD: {np.std(valid_ratios):.2f})')
    print(f'Prompt Length: {sum(prompt_lengths)/len(prompt_lengths)} (SD: {np.std(prompt_lengths):.2f})')

    return metric


if __name__ == '__main__':
    args = get_args()
    
    with open(args.data_dir, "r", encoding="utf-8") as f:
        mc_data = json.load(f)
    mc_data = {int(k):v for k,v in mc_data.items()}
    print(f"{len(mc_data)} # of data loaded...")
    #mc_data = {key: mc_data[key] for key in range(0, 300)}
    metrics = score_distribution(mc_data, args.data_type, args.gen_type)



    

