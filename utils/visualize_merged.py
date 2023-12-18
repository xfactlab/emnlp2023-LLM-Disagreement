import os
import sys
import json
import math
import argparse
import pandas as pd
from pathlib import Path
from itertools import combinations_with_replacement
import collections
from collections import defaultdict
import numpy as np
import seaborn as sns
from seaborn.objects import Text
import matplotlib.pyplot as plt

from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon


def get_args():
    """ args for """
    parser = argparse.ArgumentParser(description="Args for Merged Visualizer")

    parser.add_argument('--data_type',
                        default = "alphanli",
                        choices = ['snli','mnli','alphanli','Pavlick'],
                        type = str,
    )
    parser.add_argument('--model_type',
                        default = "all",
                        choices = ["notopenai","text-davinci-003","openai","Nie","all"],
                        type = str,
    )
    parser.add_argument('--visualize_type',
                        default = "heatmap",
                        choices = ["rainbow","heatmap"],
                        type = str,
    )
       
    args = parser.parse_args()
    return args

def jitter(values):
    np.random.seed(77)
    return values + np.random.normal(0, 0.1, values.shape)

def pltcolor(hum_arr, mc_arr):
    hum_lst, mc_lst = hum_arr.tolist(), mc_arr.tolist()
    
    cols=[]
    for i in range(len(hum_lst)):
        diff = abs(hum_lst[i] - mc_lst[i])
        if diff < 0.1:
            cols.append('tomato')
        elif diff < 0.2:
            cols.append('orange')
        elif diff < 0.3:
            cols.append('gold')
        elif diff < 0.4:
            cols.append('limegreen')
        elif diff < 0.5:
            cols.append('lightblue')
        elif diff < 0.6:
            cols.append('darkviolet')
        else:
            cols.append('violet')
    return np.array(cols)
    
def compare_LLMs(mc_data, data_type, model_type, visualize_type, figure_dir):
    if data_type != 'alphanli':
        order = ['e', 'n', 'c']
    else:
        order = ['1', '2']
    
    low_lim = -0.30
    if data_type == 'alphanli':
        high_lim = 1.0
        high_tick = 0.9
    else:
        high_lim = 1.30
        high_tick = 1.1
        
#     sns.set_context("paper", rc={"axes.titlesize":13,"axes.labelsize":12, "xtick.labelsize":8, "ytick.labelsize":8})   
#     sns.set_style('whitegrid', rc={'font.family': 'serif', 'font.serif': 'Times New Roman', 'axes.grid' : True, "grid.linestyle": ":", "grid.linewidth":0.5})
    
    if model_type  != 'text-davinci-003':        
        if model_type == 'notopenai':
            model_types = ['flan-t5-large','flan-t5-xl','flan-t5-xxl','flan-ul2',
                           'opt-iml-max-1.3b', 'opt-iml-max-30b',
                           'text-davinci-003', 'text-davinci-002', 'stable_vicuna_13b']
        elif model_type == 'openai':
            model_types = ['text-davinci-002','text-davinci-003','gpt-3.5-turbo']
        elif model_type == 'Nie':
            model_types = ['bert-base','bert-large','roberta-base','roberta-large']
        elif model_type == 'all':
             model_types = ['bert-base','bert-large','roberta-base','roberta-large',
                            'flan-t5-large','flan-t5-xl','flan-t5-xxl','flan-ul2',
                            'opt-iml-max-1.3b', 'opt-iml-max-30b',
                            'text-davinci-002', 'text-davinci-003', 'stable_vicuna_13b']
        Nie_model_types = ['bert-base','bert-large','roberta-base','roberta-large']
        
        model_num = len(model_types)
        if visualize_type == "rainbow":
            COLUMN_NUM = 1
            ROW_NUM = model_num // COLUMN_NUM
            fig, ax = plt.subplots(ROW_NUM, COLUMN_NUM, figsize=(COLUMN_NUM * 3.3, ROW_NUM * 3))
        elif visualize_type == "heatmap":
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
       
        if visualize_type == "rainbow":
            for (i, model) in enumerate(model_types):
                mc_entropy_lst, hum_entropy_lst = [], []

                data_dict = mc_data[model]
                for idx in data_dict.keys():
                    if model_type in ["Nie", "all"]:
                        mc_entropys = list(data_dict[idx]['mc_dist'].values())
                        hum_entropys = list(data_dict[idx]['orig_dist'].values())
                    else:
                        mc_entropys = data_dict[idx]['mc_dist']
                        hum_entropys = data_dict[idx]['orig_dist']

                    mc_entropy_lst.append(entropy(mc_entropys))
                    hum_entropy_lst.append(entropy(hum_entropys))

                mc_entropy_arr = np.array(mc_entropy_lst)
                hum_entropy_arr = np.array(hum_entropy_lst)

                jit_mc_entropy_arr = jitter(mc_entropy_arr)
                jit_hum_entropy_arr = jitter(hum_entropy_arr)  

                color = pltcolor(jit_hum_entropy_arr, jit_mc_entropy_arr)

                ax[i].scatter(jit_hum_entropy_arr, jit_mc_entropy_arr, c=color, s=2)
                ax[i].set_xlim([low_lim, high_lim])
                if i == model_num - 1:
                    ax[i].set_xlabel('Human entropys')
                ax[i].set_xticks(np.arange(0, high_tick, 0.2))
                ax[i].set_ylim([low_lim, high_lim])
                ax[i].set_ylabel(f'{model.capitalize()} entropys')
                ax[i].set_yticks(np.arange(0, high_tick, 0.2))
            
        elif visualize_type == "heatmap":
            model_combs = list(combinations_with_replacement(model_types, 2))
            heatmap_dict = defaultdict(float)
            for i,(model1,model2) in enumerate(model_combs):
                if model1 not in ['text-davinci-002', 'text-davinci-003', 'stable_vicuna_13b']:
                    dist_type1 = 'mc_dist'
                else:
                    dist_type1 = 'lpe_dist'
                if model2 not in ['text-davinci-002', 'text-davinci-003', 'stable_vicuna_13b']:
                    dist_type2 = 'mc_dist'
                else:
                    dist_type2 = 'lpe_dist'
                    
                data_dict1 = mc_data[model1]
                data_dict2 = mc_data[model2]
#                 print(model1, model2, len(data_dict1), len(data_dict2))
                js_div_lst = []
                for idx in data_dict1.keys():
                    if model_type == "all":
                        if model1 in Nie_model_types:
                            mc_dist1 = data_dict1[idx][dist_type1]
                        else:
                            data_dict1[idx][dist_type1] = {str(k):v for k,v in data_dict1[idx][dist_type1].items()}
                            try:
                                mc_dist1 = [data_dict1[idx][dist_type1][category] for category in order]
                            except KeyError:
                                mc_dist1 = [1/len(order)] * len(order)
                        if model2 in Nie_model_types:
                            mc_dist2 = data_dict2[idx][dist_type2]
                        else:
                            data_dict2[idx][dist_type2] = {str(k):v for k,v in data_dict2[idx][dist_type2].items()}
                            try:
                                mc_dist2 = [data_dict2[idx][dist_type2][category] for category in order]
                            except KeyError:
                                mc_dist2 = [1/len(order)] * len(order)

                    elif model_type == "Nie":
                        mc_dist1 = list(data_dict1[idx][dist_type1].values())
                        mc_dist2 = list(data_dict2[idx][dist_type1].values())
                    else:
                        try:
                            mc_dist1 = [data_dict1[idx][dist_type1][ans] for ans in order]
                        except:
                            mc_dist1 = [1/len(order)] * len(order)
                        try:
                            mc_dist2 = [data_dict2[idx][dist_type2][ans] for ans in order]
                        except:
                            mc_dist2 = [1/len(order)] * len(order)
                    
                    try:
                        cur_js_dis = jensenshannon(mc_dist1, mc_dist2).item()
                        js_div_lst.append(cur_js_dis)
                    except Exception as e:
                        print(e)
                        print(model1, model2, mc_dist1, mc_dist2)
                    
                heatmap_dict[(model1, model2)] = np.nansum(js_div_lst) / len(js_div_lst)
            for (j, pair) in enumerate(heatmap_dict.keys()):
                jsd = heatmap_dict[pair]
                model1, model2 = pair
                if j == 0:
                    heatmap_df = pd.DataFrame([[model1, model2, jsd]])
                else:
                    heatmap_df = pd.concat([heatmap_df, pd.DataFrame([[model1, model2, jsd]])])
                    
            heatmap_df = pd.pivot_table(heatmap_df, index=[0], columns=[1], values=[2], sort=False)
            heatmap_df.index = ['Flan-T5-L','Flan-T5-XL','Flan-T5-XXL','Flan-UL2',
                                'OPT-IML-M-S', 'OPT-IML-M-L',
                                'GPT-3.5-D3', 'GPT-3.5-D2', 'Stable Vicuna']#[x.capitalize() for x in heatmap_df.index]
            heatmap_df.columns = heatmap_df.index #[y.capitalize() for _, y in heatmap_df.columns.to_flat_index()]
            print(heatmap_df)
            s = sns.heatmap(heatmap_df,
                            vmin=0.0, vmax=0.5,
                            annot=True, square=True, cbar=False, linewidth=0.5,
                            annot_kws={"size":9})
            s.set_xlabel('LLMs', fontsize=10)
            s.set_ylabel('LLMs', fontsize=10)
            

    elif model_type == "text-davinci-003":
        plt.figure(figsize=(3, 3))
        
        mc_entropy_lst, hum_entropy_lst = [], []

        for idx in mc_data.keys():
            data_dict = mc_data[idx]
            
            mc_entropys = list(data_dict['mc_dist'].values())
            hum_entropys = list(data_dict['orig_dist'].values())

            mc_entropy_lst.append(entropy(mc_entropys))
            hum_entropy_lst.append(entropy(hum_entropys))

        jit_mc_entropy_arr = jitter(np.array(mc_entropy_lst))
        jit_hum_entropy_arr = jitter(np.array(hum_entropy_lst))  

        color = pltcolor(jit_hum_entropy_arr, jit_mc_entropy_arr)

        plt.scatter(jit_hum_entropy_arr, jit_mc_entropy_arr, c=color, s=2)
        plt.xlim([low_lim, high_lim])
        plt.xlabel('Human entropys')
        plt.xticks(np.arange(0, high_tick, 0.2))
        plt.ylim([low_lim, high_lim])
        plt.ylabel(f'{model_type.capitalize()} entropys')
        plt.yticks(np.arange(0, high_tick, 0.2))

    plt.tight_layout()
    plt.savefig(figure_dir, dpi=300)
    
    
if __name__ == '__main__':
    args = get_args()
    
    if args.model_type not in ['text-davinci-003', "Nie", "all"]:
        data_dir = f"../data/final/merged_postprocessed_outputs/dict_{args.data_type}_mc_dist_all.json"
    elif args.model_type == "Nie":
        data_dir = f"../data/final/outputs/{args.model_type}/dict_{args.data_type}_mc_dist_{args.model_type}.json"
    elif args.model_type == "all":
        data_dir = f"../data/final/merged_postprocessed_outputs/dict_{args.data_type}_mc_dist_{args.model_type}.json"
    else:
        data_dir = f"../data/final/postprocessed_outputs/{args.model_type}/dict_{args.data_type}_logprob_mc_dist_{args.model_type}_pp.json"
     
    with open(Path(data_dir), "r", encoding="utf-8") as f:
        mc_data = json.load(f)
    mc_data = {model_type:v for model_type,v in mc_data.items()}
    print(f"Types of models loaded: {list(mc_data.keys())}")
    
    if args.visualize_type == "rainbow":
        figure_dir = f"../data/final/merged_figures/rainbow_{args.data_type}_mc_dist_{args.model_type}.png"
    elif args.visualize_type == "heatmap":
        figure_dir = f"../data/final/merged_figures/heatmap_{args.data_type}_mc_dist_{args.model_type}.png"
    
    compare_LLMs(mc_data, args.data_type, args.model_type, args.visualize_type, Path(figure_dir))
        