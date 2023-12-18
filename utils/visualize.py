import os
import sys
import json
import math
import argparse
from pathlib import Path
import collections
from collections import defaultdict
import random
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from scipy.special import kl_div
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from wordcloud import WordCloud, STOPWORDS

from metrics import kl_divergence, jensen_shannon, ent_ce, rank_cs, tvd


def get_args():
    """ args for """
    parser = argparse.ArgumentParser(description="Args for Visualizer")

    parser.add_argument('--data_dir',
                        default = "./data/outputs/dict_snli_500_mc_dist_flant5-xxl.json",
                        type = str,
    )
    parser.add_argument('--data_type',
                        default = "snli",
                        choices = ['snli','mnli','alphanli','Pavlick','Nie'],
                        type = str,
    )
    parser.add_argument('--model_type',
                        default = "flant5-xxl",
                        choices = ['flant5-large','flant5-xl','flant5-xxl','opt-iml-max-s','text-davinci-003'],
                        type = str,
    )
    parser.add_argument('--figure_dir_1',
                        default = "./data/figures/pic_snli_500_mc_dist_flant5-xxl_1.png",
                        type = str,
    )
    parser.add_argument('--entropy_type',
                        default = "high",
                        type = str,
    )
    parser.add_argument('--figure_dir_2',
                        default = "./data/figures/pic_snli_500_mc_dist_flant5-xxl_2.png",
                        type = str,
    )
    parser.add_argument('--figure_dir_3',
                        default = "./data/figures/pic_snli_500_mc_dist_flant5-xxl_3.png",
                        type = str,
    )
    parser.add_argument('--figure_dir_4',
                        default = "./data/figures/pic_snli_500_mc_dist_flant5-xxl_4.png",
                        type = str,
    )
    parser.add_argument('--figure_dir_5',
                        default = "./data/figures/pic_snli_500_mc_dist_flant5-xxl_5.png",
                        type = str,
    )
    parser.add_argument('--figure_dir_6',
                        default = "./data/figures/pic_snli_500_mc_dist_flant5-xxl_4.png",
                        type = str,
    )
    parser.add_argument('--figure_dir_7',
                        default = "./data/figures/pic_snli_500_mc_dist_flant5-xxl_5.png",
                        type = str,
    )   
    
    args = parser.parse_args()
    return args


def preprocess_data(json_data, data_type, ordered_relations, new_ordered_relations):
        
    for key in json_data.keys():
        # LLM
        n = sum(json_data[key]['mc_count'].values())
        json_data[key]['norm_mc_dist'] = {c : v / n for (c, v) in zip(json_data[key]['mc_count'].keys(), json_data[key]['mc_count'].values())}
        json_data[key]['norm_mc_dist'] = {k : json_data[key]['norm_mc_dist'][k] for k in ordered_relations}
        mc_entropy = entropy(np.array(list(json_data[key]['norm_mc_dist'].values())))
        json_data[key]['mc_entropy'] = mc_entropy

        norm_mc_dist = json_data[key]['norm_mc_dist']
        ambiguous_labels = {k for (k, v) in norm_mc_dist.items() if (v > 0.2) and (v < 0.6)}
        if len(ambiguous_labels) > 1:
            ambiguous_labels = sorted(ambiguous_labels)
            new_label = ''.join(ambiguous_labels)
        else:
            confident_labels = {k for (k, v) in norm_mc_dist.items() if (v > 0.8)}
            if len(confident_labels) == 1:
                new_label = max(norm_mc_dist, key=norm_mc_dist.get)
            else:
                new_label = None
        json_data[key]['mc_entropy_per_newlabel'] = {new_label : mc_entropy}

        # Human
        m = sum(json_data[key]['label_counter'].values())
        json_data[key]['norm_hum_dist'] = {c : v / m for (c, v) in zip(json_data[key]['label_counter'].keys(), json_data[key]['label_counter'].values())}
        json_data[key]['norm_hum_dist'] = {k : json_data[key]['norm_hum_dist'][k] for k in ordered_relations}
        hum_entropy = entropy(np.array(list(json_data[key]['norm_hum_dist'].values())))
        json_data[key]['hum_entropy'] = hum_entropy

        norm_hum_dist = json_data[key]['norm_hum_dist']
        ambiguous_labels = {k for (k, v) in norm_hum_dist.items() if (v > 0.2) and (v < 0.6)}
        if len(ambiguous_labels) > 1:
            ambiguous_labels = sorted(ambiguous_labels)
            new_label = ''.join(ambiguous_labels)
        else:
            confident_labels = {k for (k, v) in norm_hum_dist.items() if (v > 0.8)}
            if len(confident_labels) == 1:
                new_label = max(norm_hum_dist, key=norm_hum_dist.get)
            else:
                new_label = None
        json_data[key]['hum_entropy_per_newlabel'] = {new_label : hum_entropy}

        json_data[key]['count_usage'] = sum(list(json_data[key]['mc_count'].values()))
        
    return json_data
  
    
def preprocess_data_Nie(json_data, ordered_relations, new_ordered_relations):  

    for model_name in json_data.keys():
        for _ in range(len(data)):
            mc_entropy = entropy(mc_dist)
            json_data[model_name][i]['entropy'] = mc_entropy

            norm_mc_dist = {k:v for (k, v) in zip(ordered_relations, mc_dist)}
            ambiguous_labels = {k for (k, v) in norm_mc_dist.items() if (v > 0.2) and (v < 0.6)}
            if len(ambiguous_labels) > 1:
                ambiguous_labels = sorted(ambiguous_labels)
                new_label = ''.join(ambiguous_labels)
            else:
                confident_labels = {k for (k, v) in norm_mc_dist.items() if (v > 0.8)}
                if len(confident_labels) == 1:
                    new_label = max(norm_mc_dist, key=norm_mc_dist.get)
                else:
                    new_label = None
                    
            json_data[model_name][i][f'entropy_{new_label}'] = mc_entropy
            
    return json_data

    
def calculate_random(json_data, data_type):
    old_relation_lst = [json_data[key]['old_label'] for key in range(len(json_data))]
    relation_lst = [json_data[key]['majority_label'] for key in range(len(json_data))]
    
    if data_type != 'Pavlick':
        c = collections.Counter(old_relation_lst)
        max_relation = max(c)
        ran_acc = round(c[max_relation] / len(old_relation_lst) * 100, 2)
        print(f"Random Accuracy (old): {ran_acc}%")

    c = collections.Counter(relation_lst)
    max_relation = max(c)
    ran_acc = round(c[max_relation] / len(relation_lst) * 100, 2)
    print(f"Random Accuracy (new): {ran_acc}%")
    
    kls, jsds, jsdistances, ent_ces, rank_css, tvds = [], [], [], [], [], []

    for key in json_data.keys():
        data = json_data[key]      

        hum = np.array(list(json_data[key]['norm_hum_dist'].values()))
        rand_val = 1/len(ordered_relations)
        rand_lst = [rand_val] * len(ordered_relations)
        llm = np.array(rand_lst)

        kls.append(kl_divergence(hum, llm, True))
        jsds.append(jensen_shannon(hum, llm)**2)
        jsdistances.append(jensen_shannon(hum, llm))
        ent_ces.append(ent_ce(hum, llm))
        rank_css.append(rank_cs(hum, llm))
        tvds.append(tvd(hum, llm))

    final_kl = sum(kls) / len(kls)
    final_jsd = sum(jsds) / len(jsds)
    final_jsdistance = sum(jsdistances) / len(jsdistances)
    final_ent_ce = sum(ent_ces) / len(ent_ces)
    final_rank_cs = sum(rank_css) / len(rank_css)
    final_tvd = sum(tvds) / len(tvds)

    print(f"KL: {round(final_kl, 4)}")
    print(f"JSD: {round(final_jsd, 4)}")
    print(f"JSDistance: {round(final_jsdistance, 4)}")
    print(f"EntCE: {round(final_ent_ce, 4)}")
    print(f"RankCS: {round(final_rank_cs, 4)}")
    print(f"TVD: {round(final_tvd, 4)}")

    
def plot_one_sample(json_data, data_type, model_type, figure_dir):
    
    random.seed(7)    
    sample = random.randint(0,len(json_data))
    print(f"Premise: {json_data[sample]['premise']}")
    print(f"Hypothesis: {json_data[sample]['hypothesis']}")
    print(f"LLM Entropy: {round(json_data[sample]['mc_entropy'], 4)}")
    print(f"Human Entropy: {round(json_data[sample]['hum_entropy'], 4)}")
    
    if data_type != 'ChaosNLI-alpha':
        order = ['e','n','c']
    else:
        order = ['1','2']

    blue_bar = [json_data[sample]['orig_dist'][ans] for ans in order]
    orange_bar = [json_data[sample]['mc_dist'][ans] for ans in order]

    N = 3
    ind = np.arange(N)

    plt.figure(figsize=(5,4))

    width = 0.3       

    plt.bar(ind, blue_bar , width, label='Human Distribution')
    plt.bar(ind + width, orange_bar, width, label=f'{model_type.capitalize()} Distribution')

    plt.xlabel('NLI Options')
    plt.xticks(ind + width / 2, ('Entail', 'Neutral', 'Contradiction'), rotation=45)
    plt.ylabel('Ratio')
    plt.ylim(0, 1)
    plt.title(f'Sample Distribution of {data_type.capitalize()}')
    plt.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(figure_dir, dpi=300)
    

def plot_several_samples(json_data, entropy_type, figure_dir2, figure_dir3):
    ROW_NUM, COLUMN_NUM = 2, 3
    
    if entropy_type == 'high':
        hum_entropys = {key : json_data[key]['hum_entropy'] for key in range(len(json_data)) if json_data[key]['hum_entropy'] > 0.8}
    elif entropy_type == 'low':
        hum_entropys = {key : json_data[key]['hum_entropy'] for key in range(len(json_data)) if json_data[key]['hum_entropy'] < 0.2}
    
    random.seed(42)
    rand_indices = random.sample(hum_entropys.keys(), ROW_NUM * COLUMN_NUM)
    fig_dict = {rand_idx : (i // COLUMN_NUM, i % COLUMN_NUM) for (i, rand_idx) in enumerate(rand_indices)}
    
    # graphs
    fig, ax = plt.subplots(ROW_NUM, COLUMN_NUM, figsize=(COLUMN_NUM * 5, ROW_NUM * 3))

    prompts = {'premise' : [], 'hypothesis' : []}
    for (t, key) in enumerate(fig_dict.keys()):
        data = json_data[key]

        prompts['premise'].append(data['premise'])
        prompts['hypothesis'].append(data['hypothesis'])

        hum_entropy = round(data['hum_entropy'], 2)
        mc_entropy = round(data['mc_entropy'], 2)
        if hum_entropy == mc_entropy:
            min_idx, max_idx = "HE", "ME"
            sign = "="
            fontweight = "bold"
        else:
            sign = "<"
            if hum_entropy < mc_entropy:
                min_idx, max_idx = "HE", "ME"
                fontweight = "bold"
            else:
                min_idx, max_idx = "ME", "HE"
                fontweight = None

        i, j = fig_dict[key][0], fig_dict[key][1]
        ax[i][j].bar(data['norm_hum_dist'].keys(), data['norm_hum_dist'].values(), width=-0.4, align='edge', color='#c22214')
        ax[i][j].bar(data['norm_mc_dist'].keys(), data['norm_mc_dist'].values(), width=0.4, align='edge', color='#0070c0')
        ax[i][j].set_ylim([0, 1.1])
        ax[i][j].tick_params(axis='x', labelsize=24)
        ax[i][j].tick_params(axis='y', labelsize=24)
        ax[i][j].set_title(f"{min_idx} ({min(hum_entropy, mc_entropy)}) {sign} {max_idx} ({max(hum_entropy, mc_entropy)})", fontsize=24, fontweight=fontweight)
        if j == 0:
            ax[i][j].set_yticks([0, 1.0])
        else:
            ax[i][j].set_yticks([])

    fig.tight_layout()
    fig.savefig(figure_dir2, dpi=300)
    
    # word samples
    for i in range(len(prompts['premise'])):
        print("P: ", prompts['premise'][i])
        print("H: ", prompts['hypothesis'][i])
        print("#"*115)
    
    # wordclouds
    fig, ax = plt.subplots(ROW_NUM, COLUMN_NUM, figsize=(COLUMN_NUM * 5, ROW_NUM * 3))

    for (i, sample_idx) in enumerate(rand_indices[:ROW_NUM*COLUMN_NUM]):
        word_data = json_data[sample_idx]['generated']
        text = ""
        for j in range(len(word_data)):
            text += " " + word_data[j]

        wordcloud = WordCloud(width=500,
                          height=300,
                          max_words=10,
                          stopwords=list(STOPWORDS),
                          background_color='white',
                          colormap='Paired',
                          random_state=42).generate(text)

        ax[i//COLUMN_NUM][i%COLUMN_NUM].imshow(wordcloud, interpolation='bilinear')
        ax[i//COLUMN_NUM][i%COLUMN_NUM].set_xticks([])
        ax[i//COLUMN_NUM][i%COLUMN_NUM].set_yticks([])

    fig.tight_layout()   
    fig.savefig(figure_dir3, dpi=300)


def plot_entropy_dist(json_data, data_type, ordered_relations, figure_dir4, figure_dir5):
    if data_type != 'Nie':
        # Human
        hum_entropys = {key : json_data[key]['hum_entropy'] for key in range(len(json_data))}
        
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        fig.patch.set_alpha(0.0)
        pad = 0.03

        _, _, labels = ax.hist(hum_entropys.values(), color='#c22214')
        ax.bar_label(labels, fontsize=10, color='#c22214')
        ax.set_xlim([0-pad, math.log(len(ordered_relations))+pad])
        # ax.set_ylim([0, height1+100])
        ax.set_xlabel('Entropy level', fontsize=12)
        ax.set_ylabel('Counts', fontsize=12)

        fig.tight_layout()
        fig.savefig(figure_dir4, dpi=300)
        
        # Machine
        mc_entropys = {key : json_data[key]['mc_entropy'] for key in range(len(json_data))}

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        fig.patch.set_alpha(0.0)
        pad = 0.03

        _, _, labels = ax.hist(mc_entropys.values(), color='#0070c0')
        ax.bar_label(labels, fontsize=10, color='#0070c0')
        ax.set_xlim([0-pad, math.log(len(ordered_relations))+pad])
        # ax.set_ylim([0, height1+100])
        ax.set_xlabel('Entropy level', fontsize=12)
        ax.set_ylabel('Counts', fontsize=12)

        fig.tight_layout()
        fig.savefig(figure_dir5, dpi=300)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        fig.patch.set_alpha(0.0)
        pad = 0.03

        _, _, labels = ax.hist(mc_entropys['roberta-large'], color='#0070c0')
        ax.bar_label(labels, fontsize=7, color='#0070c0')
        ax.set_xlim([0-pad, math.log(len(ordered_relations))+pad])
        #ax.set_ylim([0, height1+100])
        ax.set_xlabel('Entropy level', fontsize=12)
        ax.set_ylabel('Counts', fontsize=12)

        fig.tight_layout()
        fig.savefig(figure_dir4, dpi=300)

        
def plot_entropy_dist_classes(json_data, data_type, new_ordered_relations, figure_dir6, figure_dir7):
    mc_entropy_lst = [json_data[key]['mc_entropy_per_newlabel'] for key in json_data.keys()]
    relation_mc_entropys = defaultdict(list)
    for i in range(len(mc_entropy_lst)):
        temp_dict = mc_entropy_lst[i]
        for (k, v) in temp_dict.items():
            relation_mc_entropys[k].append(v)
    
    pad = 0.03
    color_lst = ["#ee4339", "#ee9336", "#eed236", "#a7ee70", "#36abee", "#476cee", "#a244ea"]
    alpha_lst = [0.7, 0.5, 0.3, 0.7, 0.5, 0.3, 0.5]

    if data_type != 'Nie':
        hum_entropy_lst = [json_data[key]['hum_entropy_per_newlabel'] for key in json_data.keys()]
        relation_hum_entropys = defaultdict(list)
        for i in range(len(hum_entropy_lst)):
            temp_dict = hum_entropy_lst[i]
            for (k, v) in temp_dict.items():
                relation_hum_entropys[k].append(v)
            
        fig = plt.figure(figsize=(4, 4))
        fig.patch.set_alpha(0.0)   

        for (i, relation) in enumerate(new_ordered_relations):
            _, _, labels = plt.hist(relation_hum_entropys[relation], color=color_lst[i], alpha=alpha_lst[i])

        plt.xlim([0-pad, math.log(len(ordered_relations))+pad])
        #plt.ylim([0, height2])
        plt.xlabel('Entropy level', fontsize=12)
        plt.ylabel('Counts', fontsize=12)
        plt.savefig(figure_dir6, dpi=300)


        fig = plt.figure(figsize=(4, 4))
        fig.patch.set_alpha(0.0)

        for (i, relation) in enumerate(new_ordered_relations):
            _, _, labels = plt.hist(relation_mc_entropys[relation], color=color_lst[i], alpha=alpha_lst[i])

        plt.xlim([0-pad, math.log(len(ordered_relations))+pad])
        #plt.ylim([0, height2])
        plt.xlabel('Entropy level', fontsize=12)
        plt.ylabel('Counts', fontsize=12)
        plt.savefig(figure_dir7, dpi=300)
        
    else:
        fig = plt.figure(figsize=(4, 4))
        fig.patch.set_alpha(0.0)

        for (i, relation) in enumerate(new_ordered_relations):
            _, _, labels = plt.hist(relation_mc_entropys[relation], color=color_lst[i], alpha=alpha_lst[i])

        plt.xlim([0-pad, math.log(len(ordered_relations))+pad])
        #plt.ylim([0, height2+100])
        plt.xlabel('Entropy level', fontsize=12)
        plt.ylabel('Counts', fontsize=12)

        plt.savefig(figure_dir6, dpi=300)

        
def count_data_usages(json_data, instance_num):
    count_usages = [json_data[key]['count_usage'] for key in range(len(json_data))]
    percent_usages = [c / instance_num for c in count_usages]

    mean = sum(percent_usages) / len(percent_usages)
    std = np.std(percent_usages)

    print("Mean of generated data usage: ", round(mean, 4))
    print("SE of generated data usage: ", round(std, 4))
    
    
if __name__ == '__main__':
    args = get_args()
    
    with open(args.data_dir, "r", encoding="utf-8") as f:
        mc_data = json.load(f)
    mc_data = {int(k):v for k,v in mc_data.items()}
    
    if args.data_type != 'alphanli':
        ordered_relations = ['e', 'c', 'n']
        new_ordered_relations = ['e', 'c', 'n', 'ce', 'en', 'cn', 'cen']
    else:
        ordered_relations = ['1', '2']
        new_ordered_relations = ['1', '2', '12']
    
    if args.data_type != 'Nie':
        preprocessed_mc_data = preprocess_data(mc_data, args.data_type, ordered_relations, new_ordered_relations)
    else:
        preprocessed_mc_data = preprocess_data_Nie(mc_data, ordered_relations, new_ordered_relations)
    
    if args.data_type != 'Nie':
        calculate_random(preprocessed_mc_data, args.data_type)
        
    plot_one_sample(preprocessed_mc_data, args.data_type, args.model_type, Path(args.figure_dir_1))
    plot_several_samples(preprocessed_mc_data, args.entropy_type, Path(args.figure_dir_2), Path(args.figure_dir_3))
    
    plot_entropy_dist(preprocessed_mc_data, args.data_type, ordered_relations, Path(args.figure_dir_4), Path(args.figure_dir_5)) 
    plot_entropy_dist_classes(preprocessed_mc_data, args.data_type, new_ordered_relations, Path(args.figure_dir_6), Path(args.figure_dir_7))
    
    if args.data_type != 'Nie':
        file_name = os.path.basename(args.data_dir)
        instance_num = int(file_name.split('_')[2])
        count_data_usages(preprocessed_mc_data, instance_num)