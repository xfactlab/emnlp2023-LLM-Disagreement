import os
import json
import collections
import logging
from tqdm.auto import tqdm
from datetime import datetime
import pandas as pd
import torch

from utils.args import get_args
from utils.prompts import prompt_generator, prompt_fs_generator
from utils.processing import option_postprocess, dict_postprocess
import utils.generation
from utils.generation import Generation 
from utils.models import load_model
    

if __name__ == '__main__':
    
    args = get_args()    
    
    # Prepare dataset
    with open(args.data_dir, "r", encoding="utf-8") as f:
        sampled_data = json.load(f)
    
    sampled_data = {int(k):v for k,v in sampled_data.items()}
    print(f'Total of {len(sampled_data)} data loaded')

    # Check device in utilization
    if args.device:
        device = args.device
    elif args.multi_gpu:
        device = 'cuda'
    else:
        device = torch.device(0 if torch.cuda.is_available() else "cpu")   

    # Prepare models
    if args.model in ['gpt-3.5-turbo','text-davinci-003', 'text-davinci-002']:
        import openai
        
        # Place openai key in key.txt
        with open('key.txt') as f:
            lines = f.readlines()
        openai.api_key = lines[0]

    else:
        tokenizer, model = load_model(args.model, args.model_dir, args.multi_gpu, device)
    
    # Generate Response
    for j,(idx,dict) in enumerate(tqdm(sampled_data.items())):
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

        if int(args.few_shot_k) > 0:
            input = prompt_fs_generator(dict, args.model, args.data_type,
                                    template_num, args.shuffle,
                                    args.few_shot_k, args.few_shot_data_dir)

        else:
            input = prompt_generator(dict, args.model, args.data_type,
                                    template_num, args.shuffle, args.prompt_variation)
        
        gen = Generation(input, args.model, args.data_type, option_postprocess,
                             args.num_iter, args.max_length,
                             args.num_samples, (args.gen_type == 'lpe'))
        
        if args.model in ['flan-t5-large','flan-t5-xl','flan-t5-xxl','flan-ul2','opt-iml-max-1.3b','opt-iml-max-30b','stable_vicuna_13b', 'alpaca-7b']:
            sampled, resp = gen.generate_distribution(device, tokenizer, model, topk=5)     
            
        elif args.model in ['davinci','text-davinci-002','text-davinci-003','gpt-3.5-turbo']:
            sampled, resp = gen.generate_distribution_openai_api()                

        print(f'Input: {input}\nSampled: {sampled}')
        
        if args.time:
            time_results = pd.read_json(utils.generation.lpbench.outfile.getvalue(), lines=True)
            run_time = (time_results['finish_time']-time_results['start_time']).mean()
            run_time_lst = str(run_time).split(":")
            hours = float(run_time_lst[0].split("days")[-1])               
            minutes, seconds = float(run_time_lst[1]), float(run_time_lst[-1])
            run_time = 3600*hours + 60*minutes + seconds
            sampled_data[idx]['run_time'] = run_time
            print(f"Generation run time: {run_time} sec.")
            
        if args.data_type in ['snli', 'mnli','Pavlick','alphanli']:  
            sampled_dict, label_counter, orig_dist, sampled_dist = dict_postprocess(dict, sampled, args.data_type, is_logprob=(args.gen_type == 'lpe'))
        else:
            sampled_dict, sampled_dist = dict_postprocess(dict, sampled, args.data_type, is_logprob=(args.gen_type == 'lpe'))
        
        sampled_data[idx]['valid_ratio'] = resp['valid_ratio']
        if args.gen_type == 'lpe':
            sampled_data[idx]['lpe_dist'] = sampled_dist
        else:
            sampled_data[idx]['mc_count'] = sampled_dict
            sampled_data[idx]['mc_dist'] = sampled_dist

        if args.data_type in ['snli','mnli','Pavlick','alphanli']:   
            sampled_data[idx]['label_counter'] = label_counter
            sampled_data[idx]['orig_dist'] = orig_dist        

        if args.model in ['davinci','text-davinci-002','text-davinci-003']:
            sampled_data[idx]['generated'] = resp['messages']
            sampled_data[idx]['logprobs'] = resp['logprobs']
        else:
            sampled_data[idx]['generated'] = resp['generated']
            if (args.gen_type == 'lpe'):
                sampled_data[idx]['logprobs'] = resp['logprobs']

                        
    print(f"{'#'*100}\nGeneration Completed!!!!")   
    
    # Save responses
    if args.dump:
        if not args.file_name:
            file_name = 'dict_' + args.data_type + '_dist'
        else:
            file_name = args.file_name

        if not args.out_dir:
            out_dir = os.path.join(os.curdir, 'data')
        else: 
            out_dir = args.out_dir
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        json_file = os.path.join(out_dir, f'{file_name}.json')

        with open(json_file, 'w') as f:
            json.dump(sampled_data, f, indent=2)

        print(f'JSON File dumped as {os.path.basename(json_file)}')