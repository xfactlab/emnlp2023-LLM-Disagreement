import torch
import os

def load_model(model_type, model_dir,multi_gpu, device):
    if model_type in ['flan-t5-large','flan-t5-xl', 'flan-t5-xxl']:
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        
        tokenizer = T5Tokenizer.from_pretrained(f"google/{model_type}")
        if multi_gpu:
            try:
                model = T5ForConditionalGeneration.from_pretrained(f"google/{model_type}", device_map = "auto")
            except:
                raise NotImplementedError('The model does not support device_map = auto')
        elif model_type == 'flan-t5-xxl':
            model = T5ForConditionalGeneration.from_pretrained(f"google/{model_type}",torch_dtype=torch.bfloat16).to(device)
        else:
            model = T5ForConditionalGeneration.from_pretrained(f"google/{model_type}").to(device)

    elif model_type in ['flan-ul2']:
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        
        tokenizer = T5Tokenizer.from_pretrained(f"google/{model_type}")
        if multi_gpu:
            try:
                model = T5ForConditionalGeneration.from_pretrained(f"google/{model_type}",
                                                                   torch_dtype=torch.bfloat16, device_map = "auto")
            except:
                raise NotImplementedError('The model does not support device_map = auto')
        
    elif model_type == 'opt-iml-max-1.3b':
        from transformers import AutoTokenizer, OPTForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_type}")

        if multi_gpu:
            try:
                model = OPTForCausalLM.from_pretrained(f"facebook/{model_type}", torch_dtype=torch.bfloat16, device_map = "auto")
            except:
                raise NotImplementedError('The model does not support device_map = auto')
        else:
            model = OPTForCausalLM.from_pretrained(f"facebook/{model_type}").to(device)
    
    elif model_type == 'opt-iml-max-30b':
        from torch.nn.parallel import DataParallel
        from transformers import AutoTokenizer, AutoModelForCausalLM, OPTForCausalLM
        
        tokenizer = AutoTokenizer.from_pretrained(f"/scratch/{model_type}", use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(f"/scratch/{model_type}", torch_dtype=torch.bfloat16, device_map='auto')
        
    elif model_type == 'stable_vicuna_13b':
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir,"stable_vicuna_13b"))
        model = AutoModelForCausalLM.from_pretrained(os.path.join(model_dir,"stable_vicuna_13b"), torch_dtype=torch.bfloat16,  device_map = "auto")


    # elif model_type =='alpaca-7b':
    #     from transformers import LlamaForCausalLM, LlamaTokenizer
    #     tokenizer = LlamaTokenizer.from_pretrained("/projects/shared/Models/stanford_alpaca_7b")
    #     model = LlamaForCausalLM.from_pretrained("/projects/shared/Models/stanford_alpaca_7b").to(device)    

    return tokenizer, model 