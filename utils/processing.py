import math
import collections

def _valid_cands_check(generated_labels, data_type, is_logprob):
    if data_type in ['snli','mnli','Pavlick','anli']:
        option_lst = ['entailment', 'contradiction', 'neutral']
        option_dict = {
            "e" : ['1','entail','yes','imply','implies'],
            "c" : ['2','cont','oppos','negat'],
            "n" : ['3','neut','unanswerable','null','it is not possible to tell']
        }        
    elif data_type == 'alphanli':
        option_lst = ['Hypothesis 1', 'Hypothesis 2']
        option_dict = {
            "1" : ['1'],
            "2" : ['2']
        }    
    elif data_type == 'qnli':
        option_lst = ['yes', 'no']
        option_dict = {
            "y" : ['1','entail','yes','imply','implies'],
            "n" : ['2','cont','oppos','negat']
        }
    
    if is_logprob:
        generated = list(generated_labels.keys())
        logits = list(generated_labels.values())
        idx_to_logits = {idx:logit for idx,logit in enumerate(logits)}
    else:
        generated = generated_labels
        
    processed = []
    original_num = len(generated)
    
    for i,cands in enumerate(generated):
        cands = cands.lower()
        cands = cands.replace(" ", "")

        if cands in option_lst:
            processed.append((i, cands[0]))

        elif cands == 'no' and data_type in ['snli','mnli','Pavlick','anli']:
            processed.append((i, 'c'))

        else:
            flag = True
            for k,v_list in option_dict.items():
                for v in v_list:
                    if v in cands:
                        processed.append((i, k))
                        flag = False
                        break
                if not flag:
                    break
                    
    processed_labels = [label for idx,label in processed]    
    if is_logprob:
        cand_indices = [idx for idx,label in processed]
        processed_logprobs = [idx_to_logits[idx] for idx in cand_indices]
    else:
        processed_logprobs = None
        
    raw_valid_ratio = len(processed) / original_num * 100        

    return processed_labels, processed_logprobs, raw_valid_ratio


def option_postprocess(generated_inputs, data_type, is_logprob=False):

    if not is_logprob:
        generated_labels = generated_inputs
        processed_labels, _, valid_ratio = _valid_cands_check(generated_labels, data_type, is_logprob=False)
        c = collections.Counter(processed_labels)
        sampled = list(c.items())
        
    else: 
        sampled = collections.defaultdict(list)  
        for i in range(len(generated_inputs)):
            label_logprob_dic = generated_inputs[i]    
            generated_labels = label_logprob_dic
            processed_labels, processed_logprobs, raw_valid_ratio = _valid_cands_check(generated_labels, data_type, is_logprob=True)
            assert len(processed_labels) == len(processed_logprobs)
            for j in range(len(processed_labels)):
                label = processed_labels[j]
                logprob = processed_logprobs[j]
                sampled[label].append(math.exp(logprob))
        sampled = {k:sum(v) for k,v in sampled.items()}
        
        if data_type == 'alphanli':
            num_classes = 2
        elif data_type in ['snli', 'mnli']:
            num_classes = 3
        valid_ratio = len(sampled.keys()) / num_classes * 100
        
    print(f'Maintained data percentage from generation noise : {valid_ratio:.2f}% \n')

    return sampled, valid_ratio


def dict_postprocess(dict, sampled, data_type, is_logprob):
    if data_type == 'alphanli':
        opt_cand = ['1', '2']
    elif data_type == 'qnli':
        opt_cand = ['y', 'n']
    else:
        opt_cand = ['e', 'c', 'n']

    if is_logprob:
        sampled_dict = sampled
    else:
        sampled_dict = {k:v for k,v in sampled}
    
    if len(sampled_dict) < len(opt_cand):
        for cand in opt_cand:
            if cand not in sampled_dict.keys():
                sampled_dict[cand] = 0
    
    try:
        sampled_dist = {k : v/sum(sampled_dict.values()) for k,v in sampled_dict.items()}
    except ZeroDivisionError:
        print("No valid answers generated.")
        sampled_dist = {}
    except:
        print("Other type of error occured.")
        sampled_dist = {}

    if data_type in ['snli','mnli','alphanli','Pavlick']:
        if len(dict['label_counter']) < len(opt_cand):
            for cand in opt_cand:
                if cand not in dict['label_counter'].keys():
                    dict['label_counter'][cand] = 0
                    
        tot_human = sum(dict['label_counter'].values())
        orig_dist = {k : v/tot_human for k,v in dict['label_counter'].items()}

        return sampled_dict, dict['label_counter'], orig_dist, sampled_dist
    
    else:
        return sampled_dict, sampled_dist