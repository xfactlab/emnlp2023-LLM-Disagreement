import random
import json
import re

def generate_prompt_template(premise, hypothesis, options, template_type, idx):
    if template_type in [1,4]:
        template_variations = [
            (f"Premise: {premise}\nHypothesis: {hypothesis}\nDoes the premise entail the hypothesis?\n{', '.join(options)}\nAnswer: "),
            (f"Premise: {premise}\nHypothesis: {hypothesis}\nIs the hypothesis entailed by the premise?\n{', '.join(options)}\nAnswer: "),
            (f"Here is a premise:\n{premise}\n\nHere is a hypothesis:\n{hypothesis}\n\nHere are the options: {', '.join(options)}\nIs it possible to conclude that if the premise is true, then so is the hypothesis?\nAnswer: "),
            (f"Choose your answer: If {premise}, does it logically follow that {hypothesis}?\n\n{', '.join(options)}, \n\nAnswer: "),
            (f"Read the following and determine if the hypothesis can be inferred from the premise: \nPremise: {premise}\nHypothesis: {hypothesis} \nOptions: {', '.join(options)} \nAnswer: ")
        ]
    elif template_type in [2,5]:
        template_variations = [
            (f"Observation Start: {premise[0]} \nObservation End: {premise[1]}\nWhich is likely to cause the observation?\nHypothesis 1: {hypothesis[0]} \nHypothesis 2: {hypothesis[1]}\n{', '.join(options)}\nAnswer: "),
            (f"Hypothesis 1: {hypothesis[0]} \nHypothesis 2: {hypothesis[1]}\nObservation Start: {premise[0]} \nObservation End: {premise[1]}\nWhich hypothesis is likely to cause the observation?\n\n{', '.join(options)}\nAnswer: "),
            (f"Here is the observation: Observation Start: {premise[0]} Observation End: {premise[1]})\nDetermine which of the two hypotheses is more natural to cause Observation Start to turn into Observation End.\nHypothesis 1: \
{hypothesis[0]} \nHypothesis 2: {hypothesis[1]}\nOptions: {', '.join(options)} \nAnswer: "),
            (f"Observation Start: {premise[0]} \nObservation End: {premise[1]}\nHypothesis 1: {hypothesis[0]} \nHypothesis 2: {hypothesis[1]}\Determine which of the two hypotheses is more likely to cause \
Observation Start to Observation End.\nOptions: {', '.join(options)} \nAnswer: "),
            (f"Read the following and determine which of the two hypotheses is more likely to cause Observation Start to turn into Observation End.\nObservation Start: {premise[0]} \nObservation End: {premise[1]}\nHypothesis 1: \
{hypothesis[0]} \nHypothesis 2: {hypothesis[1]}\nOptions: {', '.join(options)} \nAnswer: "),
    ]
    elif template_type in [3,6]:
        template_variations = [
            (f"Read the following and determine whether the Context Sentence contains the answer to the Question: \nQuestion: {premise}\nContext Sentence: {hypothesis} \nOptions: {', '.join(options)} \nAnswer: ")
        ]
    else:
        raise NotImplementedError('Under Implementation for other types of datasets')
    
    return template_variations[idx]

def prompt_generator(dict, model_type, data_type, template_type, shuffle=False, prompt_variation=False):
    if data_type in ['snli', 'mnli','Pavlick','anli', 'qnli']:
        sent1_type = 'premise'
    elif data_type == 'alphanli':
        sent1_type = 'observation'

    hypothesis = dict['hypothesis']
    premise = dict[sent1_type]

    idx = random.randint(0,4) # Five distinct templates
    
    # P1: Option selection
    if template_type == 1:
        options = ['entailment', 'contradiction', 'neutral']
        
    elif template_type == 2:
        options = ['Hypothesis 1', 'Hypothesis 2']

    elif template_type == 3:
        options = ['yes', 'no']
    
    # P2: Number selection
    elif template_type == 4:
        options = ['1: entailment', '2: contradiction', '3: neutral']
    
    elif template_type == 5:
        options = ['1: Hypothesis 1', '2: Hypothesis 2']

    elif template_type == 6:
        options = ['1: yes', '2: no']
    
    else:
        raise NotImplementedError('Under Implementation for other types of datasets')
    
    if shuffle:
        random.shuffle(options)

    if template_type in [3,6]:
        prompted_query = generate_prompt_template(premise, hypothesis, options, template_type, -1)
        
    elif prompt_variation:
        prompted_query = generate_prompt_template(premise, hypothesis, options, template_type, idx)

    else:
        prompted_query = generate_prompt_template(premise, hypothesis, options, template_type, -1)

    if model_type == 'stable_vicuna_13b':
        prompted_query = '### Human: '+re.sub('Answer:', '### Assistant: ', prompted_query)
    
    return prompted_query


def prompt_fs_generator(dict, model_type, data_type,
                     template_type=1, shuffle=True,
                     few_shot_k=5, few_shot_data_dir="../data/inputs/dict_snli_5_few_shot.json"):
    
    if data_type in ['snli', 'mnli','Pavlick','anli', 'qnli']:
        sent1_type = 'premise'
    elif data_type == 'alphanli':
        sent1_type = 'observation'
    sent2_type = 'hypothesis'
    
    prem_lst, hyp_lst  = [], []
    ans_lst = []
    if few_shot_k != 0:
        with open(few_shot_data_dir, "r", encoding="utf-8") as f:
            few_shot_data = json.load(f)
        few_shot_data = {int(k):v for k,v in few_shot_data.items()}     
        
        for (idx,few_shot_dict) in few_shot_data.items():
            prem_lst.append(few_shot_dict[sent1_type])
            hyp_lst.append(few_shot_dict[sent2_type])
            ans_lst.append(few_shot_dict['majority_label'])
            
    prem_lst.append(dict[sent1_type])
    hyp_lst.append(dict[sent2_type])
    
    prompted_query = ""

    if template_type == 1:
        initial = f"Read the following and determine if the hypothesis can be inferred from the premise: \n"
    elif template_type == 2:
        initial = "Read the following and determine which of the two hypotheses is more likely to cause Observation Start to turn into Observation End."
    elif template_type == 4:
        initial = f"Read the following and determine if the hypothesis can be inferred from the premise: "
    elif template_type == 5:
        initial = "Read the following and determine which of the two hypotheses is more likely to cause Observation Start to turn into Observation End."
    else:
        raise NotImplementedError('Under Implementation for other types of datasets')

    for i,(premise, hypothesis) in enumerate(zip(prem_lst, hyp_lst)):
        # print(len(prem_lst))

        # P1: Option selection
        if template_type == 1:
            options = ['entailment', 'contradiction', 'neutral']
            option_dic = {'e':'entailment', 'c':'contradiction', 'n':'neutral'}
            if shuffle:
                random.shuffle(options)
            
            if i == len(prem_lst) - 1:
                prompt_1_2 = f"Premise: {premise}\nHypothesis: {hypothesis}\nOptions: {', '.join(options)}\nAnswer: "
            else:
                prompt_1_2 = f"Premise: {premise}\nHypothesis: {hypothesis}\nOptions: {', '.join(options)}\nAnswer: {option_dic[ans_lst[i]]}\n\n"
            prompted_query += prompt_1_2

        elif template_type == 2:
            options = ['Hypothesis 1', 'Hypothesis 2']
            option_dic = {1:'Hypothesis 1', 2:'Hypothesis 2'}
            if shuffle:
                random.shuffle(options)
            
            prompt_2_2 = f"\nObservation Start: {premise[0]} \nObservation End: {premise[1]}"
            prompt_2_3 = f"\nHypothesis 1: {hypothesis[0]} \nHypothesis 2: {hypothesis[1]}"
            
            if i == len(prem_lst) - 1:
                prompt_2_4 = f"\nOptions: {', '.join(options)} \nAnswer: "
            else:
                prompt_2_4 = f"\nOptions: {', '.join(options)} \nAnswer: {option_dic[ans_lst[i]]}\n"
            prompted_query += prompt_2_2 + prompt_2_3 + prompt_2_4

        # elif template_type == 3:
        #     options = ['yes', 'no']
        #     option_dic = {}
        #     if shuffle:
        #         random.shuffle(options)
        #     prompt_1_1 = f"Read the following and determine whether the Context Sentence contains the answer to the Question: \nQuestion: {premise}"
        #     prompt_1_2 = f"\nContext Sentence: {hypothesis} \nOptions: {', '.join(options)} \nAnswer: "            
        #     prompted_query += prompt_1_1 + prompt_1_2

        # P2: Number selection
        elif template_type == 4:
            options = ['1: entailment', '2: contradiction', '3: neutral']
            option_dic = {'e':'1', 'c':'2', 'n':'3'}
            if shuffle:
                random.shuffle(options)                
            
            if i == len(prem_lst) - 1:
                prompt_1_2 = f"\nPremise: {premise}\nHypothesis: {hypothesis} \nOptions: {', '.join(options)} \nAnswer: "
            else:
                prompt_1_2 = f"\nPremise: {premise}\nHypothesis: {hypothesis} \nOptions: {', '.join(options)} \nAnswer: {option_dic[ans_lst[i]]}\n"
            prompted_query += prompt_1_2

        elif template_type == 5:
            options = ['1: Hypothesis 1', '2: Hypothesis 2']
            if shuffle:
                random.shuffle(options)
            prompt_2_2 = f"\nObservation Start: {premise[0]} \nObservation End: {premise[1]}"
            prompt_2_3 = f"\nHypothesis 1: {hypothesis[0]} \nHypothesis 2: {hypothesis[1]}"
            if i == len(prem_lst) - 1:
                prompt_2_4 = f"\nOptions: {', '.join(options)} \nAnswer: "

            else:
                prompt_2_4 = f"\nOptions: {', '.join(options)} \nAnswer: {ans_lst[i]}\n"
            prompted_query += prompt_2_2 + prompt_2_3 + prompt_2_4 

        # elif template_type == 6:
        #     options = ['1: yes', '2: no']
        #     if shuffle:
        #         random.shuffle(options)

        #     prompt_1_1 = f"Read the following and determine whether the Context Sentence contains the answer to the Question: \nQuestion: {premise}"
        #     prompt_1_2 = f"\nContext Sentence: {hypothesis} \nOptions: {', '.join(options)} \nAnswer: "
        #     prompted_query += prompt_1_1 + prompt_1_2

        else:
            raise NotImplementedError('Under Implementation for other types of datasets')
        


        if model_type == 'stable_vicuna_13b':

            prompt = re.sub('Answer', '### Assistant', prompted_query)
            prompt = re.sub('\n\n', '\n\n### Human: \n', prompt)


    if model_type == 'stable_vicuna_13b':
        final = '### Human: ' + initial + prompt

    else: 
        final = initial + prompted_query #[:-1]
    return final