from tqdm import tqdm
import collections
import tenacity
from tenacity import retry, wait_random_exponential
from microbench import MicroBench
import torch

def _split_answer(generated_list):
    original_num = len(generated_list)
    new_generated_list = []
    
    if original_num != 0:
        for cands in generated_list:
            cands = cands.lower()

            if "answer: " in cands: 
                cand = cands.split("answer: ")[-1]
                new_generated_list.append(cand)
            elif "assistant: " in cands: 
                cand = cands.split("assistant: ")[-1]
                new_generated_list.append(cand)
    # print(generated_list)
    # print(new_generated_list)            
    return new_generated_list


@retry(wait=wait_random_exponential(min=1, max=60), stop=tenacity.stop_after_attempt(5))
def generate_openai(text, model, max_length, num_samples):    
    import openai
    if model in ['davinci','text-davinci-002','text-davinci-003']:
        response = openai.Completion.create(
                model=model,
                prompt=text,
                max_tokens=max_length,
                n=num_samples,
                logprobs=5)
        # print(response)
        messages = [response['choices'][num]["text"] for num in range(num_samples)]
        logprobs = [response['choices'][num]["logprobs"] for num in range(num_samples)]

        return messages, logprobs

    elif model in ['gpt-3.5-turbo']:
        response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that follows the instructions."},
                    {"role": "user", "content": text},
                ],
                max_tokens=max_length,
                n=num_samples)
        messages = [response['choices'][num]["message"]["content"] for num in range(num_samples)]

        return messages

lpbench = MicroBench()    
class Generation:
    def __init__(self, input, model_type, data_type, option_postprocess,
                 num_iter, max_length, num_samples, logprob):
        self.input = input
        self.model_type = model_type
        self.data_type = data_type
        self.processing_fnc = option_postprocess
        self.num_iter = num_iter
        self.max_length = max_length
        self.num_samples = num_samples
        self.logprob = logprob
    
    @lpbench
    def generate_distribution(self, device, tokenizer, model, topk):
        resp = {}
        if self.logprob:
            generated, logits_generated = [], {}
            t = 0
            for i in range(self.num_iter):
                input_ids = tokenizer(self.input, return_tensors="pt").input_ids.to(device)
                outputs = model.generate(input_ids, max_new_tokens=self.max_length, 
                                        #  num_return_sequences=self.num_samples, do_sample=True,
                                         return_dict_in_generate=True, output_scores=True)
                                         #num_beams=5)
                gen = tokenizer.batch_decode(outputs.sequences.cpu(), skip_special_tokens=True)
                if self.model_type in ['opt-iml-max-1.3b', 'opt-iml-max-30b', 'alpaca-7b', 'stable_vicuna_13b']:
                    gen = _split_answer(gen)
                generated.append(gen)
                # print(len(outputs.scores))
                logits = outputs.scores[0].cpu()

                if self.model_type in ['opt-iml-max-30b', 'flan-ul2', 'flan-t5-xxl']:
                    logits = torch.Tensor.float(logits)

                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                top_lp, idx = torch.topk(log_probs, k=topk)

                for j in range(idx.shape[0]):
                    decoded_tokens = tokenizer.batch_decode(idx[j])
                    logits_generated[t] = {k:v for k,v in zip(decoded_tokens,top_lp[j].tolist())}
                    t += 1

            resp['generated'] = generated            
            resp['logprobs'] = logits_generated
            processed, valid_ratio = self.processing_fnc(resp['logprobs'], self.data_type, is_logprob = True)
            resp['valid_ratio'] = valid_ratio

            return processed, resp

        else:
            generated = [] 
            for _ in tqdm(range(self.num_iter)):
                input_ids = tokenizer(self.input, return_tensors="pt").input_ids.to(device)
                outputs = model.generate(input_ids, max_new_tokens=self.max_length,
                                         num_return_sequences=self.num_samples, do_sample=True)
                                         #num_beams=5)
                gen = tokenizer.batch_decode(outputs, skip_special_tokens = True)
                if self.model_type in ['opt-iml-max-1.3b', 'opt-iml-max-30b', 'alpaca-7b', 'stable_vicuna_13b']:
                    gen = _split_answer(gen)
                generated += gen

                torch.cuda.empty_cache()

            resp['generated'] = generated
            processed, valid_ratio = self.processing_fnc(resp['generated'], self.data_type, is_logprob = False)           
            resp['valid_ratio'] = valid_ratio

            return processed, resp

    @lpbench
    def generate_distribution_openai_api(self):
        resp = {'messages':[], 'logprobs':[]}
        
        if self.num_iter == 1 and self.num_samples == 1:
            messages, logprobs = generate_openai(self.input, self.model_type, self.max_length, self.num_samples)
            resp['messages'] = messages
            resp['logprobs'] = logprobs
            if self.logprob:
                processed, valid_ratio = self.processing_fnc([resp['logprobs'][0]['top_logprobs'][0]], self.data_type, is_logprob = self.logprob)
            else:
                processed, valid_ratio = self.processing_fnc(resp['messages'], self.data_type)
            resp['valid_ratio'] = valid_ratio
            
        else:
            for _ in tqdm(range(self.num_iter)):
                messages, logprobs = generate_openai(self.input, self.model_type, self.max_length, self.num_samples)
                resp['messages'].extend(messages)
                resp['logprobs'].extend(logprobs)
                
            if self.logprob:
                processed, valid_ratio = self.processing_fnc(resp['logprobs'][0]['top_logprobs'][0], self.data_type, is_logprob = self.logprob)
            else:
                processed, valid_ratio = self.processing_fnc(resp['messages'], self.data_type)
                
        resp['valid_ratio'] = valid_ratio      
        return processed, resp