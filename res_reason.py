from concurrent.futures import ThreadPoolExecutor
import os
import torch
import re
import argparse
import json
import numpy as np
import time 
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
from load_data import DataLoader, CoTLoader
from utils import get_prompter, build_chat_input, llama_generate, baichuan_generate, mistral_generate


class ResReason():
    def __init__(self, model_name, dataset, datalength, scale_factor, penalty_weights, num_attn_candidates, res, test):
        self.model_name = model_name
        self.dataset = dataset
        self.datalength = datalength
        self.scale_factor = scale_factor
        self.penalty_weights = penalty_weights
        self.num_attn_candidates = num_attn_candidates
        self.res = res
        self.test = test

        if model_name.startswith('Mistral'):
            self.model_path = f'/mnt/publiccache/huggingface/Mistral-7B-Instruct-v0.2'
        elif '70b' in model_name:
            self.model_path = '/mnt/publiccache/huggingface/Llama-2-70b-chat-hf'
        else:
            self.model_path = f'./model/{model_name}'
        if model_name.startswith('Baichuan'):
            cot_file_path  = f'./result/{dataset}/{model_name}_cot_answer_2000.json'
            base_file_path = f'./result/{dataset}/{model_name}_direct_answer_2000.json'
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path,
                revision="v2.0",
                use_fast=False,
                trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path,
                revision="v2.0",
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.float16, trust_remote_code=True, device_map='auto')
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)  
        self.model.eval()

        self.prompter = get_prompter(model_name=model_name, dataset=dataset, task='cot_answer')

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def baichuan_get_key_position(self, question):
        question_len = len(self.tokenizer(question, return_tensors="pt").input_ids[0])
        question_msg = self.prompter.wrap_input(question, icl_cnt=5)
        question_ids = build_chat_input(self.model, self.tokenizer, question_msg)
        prompt_len = len(question_ids[0]) - question_len - 1
        stem_len = len(self.tokenizer(question.split('\n')[0],return_tensors="pt").input_ids[0])
        stem_len = prompt_len + stem_len
        key_position = {'start':prompt_len, 'end':stem_len}
        return key_position

    def mistral_get_key_position(self, question):
        stem = '\n'.join(question.split('\n')[:-1])
        stem_msg = [{"role":"user", "content": stem}]
        question_len = len(self.tokenizer.apply_chat_template(stem_msg, return_tensors="pt")[0])
        question_msg = self.prompter.wrap_input(question, icl_cnt=5)
        prompt_len = len(self.tokenizer(question_msg[:-1], return_tensors="pt")[0])
        stem_len = prompt_len + question_len
        key_position = {'start':prompt_len, 'end':stem_len}
        return key_position

    def llama_get_key_position(self, question):
        input = self.prompter.wrap_input(question, icl_cnt=5)
        question_len = len(self.prompter.user_prompt.format(question))
        prompt = input[:-question_len]
        stem = '\n'.join(input.split('\n')[:-1])
        stem_end = len(self.tokenizer(stem, return_tensors="pt").input_ids[0])
        stem_start = len(self.tokenizer(prompt, return_tensors="pt").input_ids[0]) - 1
        key_position = {'start':stem_start, 'end':stem_end}
        return key_position

    def res_inference(self, question, **kwargs):
        if not kwargs:
            kwargs = {'scale_factor':scale_factor, 'num_attn_candidates':num_attn_candidates, 'penalty_weights':penalty_weights}
        kwargs['max_new_tokens'] = 200
        kwargs['do_sample'] = False
        kwargs['res_decoding'] = True
        input = self.prompter.wrap_input(question, icl_cnt=5)
        if model_name.startswith('Llama'):
            key_position = self.llama_get_key_position(question)
            kwargs['key_position'] = key_position
            # config = GenerationConfig.from_pretrained(self.model_path, **kwargs)
            result, pred = llama_generate(self.model, kwargs, self.tokenizer, input, 'cot_answer')  
        elif model_name.startswith('Mistral'):
            key_position = self.mistral_get_key_position(question)
            kwargs['key_position'] = key_position
            # config = GenerationConfig.from_pretrained(self.model_path, **kwargs)
            result, pred = mistral_generate(self.model, kwargs, self.tokenizer, input, 'cot_answer')  
        else:
            key_position = self.baichuan_get_key_position(question)
            kwargs['key_position'] = key_position
            config = GenerationConfig.from_pretrained(self.model_path, **kwargs)
            self.model.generation_config = config
            result, pred = baichuan_generate(self.model, self.tokenizer, input, 'cot_answer')
    
        match = re.findall(r'[1-6]\)',pred)
        if match:
            pred = match[-1][:-1]
        else:
            pred = 'None'
        
        return result, pred

    def set_index(self, cot_file_path, base_file_path):
        indexloader = CoTLoader()
        _, index1 = indexloader.load_data(cot_file_path, base_file_path, mode='C2W', cnt=10)
        if self.dataset == 'siqa' and self.model_name.startswith('Baichuan'):
            _, index2 = indexloader.load_data(cot_file_path, base_file_path, mode='W2C', cnt=11)
        else:
            _, index2 = indexloader.load_data(cot_file_path, base_file_path, mode='W2C', cnt=10)
        return index1 + index2
        

    def res_call(self, dataloader, cot_file_path, base_file_path, result_path):
        with open(cot_file_path, 'r') as f:
            self.cot_data = json.load(f)[:-2]
        with open(base_file_path, 'r') as f:
            self.base_data = json.load(f)[:-2]

        index = self.set_index(cot_file_path, base_file_path)
    
        if self.test:
            max_acc = 0
            max_index = -1
            idx = 0
            weight_ls = []
            max_results = []
            for scale_factor in [40,50,60,70,80,90]:
                for penalty_weights in [0.5, 1.0, 1.5, 2.0]:
                    for num_attn_candidates in range(3, 11):
                        config = {'scale_factor':scale_factor, 'num_attn_candidates':num_attn_candidates, 'penalty_weights':penalty_weights}
                        weight_ls.append(config)
                        correct = 0
                        results = []
                        dataloader.idx = 0
                        cnt = 0
                        for data in tqdm(dataloader):
                            question = data['question']
                            label = data['label']
                            cot_msg = self.cot_data[dataloader.idx - 1]
                            base_msg = self.base_data[dataloader.idx - 1]
                            if dataloader.idx - 1 not in index and test:
                                continue
                            result, pred = self.res_inference(question=question, **config)
                            cor_flag = (pred == label)
                            if cor_flag:
                                correct += 1
                            cnt += 1  
                            cot_msg = {'question':question, 'answer':result, 'pred':pred, 'label':label, 'cor_flag':cor_flag}
                            results.append(cot_msg)
                            
                            torch.cuda.empty_cache()

                        acc = correct / cnt
                        print(f'Acc: {acc}')
                        if acc > max_acc:
                            max_acc = acc
                            max_index = idx
                            max_results = results
                            print(f'Acc: {max_acc}')
                            print(f'Config: {weight_ls[max_index]}')
                        idx += 1
                    
            print(f'Acc: {max_acc}')
            print(f'Config: {weight_ls[max_index]}')
            results = [{'config':weight_ls[max_index]}, {'acc':max_acc}]
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4)
        else:
            correct = 0
            results = []
            dataloader.idx = 0
            cnt = 0
            cost = 0
            for data in tqdm(dataloader):
                start = time.time()
                question = data['question']
                label = data['label']
                print(question, label)
                cot_msg = self.cot_data[dataloader.idx - 1]
                base_msg = self.base_data[dataloader.idx - 1]
                # if cot_msg['pred'] == base_msg['pred']:
                #     res = False
                # else:
                #     res = True
                if cot_msg['pred'] == base_msg['pred'] and res:
                    result = cot_msg['answer'] 
                    pred = cot_msg['pred']
                else:
                    result, pred = self.res_inference(question=question)
                    print("Result: ", result, pred)
                cor_flag = (pred == label)
                if cor_flag:
                    correct += 1
                cnt += 1  
                end = time.time()
                cost += end - start
                cot_msg = {'question':question, 'answer':result, 'pred':pred, 'label':label, 'cor_flag':cor_flag}
                results.append(cot_msg)
                
                torch.cuda.empty_cache()

            acc = correct / cnt
            print(f'Acc: {acc}')
            print(f'Time:{cost/cnt}')
            results.append({'acc':acc})
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Llama-2-13b-chat-hf')
    parser.add_argument('--datalength', type=int, default=5) #2000
    parser.add_argument('--dataset', type=str, default='babi')
    parser.add_argument('--scale', type=int, default=40)
    parser.add_argument('--weight', type=float, default=0.5) #2
    parser.add_argument('--num_candidates', type=int, default=8) #4
    parser.add_argument('--res', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    model_name = args.model
    dataset = args.dataset
    datalength = args.datalength
    scale_factor = args.scale
    penalty_weights = args.weight
    num_attn_candidates = args.num_candidates
    res = args.res
    test = args.test

    resReason = ResReason(model_name, dataset, datalength, scale_factor, penalty_weights, num_attn_candidates, res, test)
    resReason.setup_seed(17)

    with ThreadPoolExecutor(max_workers=20) as e:
        print("Jobs starting…")
        futures = []
        for i in range(1, 21):
            print(f"Loading data…")
            dataloader = DataLoader(dataset=dataset, data_length=datalength, task_id=str(i))
            print(f"Submitting job {i}")
            cot_file_path = f'./result/{dataset}/{model_name}_cot_answer__task-{i}_5.json'
            base_file_path = f'./result/{dataset}/{model_name}_direct_answer__task-{i}_5.json'
            result_path = f'./result/{dataset}/{model_name}_res_answer_{datalength}_s{scale_factor}_w{penalty_weights}_c{num_attn_candidates}_r{res}_t{test}_task-{i}.json'
            futures.append(e.submit(resReason.res_call, dataloader, cot_file_path, base_file_path, result_path))
        
        #Collect and handle exceptions
        for future in futures:
            try:
                future.result()  # Wait for job completion
            except Exception as exc:
                print(f"Job raised an exception: {traceback.print_exc()}")
        print("Jobs completed.")
        
        