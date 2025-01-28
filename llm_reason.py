from concurrent.futures import ThreadPoolExecutor
import os
import traceback
import torch
import re
import argparse
import json
import time 
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSeq2SeqLM
from load_data import DataLoader
from utils import llama_generate, baichuan_generate, get_config, get_prompter, chat_generate, mistral_generate


class LLMReason():
    def __init__(self, model_name, dataset, task, datalength, icl, shuffle):
        self.model_name = model_name
        self.dataset = dataset
        self.task = task
        self.datalength = datalength
        self.icl = icl
        self.shuffle = shuffle

        if self.model_name.startswith('Vicuna'):
            self.model_path = f'/netcache/huggingface/vicuna-13b'
        elif self.model_name.startswith('Mistral'):
            self.model_path = f'/mnt/publiccache/huggingface/Mistral-7B-Instruct-v0.2'
        else:
            if '70b' in self.model_name:
                self.model_path = '/mnt/publiccache/huggingface/Llama-2-70b-chat-hf'
            else:    
                self.model_path = f'./model/{self.model_name}'

        if self.model_name.startswith('Baichuan'):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path,
                revision="v2.0",
                use_fast=False,
                trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path,
                revision="v2.0",
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True)
            self.model.eval()
        elif self.model_name.startswith('Meta') or self.model_name.startswith('Vicuna') or self.model_name.startswith('Mistral'):
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='auto')
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, padding_side="left")  
            self.model.eval()
    

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


    def model_generate(self, question):
        input = self.prompter.wrap_input(question, icl_cnt=5)
        if self.task == 'sc':
            config = get_config(model_name=self.model_name, strategy='beam')
        elif self.task in ['cons_answer', 'l2m']:
            config = get_config(model_name=self.model_name, strategy='sample')
        else:
            config = get_config(model_name=self.model_name, strategy='greedy')
        if self.model_name.startswith('Baichuan'):
            self.model.generation_config = config
            return baichuan_generate(self.model, self.tokenizer, input, self.task)
        elif self.model_name.startswith('Chat'):
            return chat_generate(input, self.task)    
        elif self.model_name.startswith('Mistral'):
            return mistral_generate(self.model, config, self.tokenizer, input, self.task)
        else: # Llama cot
            return llama_generate(self.model, config, self.tokenizer, input, self.task)

    def model_call(self, dataloader, result_path):
        self.prompter = None 
        correct = 0
        cnt = 0
        results = []
        cost = 0
        if self.task == 'dpr':
            path = f'./{self.task}_{self.dataset}_documents.json'
            with open(path, 'r') as f:
                documents = json.load(f)
        self.prompter = get_prompter(self.model_name, self.dataset, self.task)
        for data in tqdm(dataloader):
            start = time.time()
            question = data['question']
            label = data['label']
            if self.task == 'l2m':
                self.prompter = get_prompter(self.model_name, self.dataset, 'l2m_question')
                result, _ = self.model_generate(question)
                split_result = result.split('\n')
                questions = []
                for q in split_result[1:]:
                    if 'Question' in q:
                        questions.append(q)
                self.prompter = get_prompter(self.model_name, self.dataset, 'l2m_mid_answer')
                for q in questions:
                    question += '\n' + q
                    result, _ = self.model_generate(question)
                    question += " " + result.split('\n')[0]
                self.prompter = get_prompter(self.model_name, self.dataset, 'l2m_final_answer')
                result, pred = self.model_generate(question)
                self.prompter = get_prompter(self.model_name, self.dataset, 'l2m_question')
            elif self.task == 'sr':
                self.prompter = get_prompter(self.model_name, self.dataset, 'cot_answer')
                result, _ = self.model_generate(question)
                question += '\nRationale: ' + result
                self.prompter = get_prompter(self.model_name, self.dataset, 'sr_feedback')
                result, _ = self.model_generate(question)
                question += ' ' + result
                self.prompter = get_prompter(self.model_name, self.dataset, 'sr_answer')
                result, pred = self.model_generate(question)
            elif self.task == 'cons_answer':
                self.prompter = get_prompter(self.model_name, self.dataset, 'cons_answer')
                result, pred = self.model_generate(question)
            elif self.task == 'direct_answer':
                result, pred = self.model_generate(question)
            elif self.task == 'dpr':   
                if self.dataset in ['wino','piqa']:
                    width = 2 
                elif self.dataset == 'hella':
                    width = 3
                else:
                    width = 4
                document = ""
                for i in range(cnt, cnt+width):
                    document += documents[i]['ctxs'][0]['text'] + '. '
                question =  document + question  
                self.prompter = get_prompter(self.model_name, self.dataset, 'direct_answer')
                result, pred = self.model_generate(question)
            elif self.task == 'bm25':   
                self.prompter = get_prompter(self.model_name, self.dataset, 'cot_answer')
                result, pred = self.model_generate(question)
            else:   #cot_answer
                result, pred = self.model_generate(question)
            if self.dataset != 'gsm8k':
                match = re.findall(r'[1-6]\)',pred)
                if match:
                    pred = match[0][:-1]
                else:
                    pred = 'None'
            else:
                output = pred.split('\n')
                output = [line for line in output if len(re.findall('\d+', line)) > 0][-1]
                answer = output.replace(',', '')  # remove middle ',' from numbers like '1,234'
                answer = re.findall('\d+', answer)
                pred = label if label in answer else answer[-1]
                pred = answer.strip()
            cor_flag = (pred == label)
            cnt += 1
            end = time.time()
            cost += end - start 
            if cor_flag:
                correct += 1
            msg = {'question':question, 'answer':result, 'pred':pred, 'label':label, 'cor_flag':cor_flag}
            results.append(msg)
            torch.cuda.empty_cache()

        print(f'Acc:{correct/cnt}')
        print(f'Time:{cost/cnt}')

        results.append({'acc':correct/cnt})
        results.append({'time':cost/cnt})

        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Meta-Llama-3-8B-Instruct')
    parser.add_argument('--datalength', type=int, default=5) #2
    parser.add_argument('--dataset', type=str, default='babi') #csqa #siqa
    parser.add_argument('--task', type=str, default='cot_answer')
    parser.add_argument('--icl', type=int, default=5)
    parser.add_argument('--shuffle', action='store_true')
    args = parser.parse_args()
    
    model_name = args.model
    dataset = args.dataset
    datalength = args.datalength
    task = args.task
    icl = args.icl
    shuffle = args.shuffle
    llm_reason = LLMReason(model_name, dataset, task, datalength, icl, shuffle=False)
    llm_reason.setup_seed(17)
    #result_path = f'./result/{dataset}/{model_name}_{task}_{datalength}.json' #_task-{i}
    #dataloader = DataLoader(dataset=dataset, data_length=datalength, shuffle=shuffle)
    #llm_reason.model_call(dataloader, result_path)
    
    with ThreadPoolExecutor(max_workers=8) as e:
        print("Jobs starting…")
        futures = []
        for i in [2,6,10,12,13,17,18,19]:
            print(f"Loading data…")
            dataloader = DataLoader(dataset=dataset, data_length=datalength, shuffle=shuffle, task_id=str(i))
            print(f"Submitting job {i}")
            result_path = f'./result/{dataset}/{model_name}_{task}__task-{i}_{datalength}.json'
            futures.append(e.submit(llm_reason.model_call, dataloader, result_path))

        #Collect and handle exceptions
        for future in futures:
            try:
                future.result()  # Wait for job completion
            except Exception as exc:
                print(f"Job raised an exception: {traceback.print_exc()}")
        print("Jobs completed.")