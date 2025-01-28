from concurrent.futures import ThreadPoolExecutor
import os
import re
import torch
import argparse
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSeq2SeqLM
from transformers.generation.utils import GenerationConfig
from utils import get_prompter, build_chat_input
from load_data import DataLoader, CoTLoader
from metrics import draw_plot, draw_heat, draw_line_plot, draw_attr_bar
from sklearn.preprocessing import normalize


class LlmCotProbe():
    def __init__(self, model, dataset, mode, score, avg, cnt, reg, loss, diff, res, swap):
        self.model_name = model
        self.dataset = dataset
        mode= mode
        self.avg = avg
        self.cnt = cnt
        self.reg = reg
        self.score = score
        self.loss_type = loss
        self.diff = diff
        self.res = res
        self.swap = swap

        self.model_path = f'./model/{self.model_name}'
       
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.float16, trust_remote_code=True, device_map='auto')
        self.device_map = self.model.hf_device_map
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        special_tokens_dict = {"additional_special_tokens" : ["user", "assistant"]}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        print(self.tokenizer.special_tokens_map, self.tokenizer.encode("assistant"), self.tokenizer.convert_ids_to_tokens(128000))
        #user_token_id = self.tokenizer.convert_tokens_to_ids("user")
        #assistant_token_id = self.tokenizer.convert_tokens_to_ids("assistant")  
        self.config = GenerationConfig.from_pretrained(self.model_path, max_new_tokens=50, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.cot_prompter = get_prompter(model_name=self.model_name, dataset=dataset, task='cot_answer')
        self.base_prompter = get_prompter(model_name=self.model_name, dataset=dataset, task='direct_answer')

        cot_file_path  = f'./result/{dataset}/{model_name}_cot_answer_2000.json'
        base_file_path = f'./result/{dataset}/{model_name}_direct_answer_2000.json'

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


class Probe(LlmCotProbe):
    def cal_attn_attr(self, question, label, cot, layers, swap=None):
        if self.swap:
            input = f'Rational: {cot}' + f'\nQuestion: {question}'
            question_len = len(self.cot_prompter.user_prompt.format(input))
            prompt = self.self.cot_prompter.wrap_input(input, icl_cnt=5)[:-question_len]
            wrap_question = self.cot_prompter.wrap_input(input, icl_cnt=5)
            input_text = wrap_question + f'\nAnswer: ({label})'
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
            prompt_len = len(self.tokenizer(prompt, return_tensors="pt").input_ids[0]) - 1
            split_ls = wrap_question.split('\nQuestion:')
            cot = '\nQuestion:'.join(split_ls[:-1])
            cot_len = len(self.tokenizer(cot, return_tensors="pt").input_ids[0])
            stem = '\nQuestion:' + question.split('\n')[0]
            stem_len = len(self.tokenizer(cot + stem, return_tensors="pt").input_ids[0])
            question_len = len(self.tokenizer(wrap_question, return_tensors="pt").input_ids[0])
        else:
            if self.model_name[:5] == 'Llama':
                question_len = len(self.tokenizer(question, return_tensors="pt").input_ids[0])
                question_msg = self.cot_prompter.wrap_input(question, icl_cnt=5)
                print("QQQQQQQQQ: ", question_msg)
                question_ids = build_chat_input(self.model, self.tokenizer, question_msg)
                prompt_len = len(question_ids[0]) - question_len - 1
                question_len = len(question_ids[0])
                if self.loss_type == 'cot':
                    assistant_msg = [{'role':'assistant', 'content':f'{cot}'}]
                else:
                    assistant_msg = [{'role':'assistant', 'content':f'{cot} So the answer is: ({label})'}]
                input_msg = question_msg + assistant_msg 

                inputs = self.tokenizer.apply_chat_template(input, return_tensors="pt")
                input_ids = inputs.to(self.model.device)                
                prompt_len = len(self.tokenizer(prompt, return_tensors="pt").input_ids[0]) - 1
                question_len = len(self.tokenizer(wrap_question, return_tensors="pt").input_ids[0])
                stem = '\n'.join(wrap_question.split('\n')[:-1])
                stem_len = len(self.tokenizer(stem, return_tensors="pt").input_ids[0])
                cot_len = len(self.tokenizer(wrap_question + cot, return_tensors="pt").input_ids[0])
            else:
                question_len = len(self.tokenizer(question, return_tensors="pt").input_ids[0])
                question_msg = self.cot_prompter.wrap_input(question, icl_cnt=5)
                question_ids = build_chat_input(self.model, self.tokenizer, question_msg)
                prompt_len = len(question_ids[0]) - question_len - 1
                question_len = len(question_ids[0])
                
                if self.loss_type == 'cot':
                    assistant_msg = [{'role':'assistant', 'content':f'{cot}'}]
                else:
                    assistant_msg = [{'role':'assistant', 'content':f'{cot} So the answer is: ({label})'}]
                input_msg = question_msg + assistant_msg
                input_ids = build_chat_input(self.model, self.tokenizer, input_msg)
                
                stem_len = len(self.tokenizer(question.split('\n')[0],return_tensors="pt").input_ids[0])
                stem_len = prompt_len + stem_len
                cot_msg = question_msg + [{'role':'assistant', 'content':f'{cot}'}]
                cot_len = len(build_chat_input(self.model, self.tokenizer, cot_msg)[0])

        
        labels = torch.full_like(input_ids, -100)
        if self.loss_type == 'cot':
            labels[:, question_len:] = input_ids[:, question_len:]
        else:
            labels[:, -2] = input_ids[:, -2]
        if self.model_name[:5] == 'Llama':
            margin = 6
        else:
            margin = 1

        
        stem_scores = []
        option_scores = []
        cot_scores = []
        
        loss = None 
        outputs = None 
        
        
        outputs = self.model(
            input_ids=input_ids.to(self.model.device),
            labels=labels.to(self.model.device),
            return_dict=True,
            output_attentions=True,
            output_hidden_states=False,
        )
        del input_ids, labels
        loss = outputs['loss']
        
        
        for layer in layers: 
            attn_values = outputs['attentions'][layer]
            attn_grad = torch.autograd.grad(loss, attn_values, create_graph=True, allow_unused=True)[0].detach().cpu()
            attn_values = torch.squeeze(attn_values).detach().cpu()
            attn_grad = torch.squeeze(attn_grad)
            attn_scores = torch.zeros_like(attn_values[0,:,:])
            for i in range(40):
                attn_scores += self.cal_attr_score(attn_values[i,:,:], attn_grad[i,:,:])
            attn_scores = attn_scores[prompt_len:,prompt_len:].cpu().numpy()
            
            if self.loss_type == 'cot':
                stem_attr = attn_scores[question_len-prompt_len:cot_len-prompt_len, :stem_len-prompt_len].sum() 
                option_attr = attn_scores[question_len-prompt_len:cot_len-prompt_len, stem_len-prompt_len:question_len-prompt_len-margin].sum() 
                cot_attr = attn_scores[question_len-prompt_len:cot_len-prompt_len, question_len-prompt_len:cot_len-prompt_len]
                for i in range(cot_len-question_len):
                    cot_attr[i,i] = 0
                cot_attr = cot_attr.sum() 
            else:
                if self.swap:
                    stem_attr = attn_scores[-3, cot_len-prompt_len:stem_len-prompt_len].sum() 
                    option_attr = attn_scores[-3, stem_len-prompt_len:question_len-prompt_len-margin].sum() 
                    cot_attr = attn_scores[-3, :cot_len-prompt_len].sum()
                else:
                    stem_attr = attn_scores[-3, :stem_len-prompt_len].sum() 
                    option_attr = attn_scores[-3, stem_len-prompt_len:question_len-prompt_len-margin].sum() 
                    cot_attr = attn_scores[-3, question_len-prompt_len:cot_len-prompt_len].sum()
            
            stem_scores.append(stem_attr)
            option_scores.append(option_attr)
            cot_scores.append(cot_attr)
            
            del attn_values, attn_grad, attn_scores
            self.model.zero_grad() 
            torch.cuda.empty_cache()
            
        del loss, outputs
        torch.cuda.empty_cache()
        return [stem_scores, option_scores, cot_scores]
    
    
    def cal_attr_score(self, value, grad, steps=20):
        grad_int = torch.zeros_like(value*grad)
        for i in range(steps):
            k = (i+1) / steps
            grad_int += k * grad
        scores = 1 / steps * value * grad_int
        return torch.abs(scores) 
       
  
    def check_tok(self, dist_toks, tok):
        return tok in dist_toks

    
    def cal_attn(self, question, label, cot, layers, result_path, dist_toks):  
        if model_name.startswith('Llama'):
            question_len = len(self.cot_prompter.user_prompt.format(question))
            prompt = self.cot_prompter.wrap_input(question, icl_cnt=5)[:-question_len]
            wrap_question = self.cot_prompter.wrap_input(question, icl_cnt=5)
            input_text = wrap_question + cot + f' So the answer is: ({label})'
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.model.device)
            prompt_len = len(self.tokenizer(prompt, return_tensors="pt").input_ids[0]) - 1
            stem = '\n'.join(wrap_question.split('\n')[:-1])
            stem_len = len(self.tokenizer(stem, return_tensors="pt").input_ids[0])
            question_len = len(self.tokenizer(wrap_question, return_tensors="pt").input_ids[0])
            cot_len = len(self.tokenizer(wrap_question + cot, return_tensors="pt").input_ids[0])
        elif model_name.startswith('Meta'): # Meta
            question_len = len(self.tokenizer(question, return_tensors="pt").input_ids[0])
            question_msg = self.cot_prompter.wrap_input(question, icl_cnt=5)
            question_ids = build_chat_input(self.model, self.tokenizer, question_msg)
            prompt_len = len(question_ids[0]) - question_len - 1
            question_len = len(question_ids[0])
            assistant_msg = [{'role':'assistant', 'content':f'{cot} So the answer is: ({label})'}]
            input_msg = question_msg + assistant_msg
            input_ids = build_chat_input(self.model, self.tokenizer, input_msg)  
            stem_len = len(self.tokenizer(question.split('\n')[0],return_tensors="pt").input_ids[0])
            stem_len = prompt_len + stem_len
            cot_msg = question_msg + [{'role':'assistant', 'content':f'{cot}'}]
            cot_len = len(build_chat_input(self.model, self.tokenizer, input_msg)[0]) # Get label answer too

        self.model.eval()
        # Set Generation config with pad_token_id
        self.model.generation_config = self.config

        outputs = self.model(
            input_ids=input_ids.to(self.model.device),
            return_dict=True,
            output_attentions=True,
            output_hidden_states=False,
        )

        print("CoT length: ", cot_len-prompt_len) #121
        print("Question length: ", question_len-prompt_len) #84
        
        input_ids = input_ids[0, prompt_len:].detach().cpu().numpy()
        scores = []

        attn_values = torch.stack(outputs['attentions'], dim=0).squeeze(1) # Tensor of attention weights
        attn_values = attn_values.mean(dim=0) # Take mean across layers
        #assert attn_values.shape[:2] == (40, 40)
        attn_scores = attn_values[:, prompt_len:, prompt_len:].detach().cpu().numpy() #w/o system prompt
        
        if self.loss_type == 'cot':
            attn_scores = attn_scores[:, question_len-prompt_len:cot_len-prompt_len, :question_len-prompt_len].mean(axis=0) # Mean over 40 heads
            attn_scores = attn_scores/attn_scores.sum(axis=-1, keepdims=True) # Normalize attention over tokens

            # Sum attention scores for each token over all Cot tokens to get top k cand. 
            top_k_attn_indices = np.argpartition(attn_scores.sum(axis=0), -20, axis=0)[-20:].squeeze() # Take top attended word from options given CoT (1, 10)
            np.sort(top_k_attn_indices) # Sort in ascending order
            
            top_k_attn_scores = attn_scores[:, top_k_attn_indices] # Take top attended word from options given CoT (41, 10)
            np.sort(top_k_attn_scores) # Sort ascending

            print("Attn scores shape: ", attn_scores.shape)
            not_att = []
            for i, cot_tok in enumerate(attn_scores):
                for ind, attn in enumerate(attn_scores[i]): # Attention for first Cot token - top_k_…; attn, ind; 
                    tok = self.tokenizer.batch_decode(input_ids)[ind] # convert ids to natural text.
                    tok = tok.strip()
                    if re.match("([^A-Za-z\d]+)|INST", tok) or not self.check_tok(dist_toks, tok): # Remove unneccessary tokens
                        not_att.append(ind)
            
            top_k_attn_indices = list(filter(lambda x: x not in not_att, list(range(attn_scores.shape[1])))) #top_k_attn_indices

            top_k_attn_scores = attn_scores[:, top_k_attn_indices]

        else:
            attn_scores = attn_scores[:, cot_len-prompt_len:, :stem_len-prompt_len].sum() 

        y_tokens = self.tokenizer.batch_decode(input_ids[question_len-prompt_len:cot_len-prompt_len])
        x_tokens = self.tokenizer.batch_decode(input_ids[top_k_attn_indices]) # Take options

        scores.append(attn_scores)

        del attn_values, attn_scores, input_ids
        torch.cuda.empty_cache()
        del outputs
        torch.cuda.empty_cache()
        return top_k_attn_scores, x_tokens, y_tokens
 
    
    def cal_mlp_attr(self, question, label, cot, layers, loss='cot'):
        question_len = len(self.cot_prompter.user_prompt.format(question))
        wrap_question = self.cot_prompter.wrap_input(question, icl_cnt=5)
        if self.loss_type == 'cot':
            input_text = wrap_question + cot 
        else:
            input_text = wrap_question + cot + f' So the answer is: ({label})'
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
        
        question_len = len(self.tokenizer(wrap_question, return_tensors="pt").input_ids[0])
        cot_len = len(self.tokenizer(wrap_question + cot, return_tensors="pt").input_ids[0])
        
        labels = torch.full_like(input_ids, -100)
        if self.loss_type =='cot':
            labels[:, question_len:cot_len] = input_ids[:, question_len:cot_len]
        else:
            labels[:, -3] = input_ids[:, -3]

        self.model.train()
        up_scores = []
        down_scores = []
        outputs = None 
        loss = None 
        idx = -1
        
        outputs = self.model(
            input_ids=input_ids.to(self.model.device),
            labels=labels.to(self.model.device),
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        loss = outputs['loss']
        
        for layer in layers:
            mlp_up_values = self.model.model.layers[layer].mlp.up_proj.weight 
            mlp_down_values = self.model.model.layers[layer].mlp.down_proj.weight 
            mlp_up_grad = torch.autograd.grad(loss, mlp_up_values, create_graph=True, allow_unused=True)[0]
            mlp_down_grad = torch.autograd.grad(loss, mlp_down_values, create_graph=True, allow_unused=True)[0]
            mlp_up_score = self.cal_attr_score(mlp_up_values, mlp_up_grad).sum().detach().cpu().numpy()
            mlp_down_score = self.cal_attr_score(mlp_down_values, mlp_down_grad).sum().detach().cpu().numpy()
            up_scores.append(mlp_up_score)
            down_scores.append(mlp_down_score)
            
            del mlp_up_values, mlp_up_grad, mlp_down_values, mlp_down_grad
            torch.cuda.empty_cache()
            
        del loss, outputs
        torch.cuda.empty_cache()
        
        return [up_scores, down_scores]
    
    def probe_data(self, data, index, dataloader, full_cot_path, result_path, dist_toks):
        if self.reg:
            with open(full_cot_path, 'r') as f:
                cots = json.load(f)
        if self.res:
            with open(res_cot_path, 'r') as f:
                cots = json.load(f)
        pred_scores_ls = []   
            
        for msg in tqdm(data):
            label = msg['label']
            pred = msg['pred']
            idx = index[data.index(msg)]
            question = msg['question'].split("\n", 1)[0]
            steps = msg['steps']
            pred_cot = '.'.join(steps) + '.'
            
            if self.score == 'attn':
                if self.reg or self.res:
                    reg_answer = cots[idx]['answer']
                    if self.reg:
                        reg_answer = reg_answer[eval(label)-1]
                    reg_steps = dataloader.split_cot(reg_answer)
                    reg_cot = '.'.join(reg_steps) + '.'
                    if self.reg:
                        reg_cot = reg_cot.replace(' Reason: ',"")
                layers = range(40)
                scores, x_tokens, y_tokens = self.cal_attn(question, pred, pred_cot, layers, result_path, dist_toks)
        return scores, x_tokens, y_tokens
        #     ##### ToDO 
        #         if self.loss_type == 'label':
        #             new_scores = self.cal_attn(question, label, pred_cot, layers, result_path)
        #             scores = scores + new_scores
        
        #         if self.reg:
        #             reg_scores = self.cal_attn(question, label, reg_cot, layers)
        #             scores = np.array(reg_scores) / len(reg_steps) - np.array(scores) / len(steps)
                
        #         if self.avg:
        #             # Take self.avg attention over all layers
        #             print(scores[0])
        #             pred_scores_ls.append(scores) 
        #         else:   
        #             fig_path = result_path + f'_idx-{idx}.pdf'
        #             layers = [i+1 for i in layers]
        #             index = list(range(len(scores))) #.shape[1]
        #             os.makedirs("/".join(result_path.split("/")[:-1]), exist_ok=True)
        #             draw_heat(index, scores, fig_path) #tolist()
            
        #     elif self.score == 'attn_attr':
        #         if self.reg or self.res:
        #             reg_answer = cots[idx]['answer']
        #             if self.reg:
        #                 reg_answer = reg_answer[eval(label)-1]
        #             reg_steps = dataloader.split_cot(reg_answer)
        #             reg_cot = '.'.join(reg_steps) + '.'
        #             if self.reg:
        #                 reg_cot = reg_cot.replace(' Reason: ',"")

        #         layers = range(40)
                
        #         scores = self.cal_attn_attr(question, pred, pred_cot, layers)
        #         # if self.loss_type == 'label':
        #         #     if self.swap:
        #         #         new_scores = self.cal_attn_attr(question, label, pred_cot, layers, self.swap=True)
        #         #     else: 
        #         #         new_scores = self.cal_attn_attr(question, label, pred_cot, layers)
        #         #     scores = scores + new_scores
                    
        
        #         if self.reg or self.res or self.loss_type == 'label':
        #             if self.reg or self.res:
        #                 reg_scores = self.cal_attn_attr(question, label, reg_cot, layers)
        #             else:
        #                 if self.swap: 
        #                     reg_scores = self.cal_attn_attr(question, label, pred_cot, layers, swap=True)
        #                 else:
        #                     reg_scores = self.cal_attn_attr(question, label, pred_cot, layers)
        #                 reg_steps = steps
                    
        #             if self.diff:
        #                 scores = np.array(reg_scores[0]) / len(reg_steps) - np.array(scores[0]) / len(steps)
        #             else:
        #                 scores = np.concatenate([np.array(scores[0]) / len(steps), np.array(reg_scores[0])/ len(reg_steps)], axis=-1)
                        
        #         if self.avg:
        #             pred_scores_ls.append(scores) 
        #         else:   
        #             fig_path = result_path + f'_idx-{idx}.pdf'
        #             layers = [i+1 for i in layers]
        #             if self.reg:
        #                 if not self.diff:
        #                     draw_plot(layers, list(scores), ['Drift', 'Correct'], fig_path)
        #             else:
        #                 draw_plot(layers, list(scores), ['stem', 'option', 'cot'], fig_path)
            
        #     elif self.score == 'mlp_attr':
        #         if self.reg:
        #             reg_answers = cots[idx]['answer']
        #             reg_answer = reg_answers[eval(label)-1]
        #             reg_steps = dataloader.split_cot(reg_answer)
        #             reg_cot = '.'.join(reg_steps) + '.'
        #             reg_cot = reg_cot.replace(' Reason: ',"")
        #         layers = range(40)

                
        #         scores = self.cal_mlp_attr(question, pred, pred_cot, layers)
                
        #         if self.reg or self.res or self.loss_type == 'label':
        #             if self.reg:
        #                 reg_scores = self.cal_mlp_attr(question, label, reg_cot, layers)
        #             else:
        #                 if self.swap: 
        #                     reg_scores = self.cal_mlp_attr(question, label, pred_cot, layers, swap=True)
        #                 else:
        #                     reg_scores = self.cal_mlp_attr(question, label, pred_cot, layers)
        #                 reg_steps = steps
                    
        #             if self.diff:
        #                 scores = np.array(reg_scores[self.cnt]) - np.array(scores[self.cnt])
        #             else:
        #                 scores = np.concatenate([scores[self.cnt], reg_scores[self.cnt]], axis=-1)
                
        #         if self.avg:
        #             pred_scores_ls.append(scores)
        #         else:   
        #             fig_path = result_path + f'_idx-{idx}.pdf'
        #             layers = [i+1 for i in layers]
        #             draw_plot(layers, list(scores), ['up_proj', 'down_proj'], fig_path)
            
            
        # if self.score in ['attn_attr', 'mlp_attr'] and self.avg:
        #     if self.diff:
        #         return np.array(pred_scores_ls).mean(axis=0)
        #     else:
        #         return np.array(pred_scores_ls)
        # ### ToDO
        # elif self.score in ['attn'] and self.avg:
        #     if self.loss_type == 'label':
        #         return np.array(pred_scores_ls)
        #     else:
        #         return np.array(pred_scores_ls).mean(axis=0)


    def get_index(self, mode, dataset):
        index = None
        if self.model_name[:5] == 'Llama':
            if mode== 'C2W':
                if dataset == "babi":
                    index = list(range(5))
                elif dataset == 'csqa': #csqa
                    # index = [41,49,158,161,174,244,276,283,286,297,386,394,402,413,424,431,441,443,457,523,539,652,700,709,754,869,881,898,939,946]
                    if self.loss_type == 'label':
                        index = [36,331,379,395,521,525,527,599,654,826,893,913,998]
                    else:
                        index = [41,49,158,161,174,244,276,283,286,297,386,394,402,413,424,431,441,443,457,523,539,652,700,709,754,869,881,898,939,946]
                elif dataset == 'wino':
                    if self.loss_type == 'label':
                        index = [40,47,73,175,180,185,197,232,255,266,274,306,316,327,333,409,423,427,433,444,454,481,493]
                    else:
                        index = [7,15,50,53,97,108,119,121,132,201,207,209,235,253,284,285,307,320,338,342,347,387,390,426,453,467,475,478,482,490,498]
        else:
            if mode== 'C2W':
                if dataset == 'csqa':
                    if self.loss_type == 'label':
                        index = [86,221,263,279,280,342,352,395,399,408,471,545,599,761,857,877,913]
                    else:
                        index =[3,8,56,158,167,175,189,225,257,283,298,318,323,439,457,492,499,540,568,578,580,582,596,623,679,700,792,838,839,860,865,899,915,954,988]
                elif dataset == 'wino':
                    if self.loss_type == 'label':
                        index = [28,53,90,93,97,102,145,148,158,183,185,201,261,316,327,348,366,393,429,437,453,465,506,584,642,658,661,678,696,710,732,734,755,756,771,805,843,882]
                    else:
                        index = [56,69,132,142,146,147,157,163,167,203,218,221,307,320,321,342,378,383,440,512,524,591,594,610,620,624,627,645,685,686,745,751,753,760,767,832,876,930,955,966,986]
        if mode== 'RAND':
            index = random.sample(range(1000), self.cnt)
            sorted(index)
        return index 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Meta-Llama-3-8B-Instruct')
    parser.add_argument('--dataset', type=str, default='babi') #csqa #wino
    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument('--score', type=str, default=None)
    parser.add_argument('--avg', action='store_true')
    parser.add_argument('--cnt', type=int, default=5) #1000
    parser.add_argument('--reg', action='store_true')
    parser.add_argument('--loss', type=str, default='cot')
    parser.add_argument('--diff', action='store_true')
    parser.add_argument('--res', action='store_true')
    parser.add_argument('--swap', action='store_true')
    # parser.add_argument('--opt', action='store_true')
    args = parser.parse_args()

    model_name = args.model
    dataset = args.dataset
    mode = args.mode
    avg = args.avg
    cnt = args.cnt
    reg = args.reg
    score = args.score
    loss_type = args.loss
    diff = args.diff
    res = args.res
    swap = args.swap

    res_cot_path = f'./result/{dataset}/res.json'
    swap_file_path = f'./result/{dataset}/{model_name}_rt_result_2000_rFalse.json'

    probe = Probe(model_name, dataset, mode, score, avg, cnt, reg, loss_type, diff, res, swap)
    probe.setup_seed(17)
    dataloader = CoTLoader()
    layers = [i+1 for i in range(40)]
    
    # Hard-coded index, instead take dynamic C2W
    #index = probe.get_index(mode='C2W', dataset=dataset) 


    # with ThreadPoolExecutor(max_workers=20) as e:
#    print("Jobs starting…")
#    futures = []

    w2c = list(filter(lambda x: x not in [2, 3, 4, 6, 8, 9, 12, 13, 16, 17, 18, 19, 20], list(range(1, 21))))
    w2c_inv = list(filter(lambda x: x in [4, 10, 16], list(range(1, 21))))

    c2w = list(filter(lambda x: x in [4, 10, 16], list(range(1, 21))))
    c2w_inv = list(filter(lambda x: x in [1, 5, 10, 11, 14], list(range(1, 21))))
    settings = {"w2c": w2c, "w2c_inv": w2c_inv, "c2w": c2w, "c2w_inv": c2w_inv}

    dist_categories = ["subj", "obj", "relation", "location", "nh-subj", "attributes"]
    dist_toks = []
    for cat in dist_categories:
        with open (f"{cat}.txt", "r") as f:
            for line in f.readlines():
                dist_toks.append(line.strip())
    

    for name, setting in settings.items():
        for i in setting: 
            print(f"Setting: {name}")
            _result_path = f'./result/{dataset}/fig/{score}/{model_name}_m-{mode}_c-{cnt}_r-{reg}_d-{diff}_l-{loss_type}_re-{res}_s{swap}'
            result_path = os.path.join(_result_path, f"{name}")
            cot_file_path = f'./result/{dataset}/{model_name}_cot_answer__task-{i}_5.json'
            base_file_path = f'./result/{dataset}/{model_name}_direct_answer__task-{i}_5.json'
            full_cot_path = f'./result/{dataset}/{model_name}_generate_cot__task-{i}_5.json' #dev_1000
            
            assert os.path.isfile(cot_file_path)
            assert os.path.isfile(base_file_path)
            assert os.path.isfile(full_cot_path)

            if len(name) > 3: #inverse
                print("Inverse Case", re.split("_.*", name)[0].upper())
                file_copy = cot_file_path
                cot_file_path = base_file_path                
                base_file_path = file_copy

            print(f"Loading data {i}")
            data, index = dataloader.load_data(cot_file=cot_file_path, base_file=base_file_path, mode=re.split("_.*", name)[0].upper(), cnt=cnt)
            
            print(f"Submitting job {i}")
            if res:
                res_data = []
                res_index = []
                with open(res_cot_path, 'r') as f:
                    res_cots = json.load(f)
                for i in range(len(data)):
                    if res_cots[index[i]]['cor_flag']:
                        res_data.append(data[i])
                        res_index.append(index[i])
                index = res_index
                data = res_data
            if swap:
                swap_data = []
                swap_index = []
                with open(swap_file_path, 'r') as f:
                    flags = json.load(f)
                for i in range(len(data)):
                    if flags[index[i]]['cor_flag']:
                        swap_data.append(data[i])
                        swap_index.append(index[i])
                index = swap_index
                data = swap_data
            

            results, x_tokens, y_tokens = probe.probe_data(data, index, dataloader, full_cot_path, result_path, dist_toks)

            fig_path = os.path.join(result_path, f'task-{i}.pdf')
            
            if score == 'attn':
                if loss_type == 'cot':    
                    #index = layers
                    index = x_tokens
                    # Attn
                    draw_heat(index, results, x_tokens, y_tokens, fig_path, mine=True)  #tolist()
                else:  
                    labels = ['Drift', 'Correct'] 
                    draw_line_plot(layers, results, labels, fig_path, 'Attn')
            
            if score in ['attn_attr', 'mlp_attr']:
                    if reg or res:
                        if diff:
                            # c_data, index = dataloader.load_data(cot_file=cot_file_path, base_file=base_file_path, self.mode='W2C', self.cnt=len(index), index=c_index)
                            c_data, index = dataloader.load_data(cot_file=cot_file_path, base_file=base_file_path, mode=mode, cnt=len(index))
                            c_results = probe.probe_data(c_data, index)
                            results = np.concatenate([results, c_results], axis=-1)     
                            labels = ['Drift', 'Normal'] 
                            draw_line_plot(layers, [results], labels, fig_path, 'Attr Div')
                        else:
                            if reg:
                                labels = ['Drift', 'Correct'] 
                            elif res:
                                labels = ['CoT', 'RD'] 
                            draw_line_plot(layers, results, labels, fig_path, 'Attr')
                    else:
                        if swap:
                            labels = ['CoT', 'SPS'] 
                            draw_line_plot(layers, results, labels, fig_path, 'Attr')
                        elif loss_type == 'label':   
                            labels = ['Drift', 'Correct'] 
                            draw_line_plot(layers, results, labels, fig_path, 'Attr')
                        else:
                            
                            scores = []
                            for result in results:
                                scores.append(np.concatenate(result.tolist(), axis=-1))
                            if score == 'attn_attr':
                                labels = ['context', 'option', 'cot']
                            else:
                                labels = ['up_proj', 'down_proj']
                            fig_path = result_path + f'.pdf'
                            draw_line_plot(layers, scores, labels, fig_path, 'Attr') 


            #futures.append(e.submit(probe.probe_data, dataloader, cot_file_path, base_file_path, result_path))
        
        #Collect and handle exceptions
        # for future in futures:
        #     try:
        #         future.result()  # Wait for job completion
        #     except Exception as exc:
        #         print(f"Job raised an exception: {traceback.print_exc()}")
        # print("Jobs completed.")

        



