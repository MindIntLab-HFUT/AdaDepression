#### https://github.com/yanweiyue/masrouter/blob/main/MAR/MasRouter/mas_router.py

from typing import List, Dict, Optional
import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import gammaln
import math
from openai import OpenAI

# from model.text_encoder import SentenceEncoder
from transformers import LongformerTokenizer, LongformerModel
from loguru import logger
import ollama
import autogen
from autogen import Cache
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir / 'prompts'))
sys.path.append(str(parent_dir / "BDI-II"))
sys.path.append(str(parent_dir / "utils"))
from item_names import items_names
from prompts import get_prompt
from get_examples import get_examples


def try_make_requests(model_name, stop_num, prompt):
    attempts = 0
    while attempts < stop_num:
        try:
            answers = make_requests(model_name, prompt)
            return answers
        except:
            attempts += 1
            if attempts == stop_num:
                print("Attempted twice but still failed, please check the input content.")
            # time.sleep(1) 
    return None


def make_requests(model_name, prompt):
    client = OpenAI(
        base_url='http://localhost:11434/v1/',
        api_key='None',
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a psychological counseling expert."},
            {"role": "user", "content": prompt}
        ],
        stream=False,
        temperature=0.0
    )
    answer =response.choices[0].message.content
    return answer

    # You can also use the Autogen or other caching methods to make requests and reduce the required time.
    # config_list = autogen.config_list_from_json(
    #     env_or_file="../OAI_CONFIG_LIST",
    #     file_location=".",
    #     filter_dict={
    #         "model": [model_name],
    #     }
    # )

    # agent = autogen.ConversableAgent(
    #     name='Assistant',
    #     system_message="You are a psychological counseling expert.",
    #     llm_config={
    #         "config_list": config_list,
    #         "cache_seed": 2024,
    #         "temperature": 0.0,
    #         "max_tokens": 1024,    # 100
    #     },
    #     human_input_mode='NEVER'
    # )

    # with Cache.disk(cache_path_root="cache") as cache:
    #     response = agent.generate_reply(
    #         messages=[
    #             {
    #                 'content': prompt, 
    #                 'role': 'user'
    #             }
    #         ],
    #         cache=cache,
    #     )
    
    # return response


class AdaDepression(nn.Module):
    """
    Input: Text descriptions of queries, tasks, LLMs, collab methods, roles, and corresponding tools
    Output: Task classification, number and types of LLMs required for each query, recommended collab reasoning methods and roles
    Description: LLMs include chatgpt, gemini, llama, etc., collab reasoning methods include single-agent CoT reasoning, multi-agent debate reasoning, multi-agent collaboration reasoning based on certain topological structures, roles include various identities, and various tools can be used, such as python compilers, wiki searches, etc.
    Requirements: Build a trainable model to construct the optimal multi-agent system
    """
    def __init__(self, in_dim:int = 384, hidden_dim:int = 64, device=None, training=True):
        super().__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.text_tokenizer = LongformerTokenizer.from_pretrained("longformer-base-4096")    # download first
        self.text_encoder = LongformerModel.from_pretrained("longformer-base-4096") 
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.in_dim = in_dim
        self.query_proj = nn.Linear(768, in_dim)
        self.key_proj = nn.Linear(768, in_dim)
        self.value_proj = nn.Linear(768, in_dim)
        self.reasoning_allocation = ReaAllocation(input_dim = in_dim, hidden_dim = hidden_dim, device=self.device, training=training)
        self.reasoning_embeddings = nn.Embedding(3, self.in_dim)    
        self.reasoning_embeddings.weight.requires_grad = False    
        self.llm_selector = LLMSelector(input_dim = in_dim, hidden_dim = hidden_dim, device=self.device, training=training)
        self.llm_embeddings = nn.Embedding(3, self.in_dim)    
        self.llm_embeddings.weight.requires_grad = False   

    def forward(self, posts:List[List[str]], queries:List[str],
                formatted_posts:List[str], formatted_items:List[str], item_idxs:List[int], 
                llms=["fakezeta/neural-chat-7b-v3-1:Q5_K_M", "qwen2.5:32b", "llama3:70b"],
                reasonings=["direct", "few-shot", "CoT"],  
                prompt_file:str='prompts'):
        """
        queries:List[Dict[str, str]]: List of queries
        tasks:List[Dict[str, str]]: List of tasks
        llms:List[Dict[str, str]]: List of llms
        collabs:List[Dict[str, str]]: List of collabs
        """
        # Preprocess data
        posts_copy = posts.copy()
        posts = self._preprocess_data(posts)

        # Text embedding
        posts_inputs = self.text_tokenizer(posts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(self.device)
        posts_embeddings = self.text_encoder(**posts_inputs).last_hidden_state    # batch_size * seq_len * hidden_size
        queries_inputs = self.text_tokenizer(queries, return_tensors="pt", padding=True).to(self.device)
        queries_embedding = self.text_encoder(**queries_inputs).last_hidden_state    # batch_size * seq_len * hidden_size(768)

        # Attention
        _query = self.query_proj(queries_embedding)    # batch_size * seq_len * in_dim(384)
        _key = self.key_proj(posts_embeddings)
        _value = self.value_proj(posts_embeddings)
        enhanced_posts_embeddings = torch.bmm(F.softmax(torch.bmm(_query, _key.transpose(1, 2))/torch.sqrt(torch.tensor(self.in_dim, dtype=torch.float32)), dim=-1), _value)[:, 0, :]    # batch_size * in_dim
        reasoning_embeddings = self.reasoning_embeddings.weight
        llm_embeddings = self.llm_embeddings.weight
        
        # Reasoning Allocation
        selected_reasoning_idxs, rea_log_probs, aux_loss_1 = self.reasoning_allocation(enhanced_posts_embeddings, reasoning_embeddings)
        selected_reasoning_embeddings = self.reasoning_embeddings(selected_reasoning_idxs)
                
        # LLM Selection
        selected_llms_idxs, llm_log_probs, aux_loss_2 = self.llm_selector(enhanced_posts_embeddings, selected_reasoning_embeddings, llm_embeddings)

        item_names = [items_names[item_idx] for item_idx in item_idxs]
        selected_llms = [llms[selected_llms_idx] for selected_llms_idx in selected_llms_idxs]
        selected_reasonings = [reasonings[selected_reasoning_idx] for selected_reasoning_idx in selected_reasoning_idxs]
        final_result = []

        # costs = []
        start_time = time.time()
        for item_idx, item_name, formatted_post, formatted_item, selected_llm, selected_reasoning, _posts in zip(item_idxs, item_names, formatted_posts, formatted_items, selected_llms, selected_reasonings, posts_copy):
            logger.info(f'Item name: {item_name}')
            logger.info(f'LLM: {selected_llm}')
            logger.info(f'Reasoning: {selected_reasoning}')
            logger.info('-----------------------------------')
            if selected_reasoning == "few-shot":
                examples = get_examples(item_idx, _posts)
                prompt = get_prompt(selected_reasoning).format(ITEM=item_name, choices=formatted_item, REDDIT_POSTS=formatted_post, examples="\n\n###\n".join(examples))
            else:
                prompt = get_prompt(selected_reasoning).format(ITEM=item_name, choices=formatted_item, REDDIT_POSTS=formatted_post)
            
            response = try_make_requests(selected_llm, 2, prompt)  
            final_result.append(response)
        end_time = time.time()
        print(f"Time taken for batch: {end_time - start_time} seconds")
        return final_result, rea_log_probs, llm_log_probs, aux_loss_1, aux_loss_2, selected_reasonings, selected_llms
    
    def _preprocess_data(self, posts:List[List[str]]):
        sep_token = ' ' + self.text_tokenizer.sep_token + ' '    
        return [sep_token.join(post) for post in posts]    
        
    def encoder_roles(self):
        """
        Return:
            task_role_database: Dict[str, List[Dict[str, str]]]: A dictionary of task-role database
            task_role_emb: Dict[str, torch.Tensor]: A dictionary of task-role embeddings. The tensor is N_t_r*d.
        """
        logger.info('Loading role embeddings...')
        task_role_database = {}
        task_role_emb = {}
        path = 'MAR/Roles'
        for task in os.listdir(path):
            task_path = os.path.join(path, task)
            if os.path.isdir(task_path):
                task_role_database[task] = []
                roles_list = []
                for role in os.listdir(task_path):
                    if role.endswith('.json'):
                        role_path = os.path.join(task_path, role)
                        role_profile = json.load(open(role_path, 'r', encoding='utf-8'))
                        task_role_database[task].append(role_profile)
                        roles_list.append(json.dumps(role_profile))
                if len(roles_list):
                    task_role_emb[task] = self.text_encoder(roles_list).to(self.device)
        logger.info('Role embeddings loaded.')
        return task_role_database, task_role_emb

class ReaAllocation(nn.Module):    # MoR
    def __init__(self, input_dim=384, hidden_dim=64, num_routers=8, k=2, device=None, training=True, aux_loss_coef=0.05):
        super().__init__()
        self.num_routers = num_routers
        self.k = k
        self.gate = nn.Linear(input_dim, num_routers)
        self.routers = nn.ModuleList([
            ReasoningRouter() for _ in range(num_routers)
        ])
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training = training
        self.aux_loss_coef = aux_loss_coef

    def forward(self, x, reasoning_embeddings):
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        
        topk_vals, topk_idxs = torch.topk(logits, k=self.k, dim=-1)    
        topk_mask = F.one_hot(topk_idxs, num_classes=self.num_routers).sum(dim=1)   
       
        if self.training:
            aux_loss = self._load_balancing_loss(probs, topk_mask, self.num_routers) * self.aux_loss_coef
        else:
            aux_loss = torch.tensor(0.0, device=x.device)
       
        router_outputs = torch.zeros(x.size(0), self.k, reasoning_embeddings.size(0), device=x.device)
        
        for i in range(x.size(0)):  
            for j in range(self.k): 
                router_idx = topk_idxs[i, j]  
                router_input = x[i].unsqueeze(0)  
                router_output = self.routers[router_idx](router_input, reasoning_embeddings)  
                router_outputs[i, j, :] = router_output.squeeze(0)  
       
        gate_weights = F.softmax(topk_vals, dim=-1).unsqueeze(-1)
        rea_probs = torch.sum(router_outputs * gate_weights, dim=1)  # [batch_size, input_dim]
      
        scores_cumsum = torch.cumsum(rea_probs, dim=1)
        random_num = torch.rand([rea_probs.size(0),1], device=self.device)
        selected_index = (scores_cumsum > random_num).float().argmax(dim=1)
        log_probs = torch.log(rea_probs[torch.arange(rea_probs.size(0)), selected_index]).unsqueeze(1)    

        return selected_index, log_probs, aux_loss

    
    def _load_balancing_loss(self, probs, mask, num_routers):
        expert_probs = probs.mean(dim=0)  # [num_experts]
        expert_mask = mask.float().mean(dim=0)  # [num_experts]
        loss = num_routers * (expert_probs * expert_mask).sum()
        return loss
    

class LLMSelector(torch.nn.Module):
    def __init__(self, input_dim:int=384, hidden_dim:int=64, num_routers=8, k=2, device=None, training=True, aux_loss_coef=0.05):
        super().__init__()
        self.num_routers = num_routers
        self.k = k
        self.gate = nn.Linear(input_dim * 2, num_routers)
        self.routers = nn.ModuleList([
            LLMRouter() for _ in range(num_routers)
        ])
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training = training
        self.aux_loss_coef = aux_loss_coef
        # self.output_dim = output_dim

    def forward(self, enhanced_posts_embeddings, selected_reasoning_embeddings, llm_embeddings):
        """
        llms: N_l*input_dim tensor, N_l is the number of llms, input_dim is the dimension of each llm
        contexts: N_q*input_dim tensor, N_q is the number of queries, input_dim is the dimension of each query
        """
        x = torch.cat((enhanced_posts_embeddings, selected_reasoning_embeddings), dim=-1)

        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        topk_vals, topk_idxs = torch.topk(logits, k=self.k, dim=-1)   
        topk_mask = F.one_hot(topk_idxs, num_classes=self.num_routers).sum(dim=1)    
        if self.training:
            aux_loss = self._load_balancing_loss(probs, topk_mask, self.num_routers) * self.aux_loss_coef
        else:
            aux_loss = torch.tensor(0.0, device=x.device)
        router_outputs = torch.zeros(x.size(0), self.k, llm_embeddings.size(0), device=x.device)
        for i in range(x.size(0)): 
            for j in range(self.k):  
                router_idx = topk_idxs[i, j]  
                router_input = x[i].unsqueeze(0)  
                router_output = self.routers[router_idx](router_input, llm_embeddings)  
                router_outputs[i, j, :] = router_output.squeeze(0)  
        gate_weights = F.softmax(topk_vals, dim=-1).unsqueeze(-1)
        llm_probs = torch.sum(router_outputs * gate_weights, dim=1)  # [batch_size, input_dim]
        scores_cumsum = torch.cumsum(llm_probs, dim=1)
        random_num = torch.rand([llm_probs.size(0),1], device=self.device)
        selected_index = (scores_cumsum > random_num).float().argmax(dim=1)
        log_probs = torch.log(llm_probs[torch.arange(llm_probs.size(0)), selected_index]).unsqueeze(1)   

        return selected_index, log_probs, aux_loss

    
    def _load_balancing_loss(self, probs, mask, num_routers):
        expert_probs = probs.mean(dim=0)  # [num_experts]
        expert_mask = mask.float().mean(dim=0)  # [num_experts]
        loss = num_routers * (expert_probs * expert_mask).sum()
        return loss

class ReasoningRouter(nn.Module):
    def __init__(self, in_dim=384, hidden_dim=64, temp=1.0):
        super().__init__()
        self.temp = temp
        self.U = nn.Linear(in_dim, hidden_dim)
        self.V = nn.Linear(in_dim, hidden_dim)
        
    def forward(self, x, reasoning_embeddings):
        x = F.normalize(self.U(x), p=2, dim=1)
        reasoning_embeddings = F.normalize(self.V(reasoning_embeddings), p=2, dim=1)
        scores = torch.matmul(x, reasoning_embeddings.T) 
        scores = torch.softmax(scores/self.temp, dim=1) 
        return scores

class LLMRouter(nn.Module):
    def __init__(self, in_dim=384, hidden_dim=64, temp=1.0):
        super().__init__()
        self.temp = temp
        self.U = nn.Linear(in_dim*2, hidden_dim)
        self.V = nn.Linear(in_dim, hidden_dim)
        
    def forward(self, x, llm_embeddings):
        x = F.normalize(self.U(x), p=2, dim=1)
        llm_embeddings = F.normalize(self.V(llm_embeddings), p=2, dim=1)
        scores = torch.matmul(x, llm_embeddings.T) 
        scores = torch.softmax(scores/self.temp, dim=1) 
        return scores
    
