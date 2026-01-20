#### https://github.com/yanweiyue/masrouter/blob/main/Experiments/run_mbpp.py

import sys
import os
import io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import time
import argparse
import yaml
import json
import re
import torch
from loguru import logger
import torch.nn.functional as F
import math

from model.AdaDepression import AdaDepression  
from utils.log import configure_logging

from erisk_dataset import eRiskDataset, eRiskDataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_VISIBLE_DEVICES'] = '4'


def fix_random_seed(seed:int=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_result(result_file):
    if not result_file.exists():
        with open(result_file, 'w',encoding='utf-8') as file:
            json.dump([], file)

    with open(result_file, 'r',encoding='utf-8') as file:
        data = json.load(file)
    return data

def dataloader(data_list, batch_size, i_batch):
    return data_list[i_batch*batch_size:i_batch*batch_size + batch_size]

def load_config(config_path):
    with open(config_path, 'r',encoding='utf-8') as file:
        return yaml.safe_load(file)
    
def parse_args():
    parser = argparse.ArgumentParser(description="Train AdaDepression!")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--batch_size', type=int, default=8, help="batch size")
    parser.add_argument('--epochs', type=int, default=10, help="Default 10.")
    parser.add_argument('--prompt_file', type=str, default='', help="prompt file for tasks")
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    train_dataset = eRiskDataset('train')

    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_file = f"log/erisk_{current_time}.txt"  
    fix_random_seed(42)
    configure_logging(log_name=log_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdaDepression(device=device, training=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    logger.info("Start training...")
    model.train()

    total_params = sum(p.numel() for p in model.parameters())

    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch}", 80*'-')
        total_solved, total_executed = (0, 0)
        train_loader = eRiskDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        if epoch < args.start_epoch:
            model.load_state_dict(torch.load(f"checkpoint/AdaDepression_epoch{epoch}.pth", map_location=torch.device('cuda')))
            continue
        for i_batch, current_batch in enumerate(train_loader):
            logger.info(f"Batch {i_batch}",80*'-')
            start_ts = time.time()
            item_idxs = [item['item_idx'] for item in current_batch]
            queries = [item['item_query'] for item in current_batch]
            posts = [item['posts'] for item in current_batch]
            labels = [item['label'] for item in current_batch]
            formatted_posts = [item['formatted_posts'] for item in current_batch]
            formatted_items = [item['formatted_items'] for item in current_batch]
            # optimizer.zero_grad()
            results, rea_log_probs, llm_log_probs, aux_loss_1, aux_loss_2, selected_reasonings, selected_llms = model(posts, queries, formatted_posts, formatted_items, item_idxs, prompt_file=args.prompt_file)
            end_ts = time.time()
            print("Time taken:", end_ts - start_ts)
            # task_loss = F.cross_entropy(tasks_probs, tasks_y)
            log_probs = rea_log_probs + llm_log_probs
            utilities = []
            answers_loss = []
            is_solved_list = []
            for result, label, log_prob, selected_reasoning in zip(results, labels, log_probs, selected_reasonings):
                
                def get_result(text):
                    general_match = re.search(r'Score: [0-3]', text)
                    special_match = re.search(r'[0-3]', text)  
                    if general_match:
                        return general_match.group(0).split(" ")[1]
                    elif special_match:
                        return special_match.group(0)
                    else:
                        return "0"
                result = get_result(result)

            
                is_solved = 1 if result == label else 0
                total_solved = total_solved + is_solved
                total_executed = total_executed + 1
                utility = 1 if result == label else -math.exp(abs(int(result)-int(label))-1)   
                utilities.append(utility)
                is_solved_list.append(is_solved)
                answer_loss = -log_prob * utility
                answers_loss.append(answer_loss)
            answer_loss = torch.stack(answers_loss).sum() / len(answers_loss)
            
            loss = answer_loss + aux_loss_1 + aux_loss_2
            loss.backward()

            if (i_batch + 1) % args.accumulation_steps == 0 or i_batch == len(train_loader) - 1:
                optimizer.step()  
                optimizer.zero_grad()  

            accuracy = total_solved / total_executed

            logger.info(f"Batch time {time.time() - start_ts:.3f}")
            logger.info(f"Accuracy: {accuracy}")
            logger.info(f"utilities:{utilities}")
        torch.save(model.state_dict(), f"checkpoint/AdaDepression_epoch{epoch}.pth")

    logger.info("End training...")
    
