import sys
import io
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from loguru import logger
import argparse
import torch
from erisk_dataset import eRiskDataset, eRiskDataLoader
import time
from model.AdaDepression import AdaDepression   
from utils.log import configure_logging
import re
import math
import json


os.environ['CUDA_VISIBLE_DEVICES'] = '4'

def parse_args():
    parser = argparse.ArgumentParser(description="Test AdaDepression!")
    parser.add_argument('--batch_size', type=int, default=21, help="batch size")   
    parser.add_argument('--prompt_file', type=str, default='prompts')
    parser.add_argument('--checkpoint', type=str, default='checkpoint/AdaDepression_epochxx.pth')
    parser.add_argument('--save_file', type=str, default='')
    args = parser.parse_args()
    return args

def fix_random_seed(seed:int=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



if __name__ == '__main__':
    args = parse_args()
    args.epoch = args.checkpoint.split('epoch')[1].split('_')[0]
    test_dataset = eRiskDataset('test')
    # valid_dataset = eRiskDataset('val')
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_file = f"log/erisk_{current_time}.txt"    
    fix_random_seed(42)
    configure_logging(log_name=log_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdaDepression(device=device, training=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cuda')))
    
    logger.info("Start testing...")
    model.eval()
    total_solved, total_executed = (0, 0)
    test_loader = eRiskDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    final_results = []    
    final_labels = []     
    user_scores = []    
    user_labels = []   
    category_results = []
    for i_batch, current_batch in enumerate(test_loader):
        logger.info(f"Batch {i_batch}",80*'-')
        # start_ts = time.time()
        item_idxs = [item['item_idx'] for item in current_batch]
        queries = [item['item_query'] for item in current_batch]
        posts = [item['posts'] for item in current_batch]
        labels = [item['label'] for item in current_batch]
        formatted_posts = [item['formatted_posts'] for item in current_batch]
        formatted_items = [item['formatted_items'] for item in current_batch]
        with torch.no_grad(): 
            results, rea_log_probs, llm_log_probs, aux_loss_1, aux_loss_2, selected_reasonings, selected_llms = model(posts, queries, formatted_posts, formatted_items, item_idxs, prompt_file=args.prompt_file)
        # end_ts = time.time()

        user_scores.extend(results)
        user_labels.extend(labels)
        if len(user_scores) == 21 and len(user_labels) == 21:  
            final_results.append(user_scores)
            user_scores = [] 
            final_labels.append(user_labels)
            user_labels = []

        utilities = []
        for result, label, selected_reasoning, selected_llm in zip(results, labels, selected_reasonings, selected_llms):
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

            
        accuracy = total_solved / total_executed
        # logger.info(f"Batch time {time.time() - start_ts:.3f}")
        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"utilities:{utilities}")
    logger.info("End testing...")
    with open(os.path.join(args.save_file, f"AdaDepression_epoch{args.epoch}.json"), 'w') as file:
        json.dump(final_results, file, indent=4)


    def clean_predict(predict):
        assert len(predict) > 0
        clean_predict = []   
        for user in predict:
            scores = []
            for s in user:
                # cot_match = re.search(r'Score: [0-3]', s)
                # others_match = re.search(r'[0-3]', s)    
                # if cot_match:    # 首先判断是否是CoT的输出格式
                #     scores.append(cot_match.group(0).split(" ")[1])
                # elif others_match:    # 再判断是否为direct/few-shot的输出格式
                #     scores.append(others_match.group(0))
                # else:    # 输出格式完全不正确
                #     scores.append('0')

                def get_result(text):
                    general_match = re.search(r'Score: [0-3]', text)
                    special_match = re.search(r'[0-3]', text)  
                    if general_match:
                        return general_match.group(0).split(" ")[1]
                    elif special_match:
                        return special_match.group(0)
                    else:
                        return "0"
                scores.append(get_result(s))
            clean_predict.append(scores)
        return clean_predict
    

    predict = clean_predict(final_results)
    label = final_labels


    def cal_AHR(label, predict):
        assert len(label) == len(predict)
        HRs = []
        for x, y in zip(predict, label):
            hit = 0
            assert len(x) == 21 and len(y) == 21, f"len(x): {len(x)}, len(y): {len(y)}"
            for i in range(len(x)):
                if x[i] == y[i]:
                    hit += 1
            HRs.append(hit/len(x))
        if len(HRs) != 0:
            return sum(HRs) / len(HRs)    
        else:
            return 0


    def cal_ACR(label, predict):
        assert len(label) == len(predict)
        CRs = []
        for x, y in zip(predict, label):
            assert len(x) == 21 and len(y) == 21
            mad = 3  
            CR_total = 0
            for i in range(len(x)):
                ad = abs(int(x[i]) - int(y[i]))    
                CR = (mad - ad) / mad
                CR_total += CR
            CRs.append(CR_total/len(x))
        if len(CRs) != 0:
            return sum(CRs) / len(CRs)    
        else:
            return 0
        

    def cal_ADODL(label, predict):
        assert len(label) == len(predict)
        DODLs = []
        for x, y in zip(predict, label):
            assert len(x) == 21 and len(y) == 21
            predict_score = 0
            label_score = 0
            for i in range(len(x)):
                predict_score += int(x[i])
                label_score += int(y[i])
            DODL = (63-abs(predict_score - label_score))/63
            DODLs.append(DODL)
        if len(DODLs) != 0:
            return sum(DODLs) / len(DODLs)    
        else:
            return 0
        

    def cal_DCHR(label, predict):
        assert len(label) == len(predict)
        hit = 0
        for x, y in zip(predict, label):
            assert len(x) == 21 and len(y) == 21
            predict_score = 0
            label_score = 0
            for i in range(len(x)):
                predict_score += int(x[i])
                label_score += int(y[i])
            
            def map(score):
                if score >= 0 and score <= 9:
                    # return "minimal"
                    return 0
                elif score <= 18:
                    # return "mild"
                    return 1
                elif score <= 29:
                    # return "moderate"
                    return 2
                else:
                    # return "severe"
                    return 3
        
            if map(predict_score) == map(label_score):
                hit += 1
        
        return hit / len(label)


    def cal_new_DCHR(label, predict):
        assert len(label) == len(predict)
        hit = 0
        for x, y in zip(predict, label):
            assert len(x) == 21 and len(y) == 21
            predict_score = 0
            label_score = 0
            for i in range(len(x)):
                predict_score += int(x[i])
                label_score += int(y[i])
            
            def map(score):
                if score >= 0 and score <= 13:
                    # return "minimal"
                    return 0
                elif score <= 19:
                    # return "mild"
                    return 1
                elif score <= 28:
                    # return "moderate"
                    return 2
                else:
                    # return "severe"
                    return 3
        
            if map(predict_score) == map(label_score):
                hit += 1
        
        return hit / len(label)


    def cal_RMSE(label, predict):
        assert len(label) == len(predict)
        error = 0
        for x, y in zip(predict, label):
            assert len(x) == 21 and len(y) == 21
            predict_score = 0
            label_score = 0
            for i in range(len(x)):
                predict_score += int(x[i])
                label_score += int(y[i])
            error += (predict_score - label_score)**2
        return math.sqrt(error/len(label))


    print("#"*10 + " AHR " + "#"*10)
    print(cal_AHR(label, predict))
    print("#"*10 + " ACR " + "#"*10)
    print(cal_ACR(label, predict))
    print("#"*10 + " ADODL " + "#"*10)
    print(cal_ADODL(label, predict))
    print("#"*10 + " DCHR " + "#"*10)
    print(cal_DCHR(label, predict))
    print("#"*10 + " NEW DCHR " + "#"*10)
    print(cal_new_DCHR(label, predict))
    print("#"*10 + " RMSE " + "#"*10)
    print(cal_RMSE(label, predict))   

