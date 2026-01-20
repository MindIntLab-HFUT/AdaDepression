import json
import os
import re
import math


# # reformat labels
# YEAR = "2021"

# year2labelpath = {
#     "2019": ["dataset/eRisk2019_T3/T3/Depression Questionnaires_anon.txt", "results/2019/label.json"],
#     "2020": ["dataset/eRisk2020_T2/T2/DATA/Depression Questionnaires_anon.txt", "results/2020/label.json"],
#     "2021": ["dataset/eRisk2021_T3/T3/eRisk2021_T3_Collection/ground-truth_eRisk2021_T3.txt", "results/2021/label.json"]
# }

# if not os.path.exists(year2labelpath[YEAR][1]):
#     with open(f'data/erisk{YEAR}/users.json', 'r') as file:
#         users = json.load(file)

#     label = []
#     with open(year2labelpath[YEAR][0], 'r') as file:
#         for user in users:
#             user_scores = []    
#             file.seek(0)    
#             for line in file:
#                 values = line.split()
#                 if user == values[0]:    
#                     for value in values[1:]:
#                         user_scores.append(str(value)[0])   
#                     break
#             label.append(user_scores)

#     with open(year2labelpath[YEAR][1], 'w') as file:
#         json.dump(label, file, indent=4)

# else:
#     with open(year2labelpath[YEAR][1], 'r') as file:
#         label = json.load(file)


predict_path = 'checkpoint/AdaDepression.json'   
with open(predict_path, 'r') as file:
    predict = json.load(file)

with open("results/2021/label.json", 'r') as file:
    label = json.load(file)


def clean_predict(predict):
    assert len(predict) > 0
    clean_predict = []   
    for user in predict:
        scores = []
        for s in user:
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

predict = clean_predict(predict)


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
