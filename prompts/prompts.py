prompts = {
    "direct": '''You are a professional psychologist.

Below are the user's REDDIT POSTS. Based on the content of these posts and the scoring criteria, predict the user's score (0–3) on the given BDI-II item, and provide a clear explanation.

### REDDIT POSTS ###    
{REDDIT_POSTS}

### BDI-II ITEM ###
{ITEM}

### SCORING CRITERIA ###
{choices}

The predicted score must be an integer ranging from 0 to 3. Your response must strictly follow the output format without any other words or content:

Score: [score]
Explanation: [explanation]

''',

    "few-shot": '''You are a professional psychologist.

Below are the user's REDDIT POSTS. Based on the content of these posts and the scoring criteria, predict the user's score (0–3) on the given BDI-II item.

### BDI-II ITEM ###
{ITEM}

### SCORING CRITERIA ###
{choices}

You can refer to the following examples.

### EXAMPLES ###
{examples}

### TARGET ###
### REDDIT POSTS ###    
{REDDIT_POSTS}

The predicted score must be an integer ranging from 0 to 3. Your response must strictly follow the output format without any other words or content:

Score: [score]

''',

    "CoT": '''You are a professional psychologist.

Based on the user's REDDIT POSTS provided below, determine their score (0–3) on the specified BDI-II item according to the scoring criteria.

### REDDIT POSTS ###    
{REDDIT_POSTS}

### BDI-II ITEM ###
{ITEM}

### SCORING CRITERIA ###
{choices}

Follow these steps:

Step 1: Identify posts that contain first-person statements related to the symptom described in the BDI-II item. Output them as a list of strings.

Step 2: Based on the content of these relevant posts and the scoring criteria, determine the user's score (0–3) for the given BDI-II item.

Step 3: Generate an explanation that corresponds to this score. Your explanation must be based solely on the identified posts.

The predicted score must be an integer ranging from 0 to 3. Your response must strictly follow the output format without any other words or content:
** Step 1 Output **
Identified Posts: ["post 1", "post 2", ... "post n"]

** Step 2 Output **
Score: [score]

** Step 3 Output **
Explanation: [explanation]

''',

    "at_once":'''You are a professional psychologist.
    
Below are user's REDDIT POSTS. 

### REDDIT POSTS ###
{REDDIT_POSTS}

Based on the content of the REDDIT POSTS, please complete the BDI-II questionnaire for the user.
The BDI-II questionnaire consists of 21 questions that assess the severity of depressive symptoms.
For each question, please assess the severity of each symptom using the BDI-II scale (0 = Not at all, 1 = Mild, 2 = Moderate, 3 = Severe). 

Your response must strictly follow the output format without any other words or content:
1. Sadness: [score]
2. Pessimism: [score]
3. Past Failure: [score]
4. Loss of Pleasure: [score]
5. Guilty Feelings: [score]
6. Punishment Feelings: [score]
7. Self-Dislike: [score]
8. Self-Criticalness: [score]
9. Suicidal Thoughts or Wishes: [score]
10. Crying: [score]
11. Agitation: [score]
12. Loss of Interest: [score]
13. Indecisiveness: [score]
14. Worthlessness: [score]
15. Loss of Energy: [score]
16. Changes in Sleeping Pattern: [score]
17. Irritability: [score]
18. Changes in Appetite: [score]
19. Concentration Difficulty: [score]
20. Tiredness or Fatigue: [score]
21. Loss of Interest in Sex: [score]
'''
}




def get_prompt(prompt_name, **kwargs):
    """
    获取指定名称的 prompt，并格式化传入的参数。
    :param prompt_name: 需要的 prompt 名称
    :param kwargs: 格式化 prompt 的额外参数
    :return: 格式化后的 prompt
    """
    prompt_template = prompts.get(prompt_name)

    if not prompt_template:
        raise ValueError(f"Prompt with name '{prompt_name}' not found!")

    return prompt_template
