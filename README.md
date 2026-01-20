# From Social Media to Psychological Scale: An Adaptive Framework with Two-Hop Retrieval for Depression Screening

As illustrated in the figure, AdaDepression consists of the following three modules. (1) In the **Two-Hop Posts Retrieval** module, a collection of representative posts for each symptom is first identified from the training set to characterize their expressions on social media. Subsequently, these posts are employed as queries to retrieve related posts for the target user. (2) Within the **Posts-Directed Reasoning Allocation** module, we first derive a symptom-enhanced post representation by applying an attention mechanism to the retrieved posts and the corresponding symptom. This enhanced representation is then utilized to select the most suitable reasoning strategy from a reasoning pool. (3) Finally, the selection of the optimal LLM from the predefined LLM pool is performed by the **Dual-Factor LLM Selection** module, leveraging the post representation and the selected reasoning strategy, both of which are previously derived. The framework is optimized via reinforcement learning to maximize alignment with ground-truth responses, thereby enhancing overall performance.

![image](https://github.com/MindIntLab-HFUT/MultiAgentESC/blob/main/image/model4.png)

## Quick Start

#### 1. Clone this project locally
```bash
git clone https://github.com/MindIntLab-HFUT/MultiAgentESC.git
```

#### 2. Navigate to the directory
```bash
cd MultiAgentESC
```

#### 3. Set up the environment
```bash
pip install -r requirements.txt
```

#### 4. Replace the relevant path

Replace paths in parse_args() in main.py

#### 5. Run the Python file run.py
```bash
python main.py
```
