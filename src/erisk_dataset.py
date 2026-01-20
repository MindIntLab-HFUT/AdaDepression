#### https://github.com/yanweiyue/masrouter/blob/main/Datasets/mbpp_dataset.py
import os
from typing import Union, Literal
import pandas as pd
import json
from tqdm import tqdm
import numpy as np
import itertools
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bdi_dir = os.path.join(parent_dir, 'BDI-II')
sys.path.append(bdi_dir)
from item_queries import items_queries
from bdi_choices import sentences_bdi
from item_names import items_names


class eRiskDataset:
    def __init__(self, split: Union[Literal['train'], Literal['val'], Literal['test']],):
        # Run TwohopRetrieval.py first to get the retrieval results and labels
        # [data, label]
        self._splits = {
            'train': [
                'data/2020/two_hop_retrieval_47.86.json',
                'data/2020/label.json'
            ], 
            'test': [
                'data/2021/two_hop_retrieval_47.86.json',
                'data/2021/label.json'
            ], 
            'val': [
                'data/2019/two_hop_retrieval_47.86.json',
                'data/2019/label.json'
            ]
        }
        
        '''
        self.data[0] = {
            "item_idx": item_idx,
            "item_query": item_query,    
            "posts": posts,
            "label": label,
            "formatted_posts": formatted_posts,
            "formatted_items": formatted_items,

        }
        '''
        self.data = []   
        with open(self._splits[split][0], 'r') as file:
            docs_retrieved = json.load(file)

        with open(self._splits[split][1], 'r') as file:
            labels = json.load(file)

        assert len(docs_retrieved) == len(labels)

        for user_idx in tqdm(range(len(docs_retrieved))):
            user_label = labels[user_idx]    
            # process each BDI-II item (21 total)
            for item_idx in range(21):   
                label = user_label[item_idx]   
                
                post_range = slice(item_idx * 4, (item_idx + 1) * 4)
                unique_posts = np.unique(
                    list(itertools.chain.from_iterable(docs_retrieved[user_idx][post_range]))
                ) 
                formatted_posts = ""
                for i, post in enumerate(unique_posts):
                    formatted_posts += f"{i+1}: {post}\n\n"
                formatted_items = ''.join([
                    f"{idx}: {item}\n" 
                    for idx, item in enumerate(sentences_bdi[post_range])
                ])    
                posts = unique_posts.tolist()

                self.data.append({
                    "item_idx": item_idx,
                    "item_query": items_queries[item_idx],    
                    "posts": posts,
                    "label": label,
                    "formatted_posts": formatted_posts,
                    "formatted_items": formatted_items    # used in few-shot prompts
                })
        
        if split == "train":
            assert len(self.data) == 70 * 21
        elif split == "test":
            assert len(self.data) == 80 * 21
        else:
            assert len(self.data) == 20 * 21


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class eRiskDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        if self.shuffle:
            self._shuffle_indices()
        self.index = 0

    def _shuffle_indices(self):
        import random
        random.shuffle(self.indices)

    def __iter__(self):
        batch = []
        for i in range(len(self.indices)):
            batch.append(self.dataset[self.indices[i]])
            if len(batch) == self.batch_size or i == len(self.indices) - 1:
                yield batch
                batch = []

    def __next__(self):
        if self.index >= len(self.dataset):
            raise StopIteration

        batch_indices = self.indices[self.index:self.index + self.batch_size]
        batch = [self.dataset[i] for i in batch_indices]
        self.index += self.batch_size
        return batch
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


