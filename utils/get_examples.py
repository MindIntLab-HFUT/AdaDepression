''' Retrieve similar examples from the eRisk2020 dataset. '''
import os
import json
import itertools
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import util, SentenceTransformer


with open("2020/two_hop_retrieval_47.86.json", 'r') as file:
    docs_retrieved = json.load(file)

with open("2020/label.json", 'r') as file:
    labels = json.load(file)

if not os.path.exists("database.json"):
    cases = {}
    for i in range(len(docs_retrieved)):
        for j in range(21):
            if j not in cases:
                cases[j] = []
            post_range = slice(j * 4, (j + 1) * 4)
            unique_posts = np.unique(
                list(itertools.chain.from_iterable(docs_retrieved[i][post_range]))
            )
            cases[j].append(unique_posts.tolist())

    with open("database.json", "w") as f:
        json.dump(cases, f, indent=4)

else:
    with open("database.json", "r") as f:
        cases = json.load(f)


MODEL_ST_ID = "all-MiniLM-L12-v2"
model = SentenceTransformer(MODEL_ST_ID)

def get_embedding(text, model):
    return model.encode(text)

def get_examples(item, posts):
    database = cases[str(item)]
    embeddings = []
    for d in database:
        embeddings.append(np.mean(get_embedding(d, model), axis=0))
    repr_embeddings = np.mean(get_embedding(posts, model), axis=0)
    
    similarities = cosine_similarity(np.vstack(embeddings), repr_embeddings.reshape(1, -1)).flatten()
    topk_indices = np.argsort(similarities)[-3:][::-1]    # k=3
    retrieved_docs = [database[i] for i in topk_indices]
    retrieved_labels = [labels[i] for i in topk_indices]
    def process_list(a, b, c):
        if a in b:
            index = b.index(a)
            b.pop(index)
            c.pop(index)
            return b, c
        else:
            return b[:2], c[:2]
    retrieved_docs, retrieved_labels = process_list(posts, retrieved_docs, retrieved_labels)

    format_text = []
    for doc, score in zip(retrieved_docs, retrieved_labels):
        formatted_posts = ""
        for i, post in enumerate(doc):
            formatted_posts += f"{i+1}: {post}\n\n"
        format_text.append(f"### REDDIT POSTS ###\n{formatted_posts}\nScore: {score[item]}")
    
    return format_text

