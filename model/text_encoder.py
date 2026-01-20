from transformers import LongformerTokenizer, LongformerModel
import torch


def get_sentence_embedding(sentence):
    tokenizer = LongformerTokenizer.from_pretrained("longformer-base-4096")
    model = LongformerModel.from_pretrained("longformer-base-4096")
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=4096)
    embeddings = model(**inputs).last_hidden_state
    return torch.tensor(embeddings)

class SentenceEncoder(torch.nn.Module):
    def __init__(self,device=None):
        super().__init__()
        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = LongformerTokenizer.from_pretrained("longformer-base-4096")
        self.model = LongformerModel.from_pretrained("longformer-base-4096", device=self.device)
        
    def forward(self, sentence):
        if len(sentence) == 0:
            return torch.tensor([]).to(self.device)
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(self.device)
        embeddings = self.model(**inputs).last_hidden_state
        return embeddings


