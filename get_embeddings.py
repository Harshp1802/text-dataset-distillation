from transformers import AutoTokenizer, BertModel
import torch
import numpy as np
import pandas as pd
import os
from tqdm import trange
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# Load pre-trained model tokenizer (vocabulary)
MODEL = "bert-base-multilingual-cased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
vocab = tokenizer.get_vocab()
#sort dictionary by values
text = [k for k,_ in sorted(vocab.items(), key=lambda item: item[1])]
model = BertModel.from_pretrained(MODEL).to(device)
model.eval()

#tokenize the entire text in batches and find the output of bert model for each batch
def get_embeddings(text):
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=3).to(device)
    with torch.no_grad():
        output = model(**tokens)
    embeddings = output[0][:,1,:].cpu().numpy()
    return embeddings

all_embeddings = []
batch_size = 1000
for i in trange(0, len(text), batch_size):
    embeddings = get_embeddings(text[i:i+batch_size])
    try:
        all_embeddings = np.concatenate((all_embeddings, embeddings), axis=0)
    except:
        all_embeddings = embeddings

np.save("embeddings.npy", all_embeddings)

