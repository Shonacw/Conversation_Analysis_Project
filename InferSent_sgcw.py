"""
STEPS
1: Use InferSent (facebook.ai) to create utterance embeddings from a podcast transcript
2: Evaluate the Cosine Similarity between every pair of (consecutive) sentence embeddings in the whole conversation
3: Build a network and leave nodes connected with a mutual similarity > some cutoff. Clusters are then topics, nodes
    are sentences, and edges are the similarity between sentences.

Notes:
    Using GloVe (V1), not FastText (V2) vectors so far.

"""
import nltk
import torch
from InferSent.models import InferSent
from pathlib import Path
import numpy as np
import pandas as pd
import itertools
import networkx as nx
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

path = Path('/Users/ShonaCW/Desktop/Imperial/YEAR 4/MSci Project/Conversation_Analysis_Project/data/shorter_formatted_plain_labelled.txt')

V = 1                                       # 1 for GloVe, 2 for FastText
all_combinations = False                    # True to compare ALL sentences, False to compare only consecutive sentences
cutoff = 0.5                                # Cutoff value for weighted graph
## STEP 1
# Build sentences list...
# Load pre-processed transcript of interview between Elon Musk and Joe Rogan
with open(path, 'r') as f:
    content = f.read()
    sentences = nltk.sent_tokenize(content)
sentences = sentences[:100]                   # SHORTENING so that it's faster/easier to deal with for now (will remove)
print('\nPreview of "sentences":', sentences)

# InferSent...
MODEL_PATH = 'encoder/infersent%s.pkl' % V
if V == 1:
    W2V_PATH = 'GloVe/glove.840B.300d.txt'
if V == 2:
    W2V_PATH = 'fastText/crawl-300d-2M.vec'
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
infersent = InferSent(params_model)
infersent.load_state_dict(torch.load(MODEL_PATH))
infersent.set_w2v_path(W2V_PATH)
infersent.build_vocab(sentences, tokenize=True)
embeddings = infersent.encode(sentences, tokenize=True)

# Visualise the importance of each word in the following sentence to the sentence as a whole
# infersent.visualize('Archangel 12, the precursor to the SR71.', tokenize=True)

## STEP 2
def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

cos_sim_df = pd.DataFrame(columns=['Sentence1', 'Sentence1_idx', 'Sentence2', 'Sentence2_idx', 'Cosine_Similarity'])

cnt = 0
if all_combinations:                # calculate cosine similarity between ALL sentences
    combinations = list(itertools.combinations(range(len(sentences)), 2))
    for i, j in combinations:
        sent1, sent2 = sentences[i], sentences[j]
        cos_sim = cosine(infersent.encode(sent1)[0], infersent.encode(sent2)[0])
        cos_sim_df.loc[cnt] = [sent1, i, sent2, j, cos_sim]
        cnt += 1
        print('pair', cnt, '/', len(combinations), '          ', cos_sim, '           ', sent1, '======', sent2)

if not all_combinations:            # calculate cosine similarity between consecutive sentences
    for i in range(len(sentences)-1):
        j = i+1
        sent1, sent2 = sentences[i], sentences[j]
        cos_sim = cosine(infersent.encode(sent1)[0], infersent.encode(sent2)[0])
        cos_sim_df.loc[cnt] = [sent1, i, sent2, j, cos_sim]
        cnt += 1
        print('pair', cnt, '/', len(sentences), '             ', cos_sim, '           ', sent1, '======', sent2)

print('Preview of cos_sim_df: ', cos_sim_df.head())

# Store dataframe
cos_sim_df.to_hdf('InferSent_Stuff/Glove_cos_sim_df.h5', key='df', mode='w')

## STEP 3

G = nx.DiGraph()     # Instantiate graph
cos_sim_df = pd.read_hdf('InferSent_Stuff/Glove_cos_sim_df.h5', key='df')  # Load dataframe with sentence embedding info

# Build graph
G.add_nodes_from(range(len(sentences)))
for row in cos_sim_df.itertuples(index=True):
    print('row.Cosine_Similarity: ', row.Cosine_Similarity)
    if row.Cosine_Similarity >= cutoff:
        G.add_edge(row.Sentence1_idx, row.Sentence2_idx, weight=row.Cosine_Similarity)


# Plot network to see clusters
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]
nx.draw(G, width=weights, with_labels=True, font_weight='bold')
plt.show()
plt.savefig("InferSent_Stuff/path.png")
