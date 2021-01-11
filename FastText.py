import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import FastText as ft
from sklearn.manifold import TSNE
from nltk.tokenize import word_tokenize, sent_tokenize
from pathlib import Path

import importlib
main = importlib.import_module("__main__")

path_to_transcript = Path('data/shorter_formatted_plain_labelled.txt')
with open(path_to_transcript, 'r') as f:
    content = f.read()
    # content = main.Preprocess_Content(content)
    content_sentences = sent_tokenize(content)
top_keywords = main.PKE_keywords(content)
print(top_keywords)


model = ft.load_fasttext_format("/Users/ShonaCW/Desktop/Imperial/YEAR 4/MSci Project/Conversation_Analysis_Project/FastText/cc.en.300.bin")
#model = ft.load_fasttext_format("cc.en.bin")

print('1:', model.similarity('teacher', 'teaches'))
print('\n2: ', model.wv.most_similar('hello'))
print('\n3:', model.wv["Artificial Intelligence"])
print('\n4:', model.wv.most_similar('Artificial Intelligence'))
print('\n5 model vector for neuralink:', model.wv["Neuralink"])
print('\n4:', model.wv.most_similar('Neuralink'))

keyword_vectors_df = pd.read_hdf('Saved_dfs/keyword_vectors_df.h5', key='df')
keywords = keyword_vectors_df['pke_keyw'].values

all_keywords = list(itertools.chain(keyword_vectors_df['noun_keyw'] , keyword_vectors_df['pke_keyw'], keyword_vectors_df['bigram_keyw'], keyword_vectors_df['trigram_keyw']))

embedding_dict = {}
for word in keywords:
    print('word', word)
    embedding_dict[word] = model.wv[word]

tsne = TSNE(n_components=2, random_state=0)
reduced_vectors = tsne.fit_transform(list(embedding_dict.values()))
words = list(embedding_dict.keys())
Xs, Ys = reduced_vectors[:, 0], reduced_vectors[:, 1]

plt.figure()
plt.scatter(Xs, Ys)
for label, x, y in zip(words, Xs, Ys):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points")

plt.legend()
plt.title('Keywords_Embedding')
plt.savefig("Saved_Images/FastText_WordEmbedding.png")
plt.show()

#fasttext.util.download_model('en', if_exists='ignore')  # English
#ft = fasttext.load_model('cc.en.300.bin')

#ft.get_word_vector('hello')
#ft.get_nearest_neighbors('hello')
