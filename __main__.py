"""
#Breakdown of Code...
Step 1:
Step 2: Now we create a word embedding on the new set of words (containg bi and trigrams)
Step 3: Extract Keywords (Ones we'll be interested in plotting)

#Notes mentioned in code...
NOTE A
    Example of why I detect trigrams first: ['brain', 'simulation'] was a detected bigram, and
    ['deep', 'brain', 'simulation'] was a detected trigram. If I condensed all the words 'brain' and 'simulation' into
    'brain_simulation' then once I searched for the trigrams there would be none left, as it would instead have
    'deep', 'brain_simulation'.

NOTE B-
    I'm using the CBOW version of Word2Vec due to this paper
    https://www.cs.cornell.edu/~schnabts/downloads/schnabel2015embeddings.pdf

NOTE C:
    When I was inputting the preprocessed-but-lengthy sentences in 'sentences_preprocessed' the Word2Vec model wasn't
    working well very clearly. When I tested it by looking for similar words to 'Neural_Nets' it was rubbish: gave me
    # rubbish... 'dope' 'table''zaps' 'though' 'person' 'money',

    However when I only input the words from 'sentences_preprocessed' which are NOUNS... much better! And it produces
    an interesting pattern to look into.  ->
    !!!!!!!
    NOTE ^HERE^ I MADE A MISTAKE. I accidentally input a list of the 539 nouns in the whole transcript (in sequential
    order but ALL in one list, rather than split by sentence) 543 times...     it meanst (as you'll read below) that
    the Word2Vec model seemed to learn word meaning very well, the layout was strange like a flower... need to think this through
    OK. It's just giving me back the words that occurred before and after 'Neural_Nets' in the list i input :-) :-(
    !!!!!
    -> Words similar to 'Neural_Net' now are...   [workers=8, window=10, min=1, sg=0]
    similar to Neural net [('simulate', 0.9886050224304199), ('brain', 0.9882206916809082), ('word', 0.96191692352294),
    ('nets', 0.9608338475227356), ('neurons', 0.9187374114990234), ('lot', 0.9116002321243286), ('babies', 0.853849649),
    ('humans', 0.8418133854866028), ('title', 0.787124752998352), ('way', 0.7816523313522339)] :-)


#Other stuff (for me)...
pip uninstall numpy
pip install -U numpy

# Tasks...
TODO: How does one evaluate the success of a Word Embedding? Then play around with params to optimise..
Notes from paper https://arxiv.org/pdf/1901.09785.pdf
    evaluation metrics, absolute intrinsic evaluation
    the method of extracting n-grams is a word embedding task in itself (II E.)
    Maybe use Word Similarity (cosine dist) as a evaluation metric
    use QVEC: https://github.com/ytsvetko/qvec  https://arxiv.org/pdf/1809.02094.pdf
TODO: Maybe explore different methods of dimensionality reduction for plotting wordvecs (might improve layout?)

"""
from nltk.collocations import BigramCollocationFinder
from nltk.corpus import stopwords
from nltk.metrics import BigramAssocMeasures
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import TrigramAssocMeasures
from gensim.models import Word2Vec
import nltk  # Importing nltk as "import nltk.pos_tag" wasn't working (?)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from itertools import groupby
import re
from pathlib import Path
import RAKE as rake
from collections import Counter
import pke
import operator
from functools import reduce

# Load pre-processed transcript of interview between Elon Musk and Joe Rogan...
path = Path('/Users/ShonaCW/Desktop/Imperial/YEAR 4/MSci Project/Conversation_Analysis_Project/data/shorter_formatted_plain_labelled.txt')
with open(path) as f:
    content = f.read()
content_tokenized = word_tokenize(content)

content_tokenized_31st = content_tokenized
print('content tokenized: ', content_tokenized)

## Step 1
words = [w.lower() for w in content_tokenized]
stopset = set(stopwords.words('english'))
filter_stops = lambda w: len(w) < 3 or w in stopset

# Extract bigrams...
bcf = BigramCollocationFinder.from_words(words)
bcf.apply_word_filter(filter_stops)                 # Ignore bigrams whose words contain < 3 chars / are stopwords
bcf.apply_freq_filter(3)                            # Ignore trigrams which occur fewer than 3 times in the transcript
bigram_list = list(list(set) for set in bcf.nbest(BigramAssocMeasures.likelihood_ratio, 20)) # Considering the top 20
print('Bigrams: ', bigram_list)

# Extract trigrams...
tcf = TrigramCollocationFinder.from_words(words)
tcf.apply_word_filter(filter_stops)                 # Ignore trigrams whose words contain < 3 chars / are stopwords
tcf.apply_freq_filter(3)                            # Ignore trigrams which occur fewer than 3 times in the transcript
trigram_list =  list(list(set) for set in tcf.nbest(TrigramAssocMeasures.likelihood_ratio, 20)) # Considering the top 20
print('Trigrams: ', trigram_list)

list_of_condensed_grams = []

# Replace Trigrams... (NOTE A)
for trigram in trigram_list:
    trigram_0, trigram_1, trigram_2 = trigram
    trigram_condensed = str(trigram_0.capitalize() + '_' + trigram_1.capitalize() + '_' + trigram_2.capitalize())
    list_of_condensed_grams.append(trigram_condensed)
    indices = [i for i, x in enumerate(content_tokenized) if x.lower() == trigram_0
               and content_tokenized[i+1].lower() == trigram_1
               and content_tokenized[i+2].lower() == trigram_2]
    for i in indices:
        content_tokenized[i] = trigram_condensed
        content_tokenized[i+1] = '-'                # Placeholders to maintain index numbering - are removed later on
        content_tokenized[i+2] = '-'

# Replace Bigrams...
for bigram in bigram_list:
    bigram_0, bigram_1 = bigram
    bigram_condensed = str( bigram_0.capitalize() + '_' + bigram_1.capitalize())
    list_of_condensed_grams.append(bigram_condensed)
    indices = [i for i, x in enumerate(content_tokenized) if x.lower() == bigram_0
               and content_tokenized[i+1].lower() == bigram_1]
    for i in indices:
        content_tokenized[i] = bigram_condensed
        content_tokenized[i+1] = '-'                # Placeholders to maintain index numbering - are removed later on


## Step 2
# Group individual words into sentences...


sents = [list(g) for k, g in groupby(content_tokenized_31st, lambda x:x == '.') if not k] # changed to content_tokenized_31st
print('\nSents before Preprocessing: ', sents)
sents_preprocessed = []

for sent in sents:
    # Make all words -except the detected bi/trigrams- lowercase
    sent_lower = [w if w in list_of_condensed_grams else w.lower() for w in sent]
    # Join into one string so can use reg expressions
    my_lst_str = ' '.join(map(str, sent_lower))
    # Remove numbers
    result1 = re.sub(r'\d+', '', my_lst_str)
    # Remove Punctuation
    result2 = re.sub(r'[^\w\s]', '', result1)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(result2)
    result = [i for i in tokens if i not in stop_words]
    # Stemming?
    # Lemmatization ?

    sents_preprocessed.append(result)

print('Sents after Preprocessing: ', sents_preprocessed, '\n')


# Useful forms of the transcript for keyword extraction...
sents_preprocessed_flat = reduce(operator.add, sents_preprocessed)
print('\n---> sents_preprocessed_flat: ', sents_preprocessed_flat)
sents_preprocessed_flat_onestring = ' '.join(sents_preprocessed_flat)
print('---> sents_preprocessed_flat_onestring: ', sents_preprocessed_flat_onestring, '\n')


# ## Step 3
# # RAKE...
# rake_object = rake.Rake("SmartStoplist.txt") #, 2, 3, 2 #min characters in word, max number of words in phrase, min number of times it's in text
# keywords = rake_object.run(content)
# print("\nRAKE Keywords:", keywords)
#
# # Counter (rubbish)...
# keywords_from_counter = Counter(sents_preprocessed_flat).most_common(10)
# print('Counter Keywords: ', keywords_from_counter)
#
# PKE...
extractor = pke.unsupervised.TopicRank()
extractor.load_document(input=sents_preprocessed_flat_onestring)
extractor.candidate_selection()
extractor.candidate_weighting()
keywords = extractor.get_n_best(30)
top_keywords = []
for i in range(len(keywords)):
  top_keywords.append(keywords[i][0])
print('Pke Keywords: ', top_keywords, '\n')

# Extract nouns (for plotting)...
words_to_plot = [word for (word, pos) in nltk.pos_tag(word_tokenize(sents_preprocessed_flat_onestring))
                 if pos[0] == 'N' and word not in ['yeah', 'yes', 'oh']]
words_to_plot = list(dict.fromkeys(words_to_plot))                      # Remove duplicate words
print('\nWords_to_plot: ', words_to_plot, '\n')

# Extract nouns in sentences
nouns_sentences = []
for sentence in sents_preprocessed:
    words_to_plot_2 = [word for (word, pos) in nltk.pos_tag(sentence)
                     if pos[0] in ['N', 'V', 'J'] and word not in ['yeah', 'yes', 'oh']]
    # Don't want to remove duplicate words as using their locations to infer semantics in the Word2Vec model
    words_to_plot_2 = list(dict.fromkeys(words_to_plot)) #when put words_to_plot makes pretty
    nouns_sentences.append(words_to_plot_2)

# print(sents_preprocessed)
print('nouns_sentences', nouns_sentences)


## Step 3
# # Define Word2Vec Model...
# input = nouns_sentences #for only nouns from the sentences #sents_preprocessed for all sentences
# model = Word2Vec(input, window=10, min_count=1, workers=8, sg=0) #sg=0 for CBOW, =1 for Skig-gram
# words = list(model.wv.vocab)
# X = model[model.wv.vocab]
# pca = PCA(n_components=2)
# results = pca.fit_transform(X)
# xs = results[:, 0]
# ys = results[:, 1]
#
# # Print information...
# # print('Model Info: ', model)
# # print('Words in Model: ', words)
#
#
# # Evaluation... (NOTE C)
# similar = model.wv.most_similar('Neural_Net')
# print('similar to Neural net', similar)
#
# # Plot Embedding...
# plt.figure()
# plt.title('Word2Vec Word Embedding Plots')
# plt.scatter(xs, ys)
# for i, word in enumerate(words):
#     if word in list_of_condensed_grams or word in words_to_plot:
#         plt.annotate(word, xy=(results[i, 0], results[i, 1]))
#
#     #NOW Want to plot a line between the words, so can see if the pattern is just due to their sequential nature?
# plt.show()





## Extras... (below)

# def Plot_Wordcloud(sents_preprocessed_flat, save=False):
#     wordcloud = WordCloud(
#                               background_color='white',
#                               stopwords=stop_words,
#                               max_words=100,
#                               max_font_size=50,
#                               random_state=42
#                              ).generate(str(sents_preprocessed_flat))
#     fig = plt.figure(1)
#     plt.imshow(wordcloud)
#     plt.axis('off')
#     plt.show()
#
#     if save:
#         fig.savefig("WordCloud.png", dpi=900)
#     return

# Plot_Wordcloud(sents_preprocessed_flat_onestring)
from gensim.models import KeyedVectors

sentences = sents_preprocessed

# model_1 = Word2Vec(sentences, size=300, min_count=1)

model_2 = Word2Vec(size=300, min_count=1)
model_2.build_vocab(sentences)
total_examples = model_2.corpus_count
model = KeyedVectors.load_word2vec_format("glove_model2.txt", binary=False)
model_2.build_vocab([list(model.vocab.keys())], update=True)
model_2.intersect_word2vec_format("glove_model2.txt", binary=False, lockf=1.0)
model_2.save("word2vec.model")

model_2 = Word2Vec.load("word2vec.model")
model_2.train(sentences, total_examples=total_examples, epochs=model_2.iter)

# Store just the words + their trained embeddings.
word_vectors = model_2.wv
model_2.init_sims(replace=True)
model_2.save("model_2.model")
word_vectors.save("word2vec.wordvectors")

# Load back with memory-mapping = read-only, shared across processes.
wv = KeyedVectors.load("word2vec.wordvectors", mmap='r')
model_2 = KeyedVectors.load_word2vec_format("model_2.model")
# fit a 2d PCA model to the vectors
X = model_2[model_2.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
plt.figure()
plt.title('Word2Vec Word Embedding Plots')
plt.scatter(result[:, 0], result[:, 1])
words = list(model_2.wv.vocab)
for i, word in enumerate(words):
    if word in words_to_plot:
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.show()

model = Word2Vec.load("word2vec.model")
similar_1 = model.wv.most_similar('neural')
print('similar to neural', similar_1)
similar_2 = model.wv.most_similar('tesla')
print('similar to tesla', similar_2)
similar_3 = model.wv.most_similar('child')
print('similar to child', similar_3)