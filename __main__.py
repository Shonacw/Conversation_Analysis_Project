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
from gensim.models import KeyedVectors
import nltk  # Importing nltk as "import nltk.pos_tag" wasn't working (?)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from itertools import groupby
import itertools
import re
from pathlib import Path
import RAKE as rake
from collections import Counter
import pke
import operator
from functools import reduce

# Load pre-processed transcript of interview between Elon Musk and Joe Rogan...
path = Path('/Users/ShonaCW/Desktop/Imperial/YEAR 4/MSci Project/Conversation_Analysis_Project/data/shorter_formatted_plain_labelled.txt')
with open(path, 'r') as f:
    content = f.read()
    content_sentences = nltk.sent_tokenize(content)
content_tokenized = word_tokenize(content)

# content_tokenized_31st = content_tokenized
# print('content tokenized: ', content_tokenized)

## Step 1
# words = [w.lower() for w in content_tokenized]
# stopset = set(stopwords.words('english'))
# filter_stops = lambda w: len(w) < 3 or w in stopset
#
# # Extract bigrams...
# bcf = BigramCollocationFinder.from_words(words)
# bcf.apply_word_filter(filter_stops)                 # Ignore bigrams whose words contain < 3 chars / are stopwords
# bcf.apply_freq_filter(3)                            # Ignore trigrams which occur fewer than 3 times in the transcript
# bigram_list = list(list(set) for set in bcf.nbest(BigramAssocMeasures.likelihood_ratio, 20)) # Considering the top 20
# print('Bigrams: ', bigram_list)
#
# # Extract trigrams...
# tcf = TrigramCollocationFinder.from_words(words)
# tcf.apply_word_filter(filter_stops)                 # Ignore trigrams whose words contain < 3 chars / are stopwords
# tcf.apply_freq_filter(3)                            # Ignore trigrams which occur fewer than 3 times in the transcript
# trigram_list =  list(list(set) for set in tcf.nbest(TrigramAssocMeasures.likelihood_ratio, 20)) # Considering the top 20
# print('Trigrams: ', trigram_list)
#
# list_of_condensed_grams = []
#
# # Replace Trigrams... (NOTE A)
# for trigram in trigram_list:
#     trigram_0, trigram_1, trigram_2 = trigram
#     trigram_condensed = str(trigram_0.capitalize() + '_' + trigram_1.capitalize() + '_' + trigram_2.capitalize())
#     list_of_condensed_grams.append(trigram_condensed)
#     indices = [i for i, x in enumerate(content_tokenized) if x.lower() == trigram_0
#                and content_tokenized[i+1].lower() == trigram_1
#                and content_tokenized[i+2].lower() == trigram_2]
#     for i in indices:
#         content_tokenized[i] = trigram_condensed
#         content_tokenized[i+1] = '-'                # Placeholders to maintain index numbering - are removed later on
#         content_tokenized[i+2] = '-'
#
# # Replace Bigrams...
# for bigram in bigram_list:
#     bigram_0, bigram_1 = bigram
#     bigram_condensed = str( bigram_0.capitalize() + '_' + bigram_1.capitalize())
#     list_of_condensed_grams.append(bigram_condensed)
#     indices = [i for i, x in enumerate(content_tokenized) if x.lower() == bigram_0
#                and content_tokenized[i+1].lower() == bigram_1]
#     for i in indices:
#         content_tokenized[i] = bigram_condensed
#         content_tokenized[i+1] = '-'                # Placeholders to maintain index numbering - are removed later on
#

## Step 2
# Group individual words into sentences...
#
#
# sents = [list(g) for k, g in groupby(content_tokenized_31st, lambda x:x == '.') if not k] # changed to content_tokenized_31st
# print('\nSents before Preprocessing: ', sents)

sents_preprocessed = []
# print('content_sentences', content_sentences)
for sent in content_sentences:
    # Remove numbers
    sent = re.sub(r'\d+', '', sent)
    # Remove Punctuation
    sent = re.sub(r'[^\w\s]', '', sent)
    # Make all words -except the detected bi/trigrams- lowercase
    #sent_lower = [w if w in list_of_condensed_grams else w.lower() for w in sent]

    # make lowercase and remove stopwords
    stop_words = set(stopwords.words('english'))
    sent_lower = [w.lower() for w in word_tokenize(sent) if w not in stop_words]
    # Join into one string so can use reg expressions
    result = ' '.join(map(str, sent_lower))

    # Stemming?
    # Lemmatization ?
    sents_preprocessed.append(result)
#
# print('Sents after Preprocessing: ', sents_preprocessed, '\n')
#
#
# # Useful forms of the transcript for keyword extraction...
# sents_preprocessed_flat = reduce(operator.add, sents_preprocessed)
# print('\n---> sents_preprocessed_flat: ', sents_preprocessed_flat)
# sents_preprocessed_flat_onestring = ' '.join(sents_preprocessed_flat)
# print('---> sents_preprocessed_flat_onestring: ', sents_preprocessed_flat_onestring, '\n')


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
extractor.load_document(input=content) #sents_preprocessed_flat_onestring
extractor.candidate_selection()
extractor.candidate_weighting()
keywords = extractor.get_n_best(30)
top_keywords = []
for i in range(len(keywords)):
  top_keywords.append(keywords[i][0])

# Now make sure bigram keywords are joined with an underscore
top_keywords_final = []
for keyword in top_keywords:
    # see if it is formed of >1 word
    try:
        words = word_tokenize(keyword)
        keyword = '_'.join(words)
        top_keywords_final.append(keyword)
    except:
        top_keywords_final.append(keyword)

print('Pke top_keywords_final: ', top_keywords_final, '\n')
#


# Extract nouns (for plotting)...
print('sents_preprocessed', sents_preprocessed)
sents_preprocessed_flat_onestring = ' '.join(sents_preprocessed)
print('sents_preprocessed_flat_onestring', sents_preprocessed_flat_onestring)
words_to_plot = [word for (word, pos) in nltk.pos_tag(word_tokenize(sents_preprocessed_flat_onestring))
                 if pos[0] == 'N' and word not in ['yeah', 'yes', 'oh', 'i', 'im', 'id', 'thats', 'shes', 'dont',
                                                   'youre', 'theyll', 'youve', 'whats', 'doesnt', 'hes', 'whos', 
                                                   'shouldnt']
                 and len(word) != 1]
nouns_to_plot = list(dict.fromkeys(words_to_plot))                      # Remove duplicate words
print('\nnouns_to_plot: ', nouns_to_plot, '\n')
#
# Extract nouns in sentences
# print('sents_preprocessed', sents_preprocessed)
# nouns_sentences = []
# for sentence in sents_preprocessed:
#     words_to_plot_2 = [word for (word, pos) in nltk.pos_tag(word_tokenize(sentence))
#                      if pos[0] in ['N', 'V', 'J'] and word not in ['yeah', 'yes', 'oh']]
#     # Don't want to remove duplicate words as using their locations to infer semantics in the Word2Vec model
#     words_to_plot_2 = list(dict.fromkeys(words_to_plot_2)) #when put words_to_plot makes pretty
#     nouns_sentences.append(words_to_plot_2)
#
# # print(sents_preprocessed)
# print('nouns_sentences', nouns_sentences)


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
# """https://datascience.stackexchange.com/questions/10695/how-to-initialize-a-new-word2vec-model-with-pre-trained-model-weights
# note using "glove_model2.txt which is the Word2Vec version of glove.840B.. etc. otherwise get an error about base 10."""
# from gensim.models import KeyedVectors
#
# sentences = sents_preprocessed
#
# model_2 = Word2Vec(size=300, min_count=1)
# model_2.build_vocab(sentences)
# total_examples = model_2.corpus_count
# #model = KeyedVectors.load_word2vec_format(r"glove_model2.txt", binary=False, unicode_errors='unicode_escape')
# model = Word2Vec.load("GloVe/glove.840B.300d.txt")
# model.init_sims(replace=True)
# model.build_vocab([list(model.vocab.keys())], update=True)
# model_2.intersect_word2vec_format(r"GloVe/glove.840B.300d.txt", binary=False, lockf=1.0)
# model_2.save("Word2Vec_Models/word2vec.model")
#
# model_2 = Word2Vec.load("Word2Vec_Models/word2vec.model")
# model_2.train(sentences, total_examples=total_examples, epochs=model_2.iter)
#
# # Store just the words + their trained embeddings.
# word_vectors = model_2.wv
# model_2.init_sims(replace=True)
# model_2.save("Word2Vec_Models/trained_model2.model")
# word_vectors.save("Word2Vec_Models/trained_model2.wordvectors")
#
# # Load back with memory-mapping = read-only, shared across processes.
# wv = KeyedVectors.load("Word2Vec_Models/trained_model2.wordvectors", mmap='r')
# model_2 = KeyedVectors.load_word2vec_format("Word2Vec_Models/trained_model2.model")
#
# # fit a 2d PCA model to the vectors
# X = model_2[model_2.wv.vocab]
# pca = PCA(n_components=2)
# result = pca.fit_transform(X)
# # create a scatter plot of the projection
# plt.figure()
# plt.title('Word2Vec Word Embedding Plots')
# plt.scatter(result[:, 0], result[:, 1])
# words = list(model_2.wv.vocab)
# for i, word in enumerate(words):
#     if word in words_to_plot:
#         plt.annotate(word, xy=(result[i, 0], result[i, 1]))
# plt.show()
#
# model = Word2Vec.load("Word2Vec_Models/trained_model2.model")
# similar_1 = model.wv.most_similar('neural')
# print('similar to neural', similar_1)
# similar_2 = model.wv.most_similar('tesla')
# print('similar to tesla', similar_2)
# similar_3 = model.wv.most_similar('child')
# print('similar to child', similar_3)

##
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

embeddings_dict = {}


"""The google pre-trained model already contains some phrases (bigrams) from gensim.models.keyedvectors import KeyedVectors
"""

## Convert .bin to .txt format (only needs run once)
# model = KeyedVectors.load_word2vec_format('Google_WordVectors/GoogleNews-vectors-negative300.bin', binary=True)
# model.save_word2vec_format('Google_WordVectors/GoogleNews-vectors-negative300.txt', binary=False)

## Go
from gensim.models import KeyedVectors
Glove_dir = r'GloVe/glove.840B.300d.txt'
Google_dir = r'Google_WordVectors/GoogleNews-vectors-negative300.txt'

# # looking at the vocab
# model_google = KeyedVectors.load_word2vec_format('Google_WordVectors/GoogleNews-vectors-negative300.bin', binary=True)
# words = list(model_google.wv.vocab)[:100000]
# phrases = [word for word in words if '_' in word]
# print('\n', phrases, '\n')


with open(Google_dir, 'r', errors='ignore', encoding='utf8') as f:
    try:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    except:
        f.__next__()

def find_closest_embeddings(embedding):
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))

# print(find_closest_embeddings(embeddings_dict["king"])[1:6])
#
# print(find_closest_embeddings(
#     embeddings_dict["twig"] - embeddings_dict["branch"] + embeddings_dict["hand"]
# )[:5])

tsne = TSNE(n_components=2, random_state=0)
words = [] #list(embeddings_dict.keys())
words_unplotted = []
vectors = []
print('nouns_to_plot', nouns_to_plot)
print('keywords to plot', top_keywords_final)
#or both
both = itertools.chain(nouns_to_plot, top_keywords_final)

for word in both:          #top_keywords: #nouns_to_plot:
    # check all cases... capitals, non capitals, etc
    #if double word, capitalise and put back together
    print('\nword:', word)
    if "_" in word:
        split_phrase = word.split("_")
        new_word = []
        for i in word.split("_"):
            new_word.append(i.title())
        capitalised_phrase = "_".join(new_word)
        possible_versions_of_word = [word, capitalised_phrase, word.upper()]
    else:
        possible_versions_of_word = [word, word.title(), word.upper()] # maybe add a check if the stem of a word is there?
    # print('boolean', boolean)
    boolean = [x in embeddings_dict for x in possible_versions_of_word]
    if any(boolean):
        # print('list(np.where(boolean)[0])[0]: ', list(np.where(boolean)[0])[0])
        idx = int(list(np.where(boolean)[0])[0]) #might have issue when multiple possible_versions are in vocab
        # print('idx', idx)
        true_word = possible_versions_of_word[idx]
        # print('true_word', true_word)
        words.append(true_word)
        vectors.append(embeddings_dict[true_word])
    else:
        words_unplotted.append(word)
##
Y = tsne.fit_transform(vectors[:250])

plt.scatter(Y[:, 0], Y[:, 1])

for label, x, y in zip(words, Y[:, 0], Y[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points")
plt.show()

print(len(words_unplotted), words_unplotted)

"""model.most_similar("woman")   model.similarity("girl", "woman")"""
