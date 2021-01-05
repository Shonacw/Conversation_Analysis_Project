"""
#Breakdown of Code...


#Notes mentioned in code...
NOTE G:
    The google pre-trained model already contains some phrases (bigrams)

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

import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import pandas as pd

## Pre-processing Functions...
def Prep_Content_for_Ngram_Extraction(content):
    """
    Given raw 'content' read in from .txt transcript, process into a list of lower case words
    from which useful bigrams and trigrams can be extracted.
    """
    content_tokenized = word_tokenize(content)
    words = [w.lower() for w in content_tokenized]
    return words

def Extract_bigrams(words):
    """
    Function to extract interesting bigrams used in the vocabulary of a podcast. Input is a list of words
    from the podcast transcript.
    """
    stopset = set(stopwords.words('english'))
    filter_stops = lambda w: len(w) < 3 or w in stopset

    bcf = BigramCollocationFinder.from_words(words)
    bcf.apply_word_filter(filter_stops)               # Ignore bigrams whose words contain < 3 chars / are stopwords
    bcf.apply_freq_filter(3)                          # Ignore trigrams which occur fewer than 3 times in the transcript
    bigrams_extracted = list(list(set) for set in bcf.nbest(BigramAssocMeasures.likelihood_ratio, 20)) # Considering top 20
    final_bigrams = []
    for bigram_list in bigrams_extracted:
        bigram_0, bigram_1 = bigram_list
        bigram_condensed = str(bigram_0 + '_' + bigram_1)
        final_bigrams.append(bigram_condensed)
    print('Bigrams: ', final_bigrams)
    return final_bigrams

def Extract_trigrams(words):
    """
    Function to extract interesting trigrams used in the vocabulary of a podcast. Input is a list of words
    from the podcast transcript.
    """
    stopset = set(stopwords.words('english'))
    filter_stops = lambda w: len(w) < 3 or w in stopset

    tcf = TrigramCollocationFinder.from_words(words)
    tcf.apply_word_filter(filter_stops)               # Ignore trigrams whose words contain < 3 chars / are stopwords
    tcf.apply_freq_filter(3)                          # Ignore trigrams which occur fewer than 3 times in the transcript
    trigrams_extracted = list(list(set) for set in tcf.nbest(TrigramAssocMeasures.likelihood_ratio, 20)) # Considering top 20
    final_trigrams = []
    for trigram_list in trigrams_extracted:
        trigram_0, trigram_1, trigram_2 = trigram_list
        trigram_condensed = str(trigram_0 + '_' + trigram_1 + '_' + trigram_2)
        final_trigrams.append(trigram_condensed)
    print('Trigrams: ', final_trigrams)
    return final_trigrams

def Replace_ngrams_In_Text(content, bigrams_list, trigrams_list):
    """
    Function to replace all cases of individual words from detected n-grams with their respective bi/trigram.
    Returns the document content as a list of words.

    Note why trigrams were replaced first:
    ['brain', 'simulation'] was a detected bigram, and ['deep', 'brain', 'simulation'] was a detected trigram. If
    I condensed all the words 'brain' and 'simulation' into 'brain_simulation' then once I searched for the trigrams
    there would be none left, as it would instead have 'deep', 'brain_simulation'.
    """
    list_of_condensed_grams = []
    content_tokenized = word_tokenize(content)

    # Replace Trigrams...
    for trigram in trigrams_list:
        trigram_0, trigram_1, trigram_2 = trigram
        trigram_condensed = str(trigram_0.capitalize() + '_' + trigram_1.capitalize() + '_' + trigram_2.capitalize())
        list_of_condensed_grams.append(trigram_condensed)
        indices = [i for i, x in enumerate(content_tokenized) if x.lower() == trigram_0
                   and content_tokenized[i+1].lower() == trigram_1
                   and content_tokenized[i+2].lower() == trigram_2]
        for i in indices:
            content_tokenized[i] = trigram_condensed
            content_tokenized[i+1] = '-'              # Placeholders to maintain index numbering - are removed later on
            content_tokenized[i+2] = '-'

    # Replace Bigrams...
    for bigram in bigrams_list:
        bigram_0, bigram_1 = bigram
        bigram_condensed = str( bigram_0.capitalize() + '_' + bigram_1.capitalize())
        list_of_condensed_grams.append(bigram_condensed)
        indices = [i for i, x in enumerate(content_tokenized) if x.lower() == bigram_0
                   and content_tokenized[i+1].lower() == bigram_1]
        for i in indices:
            content_tokenized[i] = bigram_condensed
            content_tokenized[i+1] = '-'                # Placeholders to maintain index numbering - are removed later on

    return content_tokenized

def Preprocess_Sentences(content_sentences):
    """
    Function to preprocess sentences such that they are ready for keyword extraction etc.
    """
    sents_preprocessed = []
    stop_words = set(stopwords.words('english'))
    for sent in content_sentences:
        # Remove numbers
        sent = re.sub(r'\d+', '', sent)
        # Remove punctuation
        sent = re.sub(r'[^\w\s]', '', sent)
        # Make lowercase and remove stopwords
        sent_lower = [w.lower() for w in word_tokenize(sent) if w not in stop_words]
        # Join into one string so can use reg expressions
        result = ' '.join(map(str, sent_lower))
        # Stemming? Lemmatization ?

        sents_preprocessed.append(result)

    return sents_preprocessed

## Keyword Functions...
def Rake_Keywords(content, Info=False):
    """
    Function to extract keywords from document using RAKE
    """
    rake_object = rake.Rake("SmartStoplist.txt") #, 2, 3, 2 #min characters in word, max number of words in phrase, min number of times it's in text
    keywords = rake_object.run(content)
    if Info:
        print("\nRAKE Keywords:", keywords)

    return keywords

def Counter_Keywords(content_sentences, Info=False):
    """
    Function to extract the top words used in a document using a counter.
    Note these are not 'keywords', just most popular words.
    """
    sents_preprocessed = Preprocess_Sentences(content_sentences)
    sents_preprocessed_flat = reduce(operator.add, sents_preprocessed)
    keywords = Counter(sents_preprocessed_flat).most_common(10)
    if Info:
        print('\nCounter Keywords: ', keywords)

    return keywords

def PKE_keywords(content, number=30, Info=False):
    """
    Function to extract key words and phrases from a document ('content') using the PKE implementation of TopicRank.
    """
    extractor = pke.unsupervised.TopicRank()
    extractor.load_document(input=content)
    extractor.candidate_selection()
    extractor.candidate_weighting()
    keywords = extractor.get_n_best(number)
    top_keywords = []
    for i in range(len(keywords)):
        top_keywords.append(keywords[i][0])

    # Now join ngram keywords with an underscore
    top_keywords_final = []
    for keyword in top_keywords:
        # see if it is formed of >1 word
        try:
            words = word_tokenize(keyword)
            keyword = '_'.join(words)
            top_keywords_final.append(keyword)
        except:
            top_keywords_final.append(keyword)
    if Info:
        print('Pke Keywords: ', top_keywords_final, '\n')

    return top_keywords_final

def Extract_Nouns(content_sentences, Info=False):
    """
    Function to extract all potentially-interesting nouns from a given document. Used when plotting word embedding.
    """
    sents_preprocessed = Preprocess_Sentences(content_sentences)
    sents_preprocessed_flat_onestring = ' '.join(sents_preprocessed)
    words_to_plot = [word for (word, pos) in nltk.pos_tag(word_tokenize(sents_preprocessed_flat_onestring))
                     if pos[0] == 'N' and word not in ['yeah', 'yes', 'oh', 'i', 'im', 'id', 'thats', 'shes', 'dont',
                                                       'youre', 'theyll', 'youve', 'whats', 'doesnt', 'hes', 'whos',
                                                       'shouldnt']
                     and len(word) != 1]
    nouns_to_plot = list(dict.fromkeys(words_to_plot))            # Remove duplicate words
    if Info:
        print('\nExtracted Nouns: ', nouns_to_plot)

    return nouns_to_plot

def Plot_Wordcloud(content_sentences, save=False):
    """
    Function to plot a 2D Wordcloud from the top words in a given document.
    """
    sents_preprocessed = Preprocess_Sentences(content_sentences)
    sents_preprocessed_flat = reduce(operator.add, sents_preprocessed)
    stop_words = set(stopwords.words('english'))

    wordcloud = WordCloud(background_color='white',
                            stopwords=stop_words,
                            max_words=100,
                            max_font_size=50,
                            random_state=42).generate(str(sents_preprocessed_flat))
    fig = plt.figure()
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    if save:
        fig.savefig("Saved_Images/WordCloud.png", dpi=900)
    return

## Word Embedding Functions...
def Convert_bin_to_txt(look_at_vocab=False):
    """
    Function to convert pre-trained word vector files from .bin to .txt format so that can explore vocabulary.
    Only used this function once but keeping in case.
    """
    path_in = 'Google_WordVectors/GoogleNews-vectors-negative300.bin'
    path_out = 'Google_WordVectors/GoogleNews-vectors-negative300.txt'
    model = KeyedVectors.load_word2vec_format(path_in, binary=True)
    model.save_word2vec_format(path_out, binary=False)

    if look_at_vocab:
        model_google = KeyedVectors.load_word2vec_format(path_out, binary=True)
        words = list(model_google.wv.vocab)[:100000]     # Only looking at first hundred thousand
        phrases = [word for word in words if '_' in word]
        print('\nVocab containing an underscore from Google model:', phrases)
    return

def find_closest_embeddings(embeddings_dict):
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))

def Check_Embedding(embeddings_dict):
    print(find_closest_embeddings(embeddings_dict["king"])[1:6])
    print(find_closest_embeddings(embeddings_dict["twig"] - embeddings_dict["branch"] + embeddings_dict["hand"])[:5])
    #model.most_similar("woman")   model.similarity("girl", "woman")
    return


## Functions for extracting/ plotting pre-trained word embeddings for specific vocabulary from given document

def Extract_Embeddings_For_Keywords(words_to_extract, embeddings_dict, Info=False):
    """
    Function for extracting the embedding vectors for certain keywords I would like to plot on a word embedding graph.
    By default will extract the embeddings for the top 30 PKE keywords and all potentially-interesting nouns.

    Note A:
        Must check all possible versions of given word (all-capitals, non-capitals, etc) as Google Embeddings
        are inconsistent in form.
    Note B:
        Maybe add a checker of whether the STEM / LEMMA of a word exists
    Note C:
        If multiple possible versions of a given word exists in the embedding vocab, take only the first instance
    """
    # # Extract words to plot
    # if not keywords_only and not nouns_only:
    #     keywords_and_nouns = itertools.chain(PKE_keywords(content, Info=True),
    #                                          Extract_Nouns(content_sentences, Info=True))
    #     words_to_plot = keywords_and_nouns
    # elif keywords_only:
    #     words_to_plot = PKE_keywords(content, Info=True)
    # elif nouns_only:
    #     words_to_plot = Extract_Nouns(content_sentences, Info=True)

    words, vectors, words_unplotted = [], [], []
    for word in words_to_extract:
        if "_" in word:                                                      # Note A
            new_word = []
            for i in word.split("_"):
                new_word.append(i.title())
            capitalised_phrase = "_".join(new_word)
            possible_versions_of_word = [word, capitalised_phrase, word.upper()]
        else:
            possible_versions_of_word = [word, word.title(), word.upper()]  # Note B
        print('possible_versions_of_word: ', possible_versions_of_word)
        boolean = [x in embeddings_dict for x in possible_versions_of_word]
        if any(boolean):
            idx = int(list(np.where(boolean)[0])[0])                        # Note C
            true_word = possible_versions_of_word[idx]
            words.append(true_word)
            vectors.append(embeddings_dict[true_word])
        else:
            words_unplotted.append(word)
    if Info:
        print('\nNumber of Words from Document without an embedding: ', len(words_unplotted))
        print('List of Words lacking an embedding:', words_unplotted)
    return words, vectors, words_unplotted

# def Plot_Word_Embedding(path_to_transcript, path_to_pretrained_vecs):
#     """
#     Function for plotting word embeddings.
#     """

## Code...

# Load pre-processed transcript of interview between Elon Musk and Joe Rogan...
path_to_transcript = Path(
    '/Users/ShonaCW/Desktop/Imperial/YEAR 4/MSci Project/Conversation_Analysis_Project/data/shorter_formatted_plain_labelled.txt')

Glove_path = r'GloVe/glove.840B.300d.txt'
Google_path = r'Google_WordVectors/GoogleNews-vectors-negative300.txt'  # NOTE G

path_to_vecs = Google_path

# Get content from given transcript
with open(path_to_transcript, 'r') as f:
    content = f.read()
    content_sentences = nltk.sent_tokenize(content)

words = Prep_Content_for_Ngram_Extraction(content)
print("extracted content/sentences/words")
print("getting embeddings...")
# Get embeddings dictionary of word vectors  from pre-trained word embedding
embeddings_dict = {}
with open(path_to_vecs, 'r', errors='ignore', encoding='utf8') as f:
    try:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    except:
        f.__next__()
print("got embeddings")
# # save embeddings dict
# df = pd.DataFrame([embeddings_dict])
# df.to_hdf('embeddings_dict.h5', key='df', mode='w')
#
# #load embeddings dict
# df = pd.read_hdf('embeddings_dict.h5', key='df')
# embeddings_dict = df.to_dict()
# Extract words to plot
nouns_set = Extract_Embeddings_For_Keywords(Extract_Nouns(content_sentences, Info=True), embeddings_dict, Info=True)
print('done nouns')
pke_set = Extract_Embeddings_For_Keywords(PKE_keywords(content, Info=True), embeddings_dict, Info=True)
print('done pke')
bigram_set = Extract_Embeddings_For_Keywords(Extract_bigrams(words), embeddings_dict, Info=True)
print('done bigrams')
trigram_set = Extract_Embeddings_For_Keywords(Extract_trigrams(words), embeddings_dict, Info=True)
print('done trigrams')


# Plot
tsne = TSNE(n_components=2, random_state=0)
sets_to_plot = [nouns_set, pke_set, bigram_set, trigram_set]
colours = ['blue', 'green', 'orange', 'pink']
labels = ['Nouns', 'PKE Keywords', 'Bigrams', 'Trigrams']

last_noun_vector = len(nouns_set[0])
print('last_noun_vector', last_noun_vector)
last_pke_vector = last_noun_vector + len(pke_set[0])
print('last_pke_vector', last_pke_vector)
last_bigram_vector = last_pke_vector + len(bigram_set[0])
print('last_bigram_vector', last_bigram_vector)
last_trigram_vector = last_bigram_vector + len(trigram_set[0])
print('last_trigram_vector', last_trigram_vector)

all_vectors = itertools.chain(nouns_set[1], pke_set[1], bigram_set[1], trigram_set[1])
print('number of vectors', len(all_vectors))
Y = tsne.fit_transform(all_vectors)
plt.figure()

n = [0, last_noun_vector, last_pke_vector, last_bigram_vector, last_trigram_vector]
cnt = 0
for idx, set in enumerate(sets_to_plot):
    words, vectors, words_unplotted = set

    plt.scatter(Y[n[cnt]:n[cnt+1], 0], Y[n[cnt]:n[cnt+1], 1], c=colours[idx], label=labels[idx])

    for label, x, y in zip(words, Y[n[cnt]:n[cnt+1], 0], Y[n[cnt]:n[cnt+1], 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points")
    print('\nPlotted', labels[idx])
    print('words unplotted from', labels[idx],': ', words_unplotted)
    cnt +=1

plt.legend()
plt.show()


