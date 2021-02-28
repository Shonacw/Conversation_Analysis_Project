"""
#Breakdown of Code...

#Notes mentioned in code...

"""
import torch
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors
import tensorflow_hub as hub
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from gensim.models import Phrases #Phraser
from gensim.models import FastText as ft
import fasttext
import fasttext.util
import pprint

import re
import spacy
import pke
import RAKE as rake
from collections import Counter
from nltk.collocations import BigramCollocationFinder
from nltk.corpus import stopwords
from nltk.metrics import BigramAssocMeasures
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import TrigramAssocMeasures

import os
import itertools
from pathlib import Path
import operator
from functools import reduce
import numpy as np
import pandas as pd
import sys
import unicodedata
from collections import defaultdict
from pprint import pprint
from matplotlib.lines import Line2D
from operator import add
import tabulate
import string
import more_itertools as mit
from matplotlib import cm

import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib import cm

from InferSent.models import InferSent
import importlib
# topics = importlib.import_module("msci-project.src.topics")
Analysis = importlib.import_module("Analysis")
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

## Functions for pre-processing...
# def Process_Transcript(text, names, Info=False):
#     """
#     Function to remove names of speakers / times of Utterances from transcript, returning all spoken utterances as a
#     single string.
#     """
#     if Info:
#         print('Text format before preprocessing:\n', text[:300])
#
#     # Remove speaker names
#     for name in names:
#         text = re.sub(name, "", text)
#     # Get rid of the time marks
#     content_1 = re.sub('[0-9]{2}:[0-9]{2}:[0-9]{2}', " ", text)  # \w+\s\w+;
#     # Strip new-lines
#     content_2 = re.sub('\n', " ", content_1)
#     # Strip white spaces
#     content_2.strip()
#
#     if Info:
#         print('\nText format after preprocessing:\n', content_2)
#
#     return content_2


def Extract_Names(transcript_name):
    """
    Function to extract the names of the speakers given the transcript name.
    Only works for Joe Rogan transcripts so far.
    """
    upper_names = []
    for i in transcript_name.split("_"):
        upper_names.append(i.title())
    first_name = " ".join(upper_names[:2])
    second_name = " ".join(upper_names[2:])

    return [first_name, second_name]

def Preprocess_Content(content_utterances):
    """
    Function to perform Lemmatization of the whole transcript when it is first imported.
    """
    nlp = spacy.load('en', disable=['parser', 'ner'])

    content_utterances_cleaned = []
    for utterance in content_utterances:
        utt = nlp(utterance)
        content_lemma = " ".join([token.lemma_ for token in utt])
        content_lemma = re.sub(r'-PRON-', "", content_lemma)
        content_utterances_cleaned.append(content_lemma)

    return content_utterances_cleaned

def Replace_ngrams_In_Text(content, bigrams_list, trigrams_list):
    """
    Function to replace all cases of individual words from detected n-grams with their respective bi/trigram.
    Returns the document content as a list of words.

    Note why trigrams were replaced first:
    ['brain', 'simulation'] was a detected bigram, and ['deep', 'brain', 'simulation'] was a detected trigram. If
     all the cases of 'brain' and 'simulation' are condensed into 'brain_simulation' first, then there would be no cases
     of 'deep', 'brain', and 'simulation' left in the transcript and therefore no trigrams would be found.
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
            content_tokenized[i+1] = '-'               # Placeholders to maintain index numbering - are removed later on

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
        sents_preprocessed.append(result)

    return sents_preprocessed

## Keyword Functions...
def Rake_Keywords(content, Info=False):
    """
    Function to extract keywords from document using RAKE (https://pypi.org/project/rake-nltk/).
    """
    rake_object = rake.Rake("data/Rake_SmartStoplist.txt")
    keywords = rake_object.run(content)
    if Info:
        print("\nRAKE Keywords:", keywords)

    return keywords

def Counter_Keywords(content_sentences, Info=False):
    """
    Function to extract the top words used in a document using a simple counter.
    Note these are not really 'keywords'; just the most common words.
    """
    sents_preprocessed = Preprocess_Sentences(content_sentences)
    sents_preprocessed_flat = reduce(operator.add, sents_preprocessed)
    keywords = Counter(sents_preprocessed_flat).most_common(10)
    if Info:
        print('\nCounter Keywords: ', keywords)

    return keywords

def PKE_Keywords(content, number=30, put_underscore=True, Info=False):
    """
    Function to extract keywords and phrases from a document using the PKE implementation of TopicRank.
    pke info: https://github.com/boudinfl/pke
    """
    extractor = pke.unsupervised.TopicRank()
    extractor.load_document(input=content)
    extractor.candidate_selection()
    extractor.candidate_weighting()
    keywords = extractor.get_n_best(number)
    top_keywords = []
    for i in range(len(keywords)):
        top_keywords.append(keywords[i][0])
    if not put_underscore:
        top_keywords_final = top_keywords

    if put_underscore:
        # Join ngram keywords with an underscore
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

def Extract_Nouns(content_sentences, Info=True):
    """
    Function to extract all potentially-interesting nouns from a given document. Used when plotting word embedding.

    Okay so the issue is that the spacy.nlp is rightly labelling some words which I think of as verbs as 'PROPN's given
    their position in the original text... i.e. "meet" when its tagged from the original transcript is
    -> ...('nice', 'PROPN'), ('meet', 'PROPN'), ('yeah', 'INTJ')....

    -> nlp('meet') alone does say that it's a verb, but all cases of 'meet' in the transcript are labelled as PROPN.

    SO I will run the nlp() tagger on each unique word in the transcript individually, i.e. assign the tag with no
    understanding of word context. This catches out more verbs.
    """
    nlp = spacy.load("en_core_web_sm")              #en_core_web_sm   #en_core_web_lg

    sents_preprocessed = Preprocess_Sentences(content_sentences)
    sents_preprocessed_flat_onestring = " ".join(sents_preprocessed)
    words = word_tokenize(sents_preprocessed_flat_onestring)
    words_without_duplicates  = list(dict.fromkeys(words))          # remove duplicate words
    tokens = [nlp(word)[0] for word in words_without_duplicates]    # tag INDIVIDUALLY
    all_pairs = [(token.text, token.pos_) for token in tokens]      # if interested later
    only_nouns = [token.text for token in tokens if token.pos_ in ['NOUN', 'PROPN']]

    my_toremove_list = ['use', 'react', 'reply', 'emerge', 'roll', 'thing', 'way', 'lot', 'super', 'lay', 'part']
    nouns_to_plot = [word for word in only_nouns if word not in my_toremove_list]

    if Info:
        print('-Number of unique nouns extracted with en_core_web_sm: ', len(nouns_to_plot))
        print('-nouns_to_plot', nouns_to_plot)

    return nouns_to_plot

def Extract_bigrams(words, n=20, put_underscore=True, Info=False):
    """
    Function to extract bigrams mentioned in the given transcript.
    Input is the Podcast transcript in the form of a list of words and 'n' the number of Bigrams to consider.
    """
    stopset = set(stopwords.words('english'))
    filter_stops = lambda w: len(w) < 3 or w in stopset

    bcf = BigramCollocationFinder.from_words(words)
    bcf.apply_word_filter(filter_stops)               # Ignore bigrams whose words contain < 3 chars / are stopwords
    bcf.apply_freq_filter(3)                          # Ignore trigrams which occur fewer than 3 times in the transcript
    bigrams_extracted = list(list(set) for set in bcf.nbest(BigramAssocMeasures.likelihood_ratio, n))
    final_bigrams = []
    for bigram_list in bigrams_extracted:
        bigram_0, bigram_1 = bigram_list
        if put_underscore:
            bigram_condensed = str(bigram_0 + '_' + bigram_1)
        else:
            bigram_condensed = str(bigram_0 + ' ' + bigram_1)
        final_bigrams.append(bigram_condensed)

    if Info:
        print('Bigrams: ', final_bigrams)
    return final_bigrams

def Extract_trigrams(words, n=20, put_underscore=True, Info=False):
    """
    Function to extract interesting trigrams mentioned in the given transcript.
    Input is a list of words from the podcast transcript and 'n' is the max number of trigrams to consider.
    """
    stopset = set(stopwords.words('english'))
    filter_stops = lambda w: len(w) < 3 or w in stopset

    tcf = TrigramCollocationFinder.from_words(words)
    tcf.apply_word_filter(filter_stops)               # Ignore trigrams whose words contain < 3 chars / are stopwords
    tcf.apply_freq_filter(3)                          # Ignore trigrams which occur fewer than 3 times in the transcript
    trigrams_extracted = list(list(set) for set in tcf.nbest(TrigramAssocMeasures.likelihood_ratio, n))
    final_trigrams = []
    for trigram_list in trigrams_extracted:
        trigram_0, trigram_1, trigram_2 = trigram_list
        if put_underscore:
            trigram_condensed = str(trigram_0 + '_' + trigram_1 + '_' + trigram_2)
        else:
            trigram_condensed = str(trigram_0 + ' ' + trigram_1 + ' ' +trigram_2)
        final_trigrams.append(trigram_condensed)

    if Info:
        print('Trigrams: ', final_trigrams)
    return final_trigrams

## Functions for dealing with Keywords...

def Extract_Embeddings_For_Keywords(words_to_extract, word2vec_embeddings_dict, fasttext_model, embedding_method,
                                    shift_ngrams=False, Info=False):
    """
    Function for extracting the word vectors for the given keywords.

    Note A:
        Must check all possible versions of given word (all-capitals, non-capitals, etc) as Google Embeddings
        are inconsistent in form.
    Note B:
        Maybe add a checker of whether the STEM / LEMMA of a word exists
    Note C:
        If multiple possible versions of a given word exists in the embedding vocab, take only the first instance
    """
    words, vectors, words_unplotted = [], [], []
    nlp = spacy.load("en_core_web_sm")  # doing this out here so don't need to keep reloading

    # Word2Vec
    if embedding_method == 'word2vec':
        for word in words_to_extract:
            if shift_ngrams and (len(word_tokenize(word)) > 1 or '_' in word):
                if len(word_tokenize(word)) > 1 or '_' in word:
                    word_to_use = get_word_from_ngram(word, nlp)
                    if word_to_use == 'nan':
                        words_unplotted.append(word)
                        continue
                    words.append(word) # save original version of ngram string... but altered version of embedding
                    vectors.append(word2vec_embeddings_dict[word_to_use])
            else:
                if "_" in word:  # Note A
                    new_word = []
                    for i in word.split("_"):
                        new_word.append(i.title())
                    capitalised_phrase = "_".join(new_word)
                    possible_versions_of_word = [word, capitalised_phrase, word.upper()]
                else:
                    possible_versions_of_word = [word, word.title(), word.upper()]  # Note B

                if Info:
                    print('possible_versions_of_word: ', possible_versions_of_word)

                boolean = [x in word2vec_embeddings_dict for x in possible_versions_of_word]
                if any(boolean):
                    idx = int(list(np.where(boolean)[0])[0])                        # Note C
                    true_word = possible_versions_of_word[idx]
                    words.append(true_word)
                    vectors.append(word2vec_embeddings_dict[true_word])
                else:
                    words_unplotted.append(word)

    # FastText
    if embedding_method == 'fasttext':
        embedding_dict = {}
        for word in words_to_extract:
            # for word in words_to_extract:
            if shift_ngrams and (len(word_tokenize(word)) > 1 or '_' in word): #i.e. only perform this if the word is an ngram
                    word_to_use = get_word_from_ngram(word, nlp)
                    if word_to_use == 'nan':# deal with un-useful keywords
                        words_unplotted.append(word)
                        continue
                    else:
                        embedding_dict[word] = fasttext_model.get_word_vector(word_to_use)
            else:
                try:
                    embedding_dict[word] = fasttext_model.get_word_vector(word)
                except:
                    words_unplotted.append(word)
        words, vectors = list(embedding_dict.keys()), list(embedding_dict.values())

    #if Info:
    #print('Number of Words from Document without an embedding: ', len(words_unplotted))
    print('Words lacking an embedding:', words_unplotted)

    return words, vectors, words_unplotted

def Extract_Keyword_Embeddings(content, content_sentences, embedding_method, transcript_name, put_underscore_ngrams=True,
                               shift_ngrams=False, return_all=False, Info=False):
    """
    Function to extract all types of keywords from transcript + obtain their word embeddings. Only needs to be run once
    then all the keywords + their embeddings are stored in a dataframe 'keyword_vectors_df' which is saved to hdf
    for easy loading in future tasks.

    mode

    Note A:
        Currently using GoogleNews pretrained word vectors, but could also use Glove. The benefit of the Google model is
        that it contains vectors for some 'phrases' (bigrams/ trigrams) which is helpful for the plot being meaningful!
    Note B:
        The tsne vector dimensionality must be done all together, but in a way that I can then split the vectors back
        into groups based on keyword type. Hence why code a little more fiddly.

    """

    if Info:
        print('\n-Extracting keywords + obtaining their word vectors using GoogleNews pretrained model...')

    content_tokenized = word_tokenize(content)
    words = [w.lower() for w in content_tokenized]

    if Info:
        print("-Extracted content/sentences/words from transcript.")

    # Collect keywords
    nouns_list = Extract_Nouns(content_sentences, Info=False)
    pke_list = PKE_Keywords(content, number=30, put_underscore=put_underscore_ngrams, Info=False,)
    bigrams_list = Extract_bigrams(words, n=20, put_underscore=put_underscore_ngrams, Info=False)
    trigrams_list = Extract_trigrams(words, put_underscore=put_underscore_ngrams, Info=False)
    all_keywords = list(itertools.chain(nouns_list, pke_list, bigrams_list, trigrams_list))

    if Info:
        print("-Extracted all keywords.")

    if embedding_method == 'word2vec':
        # Choose pre-trained model...   Note A
        Glove_path = r'GloVe/glove.840B.300d.txt' #GloVe
        Google_path = r'Google_WordVectors/GoogleNews-vectors-negative300.txt' # Word2Vec
        path_to_vecs = Google_path

        # Get embeddings dictionary of word vectors  from pre-trained word embedding
        embeddings_dict = {}
        if Info:
            print("-Obtaining keyword word vectors using GoogleNews embeddings... (this takes a while)")
        with open(path_to_vecs, 'r', errors='ignore', encoding='utf8') as f:
            try:
                for line in f:
                    values = line.split()
                    word = values[0]
                    vector = np.asarray(values[1:], "float32")
                    embeddings_dict[word] = vector
            except:
                f.__next__()

        # Extract words to plot
        nouns_set = Extract_Embeddings_For_Keywords(nouns_list, embeddings_dict, None, embedding_method='word2vec')
        pke_set = Extract_Embeddings_For_Keywords(pke_list, embeddings_dict, None, embedding_method='word2vec', shift_ngrams=shift_ngrams)
        bigram_set = Extract_Embeddings_For_Keywords(bigrams_list, embeddings_dict, None,  embedding_method='word2vec', shift_ngrams=shift_ngrams)
        trigram_set = Extract_Embeddings_For_Keywords(trigrams_list, embeddings_dict, None, embedding_method='word2vec', shift_ngrams=shift_ngrams)

    if embedding_method == 'fasttext':
        ft = fasttext.load_model(
            '/Users/ShonaCW/Desktop/Imperial/YEAR 4/MSci Project/Conversation_Analysis_Project/FastText/cc.en.300.bin')
        fasttext.util.reduce_model(ft, 100) # Reduce dimensionality of FastText vectors from 300->100

        # Extract words to plot
        nouns_set = Extract_Embeddings_For_Keywords(nouns_list, None, ft, embedding_method='fasttext')
        pke_set = Extract_Embeddings_For_Keywords(pke_list, None, ft, embedding_method='fasttext', shift_ngrams=shift_ngrams)
        bigram_set = Extract_Embeddings_For_Keywords(bigrams_list, None, ft, embedding_method='fasttext', shift_ngrams=shift_ngrams)
        trigram_set = Extract_Embeddings_For_Keywords(trigrams_list, None, ft, embedding_method='fasttext', shift_ngrams=shift_ngrams)

    if Info:
        print('-Extracted {0} embeddings for all keywords.'.format(embedding_method))

    # Reduce dimensionality of word vectors such that we can store X and Y positions.
    tsne = TSNE(n_components=2, random_state=0)
    sets_to_plot = [nouns_set, pke_set, bigram_set, trigram_set]

    last_noun_vector = len(nouns_set[0])
    last_pke_vector = last_noun_vector + len(pke_set[0])
    last_bigram_vector = last_pke_vector + len(bigram_set[0])
    last_trigram_vector = last_bigram_vector + len(trigram_set[0])
    all_vectors = list(itertools.chain(nouns_set[1], pke_set[1], bigram_set[1], trigram_set[1]))

    if return_all:
        return all_keywords

    # Store keywords + embeddings in a pandas data-frame
    keyword_vectors_df = pd.DataFrame(columns = ['noun_keyw',   'noun_X',    'noun_Y', 'unfamiliar_noun',
                                                 'pke_keyw',     'pke_X',    'pke_Y', ' unfamiliar_pke',
                                                 'bigram_keyw',  'bigram_X', 'bigram_Y', 'unfamiliar_bigram',
                                                 'trigram_keyw', 'trigram_X','trigram_Y', 'unfamiliar_trigram'
                                                 ])

    keyword_vectors_df.loc[:, 'noun_keyw'] = pd.Series(nouns_set[0])
    keyword_vectors_df.loc[:, 'unfamiliar_noun'] = pd.Series(nouns_set[2])

    keyword_vectors_df.loc[:, 'pke_keyw'] = pd.Series(pke_set[0])
    keyword_vectors_df.loc[:, 'unfamiliar_pke'] = pd.Series(pke_set[2])

    keyword_vectors_df.loc[:, 'bigram_keyw'] = pd.Series(bigram_set[0])
    keyword_vectors_df.loc[:, 'unfamiliar_bigram'] = pd.Series(bigram_set[2])

    keyword_vectors_df.loc[:, 'trigram_keyw'] = pd.Series(trigram_set[0])
    keyword_vectors_df.loc[:, 'unfamiliar_trigram'] = pd.Series(trigram_set[2])

    reduced_vectors = tsne.fit_transform(all_vectors)                           #Note B

    n = [0, last_noun_vector, last_pke_vector, last_bigram_vector, last_trigram_vector, -1]
    col_names_X, col_names_Y = ['noun_X', 'pke_X', 'bigram_X', 'trigram_X'], ['noun_Y', 'pke_Y', 'bigram_Y', 'trigram_Y']
    for i in range(len(sets_to_plot)):
        Xs = reduced_vectors[n[i]:n[i + 1], 0]
        Ys = reduced_vectors[n[i]:n[i + 1], 1]
        keyword_vectors_df.loc[:, col_names_X[i]] = pd.Series(Xs)
        keyword_vectors_df.loc[:, col_names_Y[i]] = pd.Series(Ys)

    if not put_underscore_ngrams:
        # Store keywords + embeddings in a hd5 file for easy accessing in future tasks.
        keyword_vectors_df.to_hdf('Saved_dfs/{0}/keyword_vectors_nounderscore_{1}_df.h5'.format(transcript_name,
                                                                                                embedding_method), key='df', mode='w')
    if put_underscore_ngrams:
        # Store keywords + embeddings in a hd5 file for easy accessing in future tasks.
        keyword_vectors_df.to_hdf('Saved_dfs/{0}/keyword_vectors_underscore_{1}_df.h5'.format(transcript_name,
                                                                                              embedding_method), key='df', mode='w')

    if Info:
        print('-Created and saved keyword_vectors_df dataframe.')

    return nouns_set, pke_set, bigram_set, trigram_set

def get_word_from_ngram(ngram, nlp):
    """
    Function to relocate n_grams to coordinates by the noun they contain.
    """
    # loop through pke keywords and look at the ngrams
    if '_' in ngram: #if joined by an undrscore replace it with a space so can use word_tokenize in next part...
        ngram = ngram.split("_")
        ngram = ' '.join(ngram)

    pos_list = [word.pos_ for word in nlp(str(ngram))] # if word.pos_ in ['NOUN']]
    if len(pos_list) > 1:
        counter = Counter(pos_list)
        words = word_tokenize(ngram)
        if 'NOUN' in counter:
            word_to_use = words[pos_list.index('NOUN')] # TO DO deal with case of two nouns

        elif 'PROPN' in counter:
            word_to_use = words[pos_list.index('PROPN')]
        else:
            return 'nan'          # if the phrase doesn't contain a noun dont want it as a keyphrase at all

    return word_to_use


def Find_Keywords_in_Segment(sents_in_segment, all_keywords, Info=False):
    """
    Function to search segment-by-segment for the given keywords.
    """
    keywords_contained_in_segment = {}

    sents_in_subsection_flat = ''.join(list(itertools.chain.from_iterable(sents_in_segment)))
    words_in_subsection = word_tokenize(sents_in_subsection_flat)

    # Convert to strings
    all_keywords = [str(word) for word in all_keywords]
    word_list = [word for word in all_keywords if len(word.split('_')) == 1]
    bigram_list = [word for word in all_keywords if len(word.split('_')) == 2]
    trigram_list = [word for word in all_keywords if len(word.split('_')) == 3]

    # Firstly search for all one-word keywords (nouns and pke)
    content = ' '.join(sents_in_segment)

    for word in word_list:
        count = len(re.findall(' ' + str(word) + ' ', content))
        # counter_dict[str(word)] = count
        # count = sents_in_subsection_flat.lower().split().count(' ' + word + ' ') #to make sure it only counts full words
        if count != 0:
            keywords_contained_in_segment[word] = count

    # Then search for bigrams/trigrams
    for trigram in trigram_list:
        trigram_0, trigram_1, trigram_2 = [w.lower() for w in trigram.split('_')]
        indices = [i for i, x in enumerate(words_in_subsection) if x.lower() == trigram_0
                   and words_in_subsection[i + 1].lower() == trigram_1
                   and words_in_subsection[i + 2].lower() == trigram_2]
        if len(indices) != 0:
            keywords_contained_in_segment[trigram] = len(indices)

    for bigram in bigram_list:
        bigram_0, bigram_1 = [w.lower() for w in bigram.split('_')]
        indices = [i for i, x in enumerate(words_in_subsection) if x.lower() == bigram_0
                   and words_in_subsection[i + 1].lower() == bigram_1]
        if len(indices) != 0:
            keywords_contained_in_segment[bigram] = len(indices)

    if Info:
        print('keywords_contained_in_segment dictionary', keywords_contained_in_segment)

    return keywords_contained_in_segment

## Functions for Segmentation...
def split_segs(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def Peform_Segmentation(content_sentences, segmentation_method='Even', Num_Even_Segs=10, cos_sim_limit=0.52, Plot=False, save_fig=False):
    """
    Function to segment up a transcript. By default will segment the transcript into 'Num_Even_Segs' even segments.
    Returns a list containing the indices of the first sentence of each segment, 'first_sent_idxs_list'.

    NOTE:
        the segment tags "===== bla bla" will count as a sentence in 'SliceCast' method (and their index will
        be given as the sentence number at which a new section starts), whereas in the 'InferSent' method,
         these tags will be removed from the document before segmentation. Therefore the sentence indices at which
         new segments start will never match up between these methods; there is a different number of sentences in each.
    """
    print('-Performing Transcript Segmentation with', seg_method, '...')

    if segmentation_method == 'InferSent':
        # 1
        # Obtain sentence embeddings using InferSent + create dataframe of consec sents cosine similarity + predict segmentation
        embeddings = Obtain_Sent_Embeddings_InferSent(content_sentences, Info=False)

        # Obtain cosine similarity info dataframe
        cos_sim_df = Calc_CosSim_InferSent(content_sentences, embeddings, cos_sim_limit, Info=True)

        # [OR if embeddings were already obtained, simply load dataframe]
        # cos_sim_df = pd.read_hdf('Saved_dfs/InferSent_cos_sim_df.h5', key='df')

        # Plot
        if Plot:
            Plot_InferSent_Clusters(content_sentences, cos_sim_df, cos_sim_limit, save_fig=save_fig)

        # 2
        first_sent_idxs_list = []
        df_mini = cos_sim_df[cos_sim_df['New_Section'] == 1]
        for idx, row in df_mini.iterrows():
            first_sent_idxs_list.append(row['Sentence2_idx'])

    if segmentation_method == 'Even':
        num_sents = len(content_sentences)
        idx_split = split_segs(range(num_sents), Num_Even_Segs)
        first_sent_idxs_list = [i[0] for i in idx_split][1:]

    if seg_method == 'SliceCast':
        first_sent_idxs_list = SliceCast_Segmentation(content_sentences, doc_labelled=True, Plot=Plot, save_fig=save_fig)

    print('Done')
    return first_sent_idxs_list


def get_segments_info(first_sent_idxs_list, content_sentences, keyword_vectors_df, folder_name, save_name='segments_info_df', Info=False):
    """
    Function to perform segment-wise keyword analysis.
    Collects information about they keywords contained in each segment of the transcript.

    params
     'first_sent_idxs_list':
        List containing the index of sentences which start a new segment.

    Returns a dataframe 'segments_info_df'

    NOTE on keyword averaging:
        If segments are small enough, i.e. only 2-5 utterances at a time, should only really contain a couple of keywords
        that will be (hopefully) semantically similar.

    """
    if Info:
        print('\n-Obtaining information about each segment using cos_sim_df...')

    def getIndexes(dfObj, value):
        ''' Get index positions of value in dataframe i.e. dfObj.'''
        listOfPos = list()
        # Get bool dataframe with True at positions where the given value exists
        result = dfObj.isin([value])
        # Get list of columns that contains the value
        seriesObj = result.any()
        columnNames = list(seriesObj[seriesObj == True].index)
        # Iterate over list of columns and fetch the rows indexes where value exists
        for col in columnNames:
            rows = list(result[col][result[col] == True].index)
            for row in rows:
                listOfPos.append([row, col])
                # listOfPos.append(col)
        # Return a list of tuples indicating the positions of value in the dataframe
        return listOfPos

    all_keywords = list(itertools.chain(keyword_vectors_df['noun_keyw'].values, keyword_vectors_df['pke_keyw'].values,
                                        keyword_vectors_df['bigram_keyw'].values,
                                        keyword_vectors_df['trigram_keyw'].values
                                        ))

    segments_dict = {'first_sent_numbers': [], 'length_of_segment': [], 'keyword_list': [], 'keyword_counts': [],
                     'total_average_keywords_wordvec': [],
                     'top_count_keyword': [], 'top_count_wordvec': [],
                     'top_3_counts_keywords': [], 'top_3_counts_wordvec': [],
                     'noun_list' : [], 'noun_counts' : [], 'top_3_counts_nouns':[], 'top_3_counts_nounwordvec':[]}

    old_idx = 0
    for idx in first_sent_idxs_list:
        # Collect basic information about this segment
        segments_dict['first_sent_numbers'].append(idx)  # POSITION of each section
        length = np.int(idx) - np.int(old_idx)  # LENGTH of each section
        segments_dict['length_of_segment'].append(length)

        # Collect keywords contained in this segment as well as their counts
        sentences_in_segment = content_sentences[old_idx:idx]
        keywords_dict = Find_Keywords_in_Segment(sentences_in_segment, all_keywords, Info=False)
        keywords_list, keywords_count = list(keywords_dict.keys()), list(keywords_dict.values())
        segments_dict['keyword_list'].append(keywords_list)
        segments_dict['keyword_counts'].append(keywords_count)

        # Find noun keywords
        nouns_dict = Find_Keywords_in_Segment(sentences_in_segment, keyword_vectors_df['noun_keyw'].values, Info=False)
        nouns_list, nouns_count = list(nouns_dict.keys()), list(nouns_dict.values())
        segments_dict['noun_list'].append(nouns_list)
        segments_dict['noun_counts'].append(nouns_count)

        # Collect the Word embeddings for the keywords contained in this segment
        df = keyword_vectors_df
        Xvecs, Yvecs = [], []
        for keyword in keywords_list:
            # get index for row and column
            row, col_name = getIndexes(df, keyword)[0]
            col_idx = [idx for idx, val in enumerate(df.columns)][0]
            # get the average vector position  #note !need to make sure columns are in right order
            Xvecs.append(df.iloc[row, col_idx + 1])
            Yvecs.append(df.iloc[row, col_idx + 2])

        # Collect vectors for possible locations to place the node representing each segment
        segments_dict['total_average_keywords_wordvec'].append([np.mean(Xvecs), np.mean(Yvecs)])

        if len(keywords_count) >= 1 :
            idx_of_top_keyword = keywords_count.index(max(keywords_count))
            keywords_list = np.array(keywords_list)
            top_keyword, top_keyword_XY = keywords_list[idx_of_top_keyword], [Xvecs[idx_of_top_keyword],
                                                                          Yvecs[idx_of_top_keyword]]
        else:
            top_keyword = None
            top_keyword_XY = None
        segments_dict['top_count_keyword'].append(top_keyword)
        segments_dict['top_count_wordvec'].append(top_keyword_XY)

        # Check that there are at least 3 keywords for the section
        Xvecs, Yvecs = np.array(Xvecs), np.array(Yvecs)

        if len(keywords_list) >= 3:
            idxs_of_top_3_keywords = sorted(range(len(keywords_count)), key=lambda i: keywords_count[i])[-3:]
            top_3_keywords = keywords_list[idxs_of_top_3_keywords]
            #print('top_3_keywords: ', top_3_keywords)
            top_3_keywords_X, top_3_keywords_Y = Xvecs[idxs_of_top_3_keywords], Yvecs[idxs_of_top_3_keywords]
            segments_dict['top_3_counts_keywords'].append(top_3_keywords)
            segments_dict['top_3_counts_wordvec'].append([np.mean(top_3_keywords_X), np.mean(top_3_keywords_Y)])
        else:
            segments_dict['top_3_counts_keywords'].append(['nan', 'nan', 'nan'])
            segments_dict['top_3_counts_wordvec'].append('nan')

        # Now do the same but specifically for nouns
        Xvecs, Yvecs = [], []
        for noun in nouns_list:
            # get index for row and column
            row, col_name = getIndexes(df, noun)[0]
            col_idx = [idx for idx, val in enumerate(df.columns)][0]
            # get the average vector position  #note !need to make sure columns are in right order
            Xvecs.append(df.iloc[row, col_idx + 1])
            Yvecs.append(df.iloc[row, col_idx + 2])

        # Check that there are at least 3 keywords for the section
        Xvecs, Yvecs = np.array(Xvecs), np.array(Yvecs)
        nouns_list = np.array(nouns_list)
        if len(nouns_list) >= 3:
            idxs_of_top_3_keywords = sorted(range(len(nouns_count)), key=lambda i: nouns_count[i])[-3:]
            top_3_keywords = nouns_list[idxs_of_top_3_keywords]
            top_3_keywords_X, top_3_keywords_Y = Xvecs[idxs_of_top_3_keywords], Yvecs[idxs_of_top_3_keywords]
            segments_dict['top_3_counts_nouns'].append(top_3_keywords)
            segments_dict['top_3_counts_nounwordvec'].append([np.mean(top_3_keywords_X), np.mean(top_3_keywords_Y)])
        else:
            segments_dict['top_3_counts_nouns'].append(['nan', 'nan', 'nan'])
            segments_dict['top_3_counts_nounwordvec'].append('nan')


        old_idx = idx

    # Convert dictionary to dataframe
    segments_info_df = pd.DataFrame({k: pd.Series(l) for k, l in segments_dict.items()})
    segments_info_df.to_hdf('Saved_dfs/{0}/{1}.h5'.format(folder_name, save_name), key='df', mode='w')
    if Info:
        print('-Created segments_info_df. Preview: ')
        print(segments_info_df.head().to_string())
        print('Lengths of Segments:', segments_dict['length_of_segment'])
        print('#Keywords in each Segment:', [len(words_list) for words_list in segments_dict['keyword_list']])
        print('#Segments with zero keywords:', [len(words_list) for words_list in list(keywords_dict.keys())].count(0))

    return segments_info_df

def Obtain_Sent_Embeddings_InferSent(sentences, V=1, Info=False):
    """
    STEPS
    1: Use InferSent (facebook.ai) to create utterance embeddings from a podcast transcript
    2: Evaluate the Cosine Similarity between every pair of (consecutive) sentence embeddings in the whole conversation
    3: Build a network and leave nodes connected with a mutual similarity > some cutoff. Clusters are then topics, nodes
        are sentences, and edges are the similarity between sentences.

    Notes:
        Using GloVe (V1), not FastText (V2) vectors so far.
        V = 1 for GloVe, 2 for FastText
    """
    if Info:
        print('\nObtaining sentence embeddings with InferSent...')

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

    if Info:
        print('-Obtained sentence embeddings with InferSent.')

    return embeddings

def Calc_CosSim_InferSent(content_sentences, embeddings, cos_sim_limit=0.52, Info=False):
    """
    Function to
    param cos_sim_limit:
        The parameter which will determine the number of segments
        if = 0.6  there are 64 segments
        if = 0.55 there are 25 segments
        if = 0.52 there are 15 segments
        if = 0.5  there are 10 segments
        if = 0.45 there are 3 segments
    """
    def cosine(u, v):
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

    tbl = dict.fromkeys(i for i in list(range(sys.maxunicode)) if unicodedata.category(chr(i)).startswith('P'))

    if Info:
        print('\n-Creating cos_sim_df dataframe...')

    idxs_blank = []
    content_sentences_copy = content_sentences
    cos_sim_df = pd.DataFrame(columns=['Sentence1', 'Sentence1_idx', 'Sentence2', 'Sentence2_idx', 'Cosine_Similarity',
                                       'New_Section'])

    # Firstly, put all sentences of length <5 to blank in a new content_sentences_object
    # (preliminary method to deal with filler sents)
    for idx, sentence in enumerate(content_sentences_copy):
        sentence = sentence.translate(tbl)  # remove_punctuation(sentence)
        num_words = len(word_tokenize(sentence))
        if num_words <= 5:
            content_sentences_copy[idx] = []
            idxs_blank.append(idx)

    idxs_sentences_to_compare = [x for x in list(range(len(content_sentences_copy))) if x not in idxs_blank]
    cnt = 0
    for idx, i in enumerate(idxs_sentences_to_compare):
        if idx == len(idxs_sentences_to_compare) - 1:
            break
        j = idxs_sentences_to_compare[idx + 1]
        sent1, sent2 = content_sentences_copy[i], content_sentences_copy[j]
        cos_sim = cosine(embeddings[i], embeddings[j])
        if cos_sim < cos_sim_limit:
            new_segment = 1
        else:
            new_segment = 0
        cos_sim_df.loc[cnt] = [sent1, i, sent2, j, cos_sim, new_segment]
        cnt += 1

        # if Info:
        #     print('pair', cnt, '/', len(idxs_sentences_to_compare), 'idxs: ', i, 'vs.', j, '        ', cos_sim,
        #           '      ',
        #           sent1, '======', sent2)

    # Save df to hdf
    cos_sim_df.to_hdf('Saved_dfs/InferSent_cos_sim_df.h5', key='df', mode='w')

    if Info:
        print('-Number of segments: ', cos_sim_df.New_Section.value_counts().loc[1])
        print('-Saved Cos_Sim_df to hd5 file. Preview of cos_sim_df:')
        print(cos_sim_df.head().to_string())

    return cos_sim_df

def SliceCast_Segmentation(content_sentences, doc_labelled=True, Plot=False, save_fig=False):
    """
    SliceCast
    note it is a bit out of date so idk if it'll work in this same python script...
    The joe1254.txt doc has '========,9,title.' tags at manually labelled segments locations

    Note:
        Only importing revelant modules IF this function is called (as opposed to putting them with the rest at top) as
        they are slow to import and throw some tensorflow warnings.

    NOT INPUTTING A LIST OF SENTENCES LIKE I DO IN THE OTHER ONES SO THE IDX OF SENTENCES WHICH START SEGMENTS CAN NOT
    BE DIRECTLY COMPARED TO THE OTHER METHODS
    """
    # Import Modules
    SliceNet = importlib.import_module("SliceCast.src.SliceNet")
    spacyOps = importlib.import_module("SliceCast.src.spacyOps")

    # Choose whether to use the base network or the network with self-attention
    attention = True

    # Current best networks
    best_base_wiki = 'SliceCast/models/04_20_2019_2300_final.h5'
    best_base_podcast = 'SliceCast/models/04_26_2019_1000_podcast.h5'
    best_attn_wiki = 'SliceCast/models/05_03_2019_0800_attn.h5'
    best_attn_podcast = 'SliceCast/models/05_02_2019_2200_attn_podcast.h5'

    if attention:
        weights_wiki = best_attn_wiki
        weights_podcast = best_attn_podcast
    else:
        weights_wiki = best_base_wiki
        weights_podcast = best_base_podcast

    # Instantiate Network
    weights_path = weights_podcast  # Transfer learning
    net = SliceNet.SliceNet(classification=True, class_weights=[1.0, 7, 0.2], attention=attention)
    content_sentences = [sent.replace("= = = = = = = = , 9,title .", "========,9,title.") for sent in content_sentences]

    sents, labels = spacyOps.customLabeler(content_sentences)

    sents = np.array(sents, dtype='object')
    sents = np.expand_dims(sents, axis=0)

    preds = net.singlePredict(sents, weights_path=weights_path)
    preds = list(np.argmax(np.squeeze(preds), axis=-1))

    # Place data into a pandas dataframe for analysis...
    df = pd.DataFrame()

    df['raw_sentences'] = sents[0]
    if doc_labelled:
        df['labels'] = labels
    df['preds'] = preds
    df['sent_number'] = df.index

    # save dataframe of segments to hdf5 file
    df.to_hdf('Saved_dfs/SliceCast_segmented_df.h5', key='dfs', mode='w')
    print('-Saved SliceCast segmentation info to hdf.')

    first_sentence_idxs = list(df[df['preds']==1]['sent_number'].values)

    if Plot:
        Plot_SliceCast(df, save_fig=save_fig)

    return first_sentence_idxs



def Cluster_Transcript(content_sentences):
    """
    NOT USING SO FAR. Taken from Msci project work that Jonas did on 5th January

    Doesn't segment like i hoped it would - just clusters sentences (in no particular order, i.e. not considering
    whether they are consecutive) so could be used for topic detection but not segmentation.
    """
    #load transcript
    print("-Loading sentence embedder, this takes a while...")
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    print("Done")

    #embed using google sentence encoder
    embeddings = embed(content_sentences)
    print('Embedded using Google Sentence Encoder')

    #approach 1: use density based clustering algorithm to cluster sentences into topics
    sim_m = topics.make_similarity_matrix(content_sentences, embeddings)

    cluster_labels = DBSCAN(eps=2.2, min_samples=1).fit_predict(sim_m) #eps is sensitivity

    clusters = defaultdict(list)
    for cl, sentence in zip(cluster_labels, content_sentences):
        clusters[cl].append(sentence)
    print('\n list(clusters.values()): ')
    pprint(list(clusters.values())) #these are sentence clusters
    print('\nNumber of Clusters: ', len(list(clusters.values())))

    #find the sentence index at which the clusters start


    #print('\nnow plotting')
    #topics.plot_similarity(content_sentences, embeddings, 90) #visualised here


## Functions for plotting...

def Plot_SliceCast(segmentation_df, save_fig=False):
    """Function to plot """
    # Plot
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    segmentation_df.plot(x='sent_number', y='preds', figsize=(10, 5), grid=True, ax=axes[0], label='Prediction')
    # axes[0].set_title('SliceCast Predictions')
    segmentation_df.plot(x='sent_number', y='labels', figsize=(10, 5), grid=True, ax=axes[1], color='green', label='Label')
    # axes[1].set_title('Manual Labels')
    axes[1].set_xlabel('Sentence Number')
    fig.suptitle('Topic-Wise Segmentation of Podcast Transcript: SliceCast Predictions vs. Manual Labels')
    if save_fig:
        plt.savefig("Saved_Images/SliceCast_Segmentation.png")
    plt.show()
    return

def Plot_InferSent_Clusters(sentences, cos_sim_df, cos_sim_cutoff, save_fig=False):
    """
    Function to plot the NetworkX graph whose nodes represent sentences and edges are formed if the cosine similarity
    between the sentence embeddings (obtained using InferSent) are > 'cos_sim_cutoff'.
    """
    G = nx.DiGraph()  # Instantiate graph

    # Build graph
    G.add_nodes_from(range(len(sentences)))
    for row in cos_sim_df.itertuples(index=True):
        if row.Cosine_Similarity >= cos_sim_cutoff:
            G.add_edge(row.Sentence1_idx, row.Sentence2_idx, weight=row.Cosine_Similarity)

    # Plot network to see clusters
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw(G, width=weights, with_labels=True, font_weight='bold')
    plt.title('InferSent Clusters with {0} Cosine Similarity Cutoff'.format(cos_sim_cutoff))

    if save_fig:
        plt.savefig("Saved_Images/InferSent_Clusters_{0}_cutoff.png".format(cos_sim_cutoff))

    plt.show()
    return


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
        fig.savefig("Saved_Images/WordCloud.png", dpi=600)
    return

def Plot_Embeddings(keyword_vectors_df, embedding_method, transcript_name, shifted_ngrams=False, save_fig=False):
    """
    Plots the layout of all the keywords from the podcast. Keywords include those extracted using TopicRank,
    all potentially-interesting nouns, and all extracted bigrams and trigrams. Includes colour coordination with respect
    to the type of keywords.
    """
    print('\n-Plotting Word Embedding for all Keywords...')
    keyword_types = ['noun', 'pke', 'bigram', 'trigram']
    colours = ['blue', 'green', 'orange', 'pink']
    labels = ['Nouns', 'PKE Keywords', 'Bigrams', 'Trigrams']

    plt.figure()

    for i in range(len(keyword_types)):
        type = keyword_types[i]
        words = keyword_vectors_df['{}_keyw'.format(type)]
        Xs, Ys = keyword_vectors_df['{}_X'.format(type)], keyword_vectors_df['{}_Y'.format(type)]
        unplotted = list(keyword_vectors_df['unfamiliar_{}'.format(type)].dropna(axis=0))

        plt.scatter(Xs, Ys, c=colours[i], label=labels[i])
        for label, x, y in zip(words, Xs, Ys):
            plt.annotate(label, xy=(x, y), xytext=(-5, 0), textcoords="offset points")
        print('Plotted: ', labels[i])
        print(labels[i], 'which were not plotted due to lack of embedding: ', list(unplotted))

    plt.legend()
    embedding_method = embedding_method.title()
    plt.title('{0} Keywords Embedding'.format(embedding_method))
    if save_fig and not shift_ngrams:
        plt.savefig("Saved_Images/{0}/{1}_WordEmbedding.png".format(transcript_name, embedding_method), dpi=600)
    if save_fig and shift_ngrams:
        plt.savefig("Saved_Images/{0}/{1}_WordEmbedding_ShiftedNgrams.png".format(transcript_name, embedding_method), dpi=600)
    plt.show()

    return



def Plot_2D_Topic_Evolution_SegmentWise(segments_info_df_1, save_name, transcript_name, names, segments_info_df_2=pd.DataFrame(),
                                        Node_Position='total_average', save_fig=False, plot_hist_too=True,
                                        colour_quiver_plots=True, speakerwise_coloring=False):
    """
    Plots the 2D word embedding space with a Quiver arrow following the direction of the topics discussed in each
    segment of the transcript.
    """
    speakerwise_colours = ['b', 'purple']
    spkr_idx = 0
    plt.figure()

    for segments_info_df in [segments_info_df_1, segments_info_df_2]:
        if segments_info_df.empty: #i.e. if not speakerwise.
            continue

        # Quiver part 1
        if Node_Position == 'total_average':
            node_position = segments_info_df['total_average_keywords_wordvec'].values
            labels = range(len(node_position))  # numerical labels

        if Node_Position == '1_max_count':
            labels_text = segments_info_df['top_count_keyword'].values
            node_position = segments_info_df['top_count_wordvec'].values

            # Check the segments all had enough keywords to have taken max(count)...
            node_position = [pos for pos in node_position if pos != None]
            labels_text = [label for label in labels_text if label != None]
            labels = range(len(node_position))

        if Node_Position == '3_max_count':
            labels_text = segments_info_df['top_3_counts_keywords'].values
            node_position = segments_info_df['top_3_counts_wordvec'].values

            # Check the segments all had enough keywords to have taken max(count)...
            node_position = [pos for pos in node_position if str(pos[0]) != 'n']
            labels_text = [label for label in labels_text if str(label) != 'nan']
            labels = range(len(node_position))

        if colour_quiver_plots or plot_hist_too:
            # define colours for the segment
            number_keywords = [sum([int(num) for num in x]) for x in list(segments_info_df['noun_counts'].values)] #careful whether I'm using noun_counts or keyword_counts !!
            number_keywords_sorted = number_keywords.copy() # make a copy that we can rearrange
            number_keywords_sorted.sort()
            groups = list(split(number_keywords_sorted, 3))
            colours = [(255/255, 255/255, 0), (255/255, round(125/255, 5), 0), (round(240/255, 5), 0, 0)]
            colour_info_dict = {k:[v[index] for index in [0, -1]] for k, v in zip(colours, groups)}
            idxs = [next(index for index, sublist in enumerate(groups) if number in sublist) for number in number_keywords]
            colour_for_segment = [colours[i] for i in idxs]
            print('colour and groups:', colour_info_dict)
            save_name = save_name + '_Coloured'


        # if plot_hist_too:
        #     # Plotting Histogram of keyword usage
        #     plt.figure(1)
        #     plt.bar(range(len(number_keywords)), number_keywords, color=colour_for_segment)
        #     limits = list(colour_info_dict.values())
        #     for limit in limits:
        #         lower_lim = limit[0]
        #         plt.plot([0, len(number_keywords)], [lower_lim, lower_lim], '--', color='k', linewidth=1)
        #     legend_elements = [Line2D([0], [0], color=colours[0], lw=1, label='{0} - {1} Keywords'.format(limits[0][0], limits[0][1])),
        #                        Line2D([0], [0], color=colours[1], lw=1, label='{0} - {1} Keywords'.format(limits[1][0], limits[1][1])),
        #                        Line2D([0], [0], color=colours[2], lw=1, label='{0} - {1} Keywords'.format(limits[2][0], limits[2][1]))]
        #     plt.title('Number of Keywords Contained in Each Segment \n(Each segment contains roughly {0} sentences)'.format(segments_info_df['length_of_segment'].values[0]))
        #     plt.ylabel('Number of Keywords')
        #     plt.xlabel('Segment Number')
        #     plt.legend(handles=legend_elements)
        #     if save_fig:
        #         plt.savefig("Saved_Images/{0}/histogram_of_keywords.png".format(transcript_name, save_name), dpi=600)
        #     plt.show()

        if speakerwise_coloring:
            colour_for_segment = speakerwise_colours[spkr_idx]

        else:
            colour_for_segment = 'b'

        xs = [x[0] for x in node_position]
        ys = [x[1] for x in node_position]

        u = [i-j for i, j in zip(xs[1:], xs[:-1])]
        v = [i-j for i, j in zip(ys[1:], ys[:-1])]

        plt.quiver(xs[:-1], ys[:-1], u, v, color=colour_for_segment, scale_units='xy',
                   angles='xy', scale=1, width=0.002)

        # To make sure labels are spread out well, going to mess around with xs and ys
        xs_, ys_ = xs, ys
        ppairs = [(i, j) for i, j in zip(xs_, ys_)]
        repeats = list(set(map(tuple, ppairs)))
        repeat_num = [0 for i in range(len(repeats))]
        plt.rc('font', size=8)
        for x, y, label in zip(xs, ys, labels):
            # first check location of annotation is unique
            if (x, y) in repeats:
                idx = repeats.index((x, y))
                addition = repeat_num[idx]
                x += addition
                y -= 1
                repeat_num[idx] += 3
                pass

            plt.annotate(str(label+1) + '.', # this is the text
                         (x, y), # this is the point to label
                         textcoords="offset points", # how to position the text
                         xytext=(0, 5), # distance from text to points (x,y)
                         ha='center') # horizontal alignment can be left, right or center
        plt.rc('font', size=10)  # putting font back to normal
        #plot special colours for the first and last point
        plt.plot([xs[0]], [ys[0]], 'o', color=colour_for_segment, markersize=14)
        plt.plot([xs[0]], [ys[0]], 'o', color='green', markersize=10)
        plt.plot([xs[-1]], [ys[-1]], 'o', color=colour_for_segment, markersize=14)
        plt.plot([xs[-1]], [ys[-1]], 'o', color='red', markersize=10)

        spkr_idx += 1
    line1 = Line2D(range(1), range(1), color="green", marker='o', markersize=7, linestyle='none')
    line2 = Line2D(range(1), range(1), color="red", marker='o', markersize=7, linestyle='none')
    line3 = Line2D([0], [0], color=speakerwise_colours[0], lw=1),
    line4 = Line2D([0], [0], color=speakerwise_colours[1], lw=1),

    plt.legend((line1, line2, line3, line4), ('Beginning of Conversation', 'End of Conversation',
                                              names[0], names[1]), prop={'size': 7})
    plt.title(' '.join(save_name.split('_')))
    if save_fig:
        plt.savefig("Saved_Images/{0}/{1}.png".format(transcript_name, save_name), dpi=600)
    plt.show()
    return

def Plot_Quiver_And_Embeddings(segments_info_df_1, keyword_vectors_df, transcript_name, save_name, names,
                               segments_info_df_2=pd.DataFrame(), Node_Position='total_average',
                               only_nouns=True, save_fig=False, colour_quiver_plots=True, speakerwise_colouring=False):
    """
    Plots BOTH a background of keywords + the 2D quiver arrow following the direction of the topics discussed in each
    segment of the transcript.
    words_to_highlight_dict.values() [total count of non_unique keyword in nodes_list, normalised count, font size]
    """
    speakerwise_colours = ['cornflowerblue', 'darkorchid'] #['b', 'purple']
    spkr_idx = 0
    plt.figure()
    for segments_info_df in [segments_info_df_1, segments_info_df_2]:
        if segments_info_df.empty: #i.e. if not speakerwise.
            continue
        # QUIVER PART  1
        if Node_Position == 'total_average':
            node_position = segments_info_df['total_average_keywords_wordvec'].values

        if Node_Position == '1_max_count':
            labels_text = segments_info_df['top_count_keyword'].values
            node_position = segments_info_df['top_count_wordvec'].values

            # Check the segments all had enough keywords to have taken max(count)...
            node_position = [pos for pos in node_position if pos != None]
            labels_text = [label for label in labels_text if label != None]

            # now check for any keywords that have >1 connection.. clusters!
            D = defaultdict(list)
            for i, item in enumerate(labels_text):
                D[item].append(i)

            D = {k: len(v) for k, v in D.items() if len(v) > 1}
            max_count = max(D.values())
            print('max count', max_count)

            # Normalise counts
            words_to_highlight_dict = {k: [v,  v / max_count] for k, v in D.items()}

            # Decide on font
            fonts_dict = {10: [0, 0.2], 11: [0.2, 0.4], 12: [0.4, 0.6], 13: [0.6, 0.8], 14: [0.8, 10]}

            fonts = []
            for v_norm in np.array(list(words_to_highlight_dict.values()))[:, 1]:
                for key, list_norms in fonts_dict.items():
                    if float(v_norm) >= float(list_norms[0]) and float(v_norm) <= float(list_norms[1]):
                        fonts.append(key)
                    else:
                        continue

            for idx, key in enumerate(words_to_highlight_dict.keys()):
                words_to_highlight_dict[key].append(fonts[idx])
            print('words_to_highlight_dict', words_to_highlight_dict)


        # EMBEDDING PART
        keyword_types = ['noun', 'pke', 'bigram', 'trigram']
        colours = ['pink', 'green', 'orange', 'blue']
        labels = ['Nouns', 'PKE Keywords', 'Bigrams', 'Trigrams']

        number_types_toplot = range(len(keyword_types))
        if only_nouns:
            number_types_toplot = [0]

        plt.rc('font', size=6)
        for i in number_types_toplot:
            type = keyword_types[i]
            words = keyword_vectors_df['{}_keyw'.format(type)]
            Xs, Ys = keyword_vectors_df['{}_X'.format(type)], keyword_vectors_df['{}_Y'.format(type)]
            unplotted = list(keyword_vectors_df['unfamiliar_{}'.format(type)].dropna(axis=0))

            plt.scatter(Xs, Ys, c=colours[i], label=labels[i], zorder=0, alpha=0.4)
            for label, x, y in zip(words, Xs, Ys):

                # First check if label is in word_to_highlight (i.e. if it has a cluster of quiver arrow heads)
                if label in words_to_highlight_dict.keys():
                    font = words_to_highlight_dict[label][2]
                    plt.rc('font', size=font)
                    clr = 'k'
                    zord = 100
                    wght = 'heavy'
                    xypos = (-8, -10)
                    # label = label.title()
                elif label in labels_text:
                    plt.rc('font', size=8)
                    clr = 'k'
                    zord = 95
                    wght = 'bold'
                    xypos = (-5, -5)

                else:
                    continue
                    # plt.rc('font', size=6)
                    # clr = 'darkgrey'
                    # zord = 5
                    # wght = 'normal'
                    # xypos = (-5, 0)

                plt.annotate(label, xy=(x, y), xytext=xypos, textcoords="offset points", color=clr, zorder=zord, weight=wght)

        # Back to QUIVER part
        labels = range(len(node_position))

        if colour_quiver_plots:
            # define colours for the segment
            number_keywords = [sum([int(num) for num in x]) for x in list(
                segments_info_df['noun_counts'].values)]  # careful whether I'm using noun_counts or keyword_counts !!
            number_keywords_sorted = number_keywords.copy()  # make a copy that we can rearrange
            number_keywords_sorted.sort()
            groups = list(split(number_keywords_sorted, 3))
            colours = [(255 / 255, 255 / 255, 0), (255 / 255, 125 / 255, 0), (240 / 255, 0, 0)]
            colour_info_dict = {k: [v[index] for index in [0, -1]] for k, v in zip(colours, groups)}
            idxs = [next(index for index, sublist in enumerate(groups) if number in sublist) for number in number_keywords]
            colour_for_segment = [colours[i] for i in idxs]
            print('colour and groups:', colour_info_dict)
            save_name = save_name +'_Coloured'

        if not colour_quiver_plots:
            colour_for_segment = 'b'

        if speakerwise_colouring:
            colour_for_segment = speakerwise_colours[spkr_idx]


        xs = [x[0] for x in node_position]
        ys = [x[1] for x in node_position]

        u = [i-j for i, j in zip(xs[1:], xs[:-1])]
        v = [i-j for i, j in zip(ys[1:], ys[:-1])]

        # To make sure labels are spread out well, going to mess around with xs and ys
        xs_, ys_ = xs, ys
        ppairs = [(i, j) for i, j in zip(xs_, ys_)]
        repeats = list(set(map(tuple, ppairs)))
        repeat_num = [0 for i in range(len(repeats))]
        # plt.rc('font', size=8)  # putting font back to normal
        for x, y, label in zip(xs, ys, labels):
            # first check location of annotation is unique - if not, update location for the sentence number
            if (x, y) in repeats:
                idx = repeats.index((x, y))
                addition = repeat_num[idx]
                x += addition
                y -= 1
                repeat_num[idx] += 3
                pass

            plt.annotate(str(label + 1) + '.',  # this is the text
                         (x, y),  # this is the point to label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 5),  # distance from text to points (x,y)
                         ha='center',
                         zorder=100)  # horizontal alignment can be left, right or center

        # Make Quiver Line very thin if using a large number of segments
        num_segs = len(colour_for_segment)
        if num_segs == 200:
            line_wdth = 0.001
        else:
            line_wdth = 0.002

        plt.rc('font', size=10)  # putting font back to normal

        plt.quiver(xs[:-1], ys[:-1], u, v, scale_units='xy',
                   angles='xy', scale=1, color=colour_for_segment, width=line_wdth, zorder=6)

        #plot special colours for the first and last point
        if speakerwise_colouring:
            plt.plot([xs[0]], [ys[0]], 'o', color=colour_for_segment, markersize=11)
        plt.plot([xs[0]], [ys[0]], 'o', color='green', markersize=8, zorder=20)
        if speakerwise_colouring:
            plt.plot([xs[-1]], [ys[-1]], 'o', color=colour_for_segment, markersize=11)
        plt.plot([xs[-1]], [ys[-1]], 'o', color='red', markersize=8, zorder=20)

        spkr_idx += 1

    line1 = Line2D(range(1), range(1), color="green", marker='o', markersize=7, linestyle='none')
    line2 = Line2D(range(1), range(1), color="red", marker='o', markersize=7, linestyle='none')
    if speakerwise_colouring:
        line3 = Line2D([0], [0], color=speakerwise_colours[0], lw=1),
        line4 = Line2D([0], [0], color=speakerwise_colours[1], lw=1),

        plt.legend((line1, line2, line3, line4), ('Beginning of Conversation', 'End of Conversation',
                                                  names[0], names[1]), prop={'size': 7})
    else:
        plt.legend((line1, line2), ('Beginning of Conversation', 'End of Conversation'))

    plt.title(' '.join(save_name.split('_')))
    if save_fig:
        plt.savefig("Saved_Images/{0}/{1}.png".format(transcript_name, save_name), dpi=600)
    plt.show()

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

def Plot_3D_Trajectory_through_TopicSpace(segments_info_df_1, keyword_vectors_df, save_name, transcript_name, names,
                                          segments_info_df_2=pd.DataFrame(),
                                          Node_Position='total_average', only_nouns=True, save_fig=False,
                                          speakerwise_colouring=False):
    """
    Note updated yet.
    Taken from my messy code in Inference. Here ready for when I have segmentation info from Jonas' method.
    """
    speakerwise_colours = ['cornflowerblue', 'darkorchid']
    spkr_idx = 0
    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=(22, 11))  # figsize=(22, 11)
    fig.suptitle('Movement of Conversation through Topic Space over Time')

    # set up the axes for the first plot
    ax1 = fig.add_subplot(111, projection='3d')  # ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    for segments_info_df in [segments_info_df_1, segments_info_df_2]:
        if segments_info_df.empty: #i.e. if not speakerwise.
            continue
        if Node_Position == 'total_average':
            node_position = segments_info_df['total_average_keywords_wordvec'].values
            first_sents = segments_info_df['first_sent_numbers'].values

        if Node_Position == '1_max_count':
            labels_text = segments_info_df['top_count_keyword'].values
            node_position = segments_info_df['top_count_wordvec'].values
            first_sents = segments_info_df['first_sent_numbers'].values

            # Check the segments all had enough keywords to have taken max(count)...
            first_sents = [i for idx, i in enumerate(first_sents) if node_position[idx]!=None]
            node_position = [pos for pos in node_position if pos != None]
            labels_text = [label for label in labels_text if label != None]

            # now check for any keywords that have >1 connection.. clusters!
            D = defaultdict(list)
            for i, item in enumerate(labels_text):
                D[item].append(i)

            D = {k: len(v) for k, v in D.items() if len(v) > 1}
            max_count = max(D.values())
            print('max count', max_count)

            # Normalise counts
            words_to_highlight_dict = {k: [v, v / max_count] for k, v in D.items()}

            # Decide on font
            fonts_dict = {9: [0, 0.2], 10: [0.2, 0.4], 11: [0.4, 0.6], 12: [0.6, 0.8], 12: [0.8, 10]}

            fonts = []
            for v_norm in np.array(list(words_to_highlight_dict.values()))[:, 1]:
                for key, list_norms in fonts_dict.items():
                    if float(v_norm) >= float(list_norms[0]) and float(v_norm) <= float(list_norms[1]):
                        fonts.append(key)
                    else:
                        continue

            for idx, key in enumerate(words_to_highlight_dict.keys()):
                words_to_highlight_dict[key].append(fonts[idx])
            print('words_to_highlight_dict', words_to_highlight_dict)

        if Node_Position == '3_max_count':
            labels_text = segments_info_df['top_3_counts_keywords'].values
            node_position = segments_info_df['top_3_counts_wordvec'].values
            first_sents = segments_info_df['first_sent_numbers'].values

            # Check the segments all had enough keywords to have taken max(count)...
            first_sents = [i for idx, i in enumerate(first_sents) if node_position[idx]!=None]
            node_position = [pos for pos in node_position if str(pos[0]) != 'n']
            labels_text = [label for label in labels_text if str(label) != 'nan']

        # EMBEDDING PART
        keyword_types = ['noun', 'pke', 'bigram', 'trigram']
        colours = ['pink', 'green', 'orange', 'blue']
        labels = ['Nouns', 'PKE Keywords', 'Bigrams', 'Trigrams']

        number_types_toplot = range(len(keyword_types))
        if only_nouns:
            number_types_toplot = [0]

        plt.rc('font', size=6)
        for i in number_types_toplot:
            type = keyword_types[i]
            words = keyword_vectors_df['{}_keyw'.format(type)]
            Xs, Ys = keyword_vectors_df['{}_X'.format(type)], keyword_vectors_df['{}_Y'.format(type)]
            unplotted = list(keyword_vectors_df['unfamiliar_{}'.format(type)].dropna(axis=0))

            ax1.scatter([0 for i in range(len(Xs))], Xs, Ys, c=colours[i], label=labels[i])

            for label, x, y in zip(words, Xs, Ys):

                # First check if label is in word_to_highlight (i.e. if it has a cluster of quiver arrow heads)
                if label in words_to_highlight_dict.keys():
                    font = words_to_highlight_dict[label][2]
                    plt.rc('font', size=font)
                    clr = 'k'
                    zord = 100
                    wght = 'heavy'
                    xypos = (-8, -10)
                    # label = label.title()
                elif label in labels_text:
                    plt.rc('font', size=8)
                    clr = 'k'
                    zord = 90
                    wght = 'normal'
                    xypos = (-5, -5)

                else:
                    continue
                    # plt.rc('font', size=6)
                    # clr = 'darkgrey'
                    # zord = 5
                    # wght = 'normal'
                    # xypos = (-5, 0)


                #plt.annotate(label, xy=(x, y), xytext=xypos, textcoords="offset points", color=clr, zorder=zord, weight=wght)
                ax1.text(0, x+xypos[0], y+xypos[1], label, size=10, color=clr, zorder=zord, weight=wght, zdir='x')

        labels = range(len(node_position))

        # Data for a three-dimensional line
        xs = [x[0] for x in node_position]
        ys = [x[1] for x in node_position]

        #ax.plot3D(xs, segment_numbers, ys, 'bo-')
        ax1.set_xlabel('$Sentence Number$', fontsize=13)
        ax1.set_ylabel('$X$', fontsize=20, rotation = 0)
        ax1.set_zlabel('$Y$', fontsize=20)
        ax1.zaxis.set_rotate_label(False)
        #ax1.set_title('Manual')

        # To make sure labels are spread out well, going to mess around with xs and ys
        xs_, ys_ = xs, ys
        ppairs = [(i, j) for i, j in zip(xs_, ys_)]
        repeats = list(set(map(tuple, ppairs)))
        repeat_num = [0 for i in range(len(repeats))]

        cnt = 0
        if not speakerwise_colouring:
            speaker_colour = 'b'
        if speakerwise_colouring:
            speaker_colour = speakerwise_colours[spkr_idx]
        # (old_x, old_y, old_z) = (0, 0, 0)

        #Plot Embedding on Sentence_Number = 0 plane

        for x, y, z, label in zip(first_sents, xs, ys, labels):
              cnt +=1
              ax1.plot([x], [y], [z],'o', markersize=2, alpha=0.2) #markerfacecolor='k', markeredgecolor='k', marker='o', markersize=5, alpha=0.6) # MAYBE REMOVE node so can clear up appearance

              # first check location of annotation is unique - if not, update location for the sentence number
              if (y, z) in repeats:
                  idx = repeats.index((y, z))
                  addition = repeat_num[idx]
                  y_txt = y + addition
                  z_txt = z - 1
                  repeat_num[idx] += 3
                  pass
              else:
                  y_txt = y
                  z_txt = z

              ax1.text(x, y_txt, z_txt, label+1, size=10)
              if cnt ==1:
                (old_x, old_y, old_z) = (x, y, z)
                continue


              a = Arrow3D([old_x, x], [old_y, y], [old_z, z], mutation_scale=10, lw=1, arrowstyle="-|>", color=speaker_colour)
              ax1.add_artist(a)

              (old_x, old_y, old_z) = (x, y, z)

        spkr_idx += 1

    if speakerwise_colouring:
        # line1 = Line2D(range(1), range(1), color="green", marker='o', markersize=7, linestyle='none')
        # line2 = Line2D(range(1), range(1), color="red", marker='o', markersize=7, linestyle='none')
        line3 = Line2D([0], [0], color=speakerwise_colours[0], lw=1),
        line4 = Line2D([0], [0], color=speakerwise_colours[1], lw=1),

        plt.legend((line3, line4), (names[0], names[1]), prop={'size': 7})

    # AXIS STUFF
    ax1.dist = 13
    ax1 = plt.gca()
    # ax.xaxis.set_ticklabels([])
    ax1.yaxis.set_ticklabels([])
    ax1.zaxis.set_ticklabels([])

    # for line in ax.xaxis.get_ticklines():
    #     line.set_visible(False)
    for line in ax1.yaxis.get_ticklines():
        line.set_visible(False)
    for line in ax1.zaxis.get_ticklines():
        line.set_visible(False)

    if save_fig:
        plt.savefig("Saved_Images/{0}/{1}.png".format(transcript_name, save_name), dpi=600)

    plt.show()
    return


def Animate(segments_info_df_1, keyword_vectors_df, transcript_name, save_name, names,
                                   segments_info_df_2=pd.DataFrame(), Node_Position='total_average',
                                   only_nouns=True, save_fig=False, colour_quiver_plots=True,
                                   speakerwise_colouring=False):
    """
    Function to create Animation of topical evolution through word embedding space

    Plots BOTH a background of keywords + the 2D quiver arrow following the direction of the topics discussed in each
    segment of the transcript.
    words_to_highlight_dict.values() [total count of non_unique keyword in nodes_list, normalised count, font size]
    """
    from matplotlib.animation import FuncAnimation
    from matplotlib import rcParams

    speakerwise_colours = ['cornflowerblue', 'darkorchid']  # ['b', 'purple']
    spkr_idx = 0
    # plt.figure()
    fig, ax = plt.subplots()
    for segments_info_df in [segments_info_df_1, segments_info_df_2]:
        if segments_info_df.empty:  # i.e. if not speakerwise.
            continue
        # QUIVER PART  1
        if Node_Position == 'total_average':
            node_position = segments_info_df['total_average_keywords_wordvec'].values

        if Node_Position == '1_max_count':
            labels_text = segments_info_df['top_count_keyword'].values
            node_position = segments_info_df['top_count_wordvec'].values

            # Check the segments all had enough keywords to have taken max(count)...
            node_position = [pos for pos in node_position if pos != None]
            labels_text = [label for label in labels_text if label != None]

            # now check for any keywords that have >1 connection.. clusters!
            D = defaultdict(list)
            for i, item in enumerate(labels_text):
                D[item].append(i)

            D = {k: len(v) for k, v in D.items() if len(v) > 1}
            max_count = max(D.values())
            print('max count', max_count)

            # Normalise counts
            words_to_highlight_dict = {k: [v, v / max_count] for k, v in D.items()}

            # Decide on font
            fonts_dict = {10: [0, 0.2], 11: [0.2, 0.4], 12: [0.4, 0.6], 13: [0.6, 0.8], 14: [0.8, 10]}

            fonts = []
            for v_norm in np.array(list(words_to_highlight_dict.values()))[:, 1]:
                for key, list_norms in fonts_dict.items():
                    if float(v_norm) >= float(list_norms[0]) and float(v_norm) <= float(list_norms[1]):
                        fonts.append(key)
                    else:
                        continue

            for idx, key in enumerate(words_to_highlight_dict.keys()):
                words_to_highlight_dict[key].append(fonts[idx])
            print('words_to_highlight_dict', words_to_highlight_dict)

        # EMBEDDING PART
        keyword_types = ['noun', 'pke', 'bigram', 'trigram']
        colours = ['pink', 'green', 'orange', 'blue']
        labels = ['Nouns', 'PKE Keywords', 'Bigrams', 'Trigrams']

        number_types_toplot = range(len(keyword_types))
        if only_nouns:
            number_types_toplot = [0]

        plt.rc('font', size=6)
        for i in number_types_toplot:
            type = keyword_types[i]
            words = keyword_vectors_df['{}_keyw'.format(type)]
            Xs, Ys = keyword_vectors_df['{}_X'.format(type)], keyword_vectors_df['{}_Y'.format(type)]
            unplotted = list(keyword_vectors_df['unfamiliar_{}'.format(type)].dropna(axis=0))

            plt.scatter(Xs, Ys, c=colours[i], label=labels[i], zorder=0, alpha=0.4)
            for label, x, y in zip(words, Xs, Ys):

                # First check if label is in word_to_highlight (i.e. if it has a cluster of quiver arrow heads)
                if label in words_to_highlight_dict.keys():
                    font = words_to_highlight_dict[label][2]
                    plt.rc('font', size=font)
                    clr = 'k'
                    zord = 100
                    wght = 'heavy'
                    xypos = (-8, -10)
                    # label = label.title()
                elif label in labels_text:
                    plt.rc('font', size=8)
                    clr = 'k'
                    zord = 95
                    wght = 'bold'
                    xypos = (-5, -5)

                else:
                    continue
                    # plt.rc('font', size=6)
                    # clr = 'darkgrey'
                    # zord = 5
                    # wght = 'normal'
                    # xypos = (-5, 0)

                plt.annotate(label, xy=(x, y), xytext=xypos, textcoords="offset points", color=clr, zorder=zord,
                             weight=wght)

        # Back to QUIVER part
        labels = range(len(node_position))

        if colour_quiver_plots:
            # define colours for the segment
            number_keywords = [sum([int(num) for num in x]) for x in list(
                segments_info_df[
                    'noun_counts'].values)]  # careful whether I'm using noun_counts or keyword_counts !!
            number_keywords_sorted = number_keywords.copy()  # make a copy that we can rearrange
            number_keywords_sorted.sort()
            groups = list(split(number_keywords_sorted, 3))
            colours = [(255 / 255, 255 / 255, 0), (255 / 255, 125 / 255, 0), (240 / 255, 0, 0)]
            colour_info_dict = {k: [v[index] for index in [0, -1]] for k, v in zip(colours, groups)}
            idxs = [next(index for index, sublist in enumerate(groups) if number in sublist) for number in
                    number_keywords]
            colour_for_segment = [colours[i] for i in idxs]
            print('colour and groups:', colour_info_dict)
            save_name = save_name + '_Coloured'

        if not colour_quiver_plots:
            colour_for_segment = 'b'

        if speakerwise_colouring:
            colour_for_segment = speakerwise_colours[spkr_idx]

        xs = [x[0] for x in node_position]
        ys = [x[1] for x in node_position]

        u = [i - j for i, j in zip(xs[1:], xs[:-1])]
        v = [i - j for i, j in zip(ys[1:], ys[:-1])]

        # To make sure labels are spread out well, going to mess around with xs and ys
        xs_, ys_ = xs, ys
        ppairs = [(i, j) for i, j in zip(xs_, ys_)]
        repeats = list(set(map(tuple, ppairs)))
        repeat_num = [0 for i in range(len(repeats))]
        # plt.rc('font', size=8)  # putting font back to normal
        for x, y, label in zip(xs, ys, labels):
            # first check location of annotation is unique - if not, update location for the sentence number
            if (x, y) in repeats:
                idx = repeats.index((x, y))
                addition = repeat_num[idx]
                x += addition
                y -= 1
                repeat_num[idx] += 3
                pass

            plt.annotate(str(label + 1) + '.',  # this is the text
                         (x, y),  # this is the point to label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 5),  # distance from text to points (x,y)
                         ha='center',
                         zorder=100)  # horizontal alignment can be left, right or center

        # Make Quiver Line very thin if using a large number of segments
        num_segs = len(colour_for_segment)
        if num_segs == 200:
            line_wdth = 0.001
        else:
            line_wdth = 0.002

        plt.rc('font', size=10)  # putting font back to normal

        # plt.quiver(xs[:-1], ys[:-1], u, v, scale_units='xy',
        #            angles='xy', scale=1, color=colour_for_segment, width=line_wdth, zorder=6)

        # -- Quiver -- #


        #players = ax.quiver(positions[0][:, 0], positions[0][:, 1], np.cos(angles[0]), np.sin(angles[0]), scale=20)
        players = ax.quiver(xs[0], ys[0], u[0], v[0]) #, scale_units='xy', angles='xy', scale=1, color='k', width=line_wdth, zorder=6)

        # def animate(idx_timestamp):
        #     print('idx_timestamp', idx_timestamp)
        #     print('xs[idx_timestamp], ys[idx_timestamp]', xs[idx_timestamp], ys[idx_timestamp])
        #     players.set_offsets([xs[idx_timestamp], ys[idx_timestamp]]) #positions[idx_timestamp])
        #     players.set_UVC(u[idx_timestamp], v[idx_timestamp])
        #         return (players)

        def animate(i):
            if i==0:
                ax.plot([xs[0]], [ys[0]], 'o', color='green', markersize=8, zorder=20)
            #ax.cla()  # clear the previous image
            print(colour_for_segment[i])
            ax.plot(xs[i-2:i], ys[i-2:i], '->', color=colour_for_segment[i])#colour_for_segment[i])  # plot the line
            # ax.set_xlim([x0, tfinal])  # fix the x axis
            # ax.set_ylim([1.1 * np.min(y), 1.1 * np.max(y)])  # fix the y axis
            if i== len(xs)-1:
                ax.plot([xs[-1]], [ys[-1]], 'o', color='red', markersize=8, zorder=20)

        anim = FuncAnimation(fig, animate, frames=len(xs)-1, interval=2, repeat=False) #1000 ms delay

        # configure full path for ImageMagick
        rcParams['animation.convert_path'] = r'/Users/ShonaCW/Downloads/ImageMagick-7.0.10/bin/convert'

        # anim.save("Animations/example.mp4", fps=1, dpi=150)
        # save animation at 30 frames per second
        anim.save('Animations/myAnimation.gif', writer='imagemagick', fps=5)
        plt.close()

        # # plot special colours for the first and last point
        # if speakerwise_colouring:
        #     plt.plot([xs[0]], [ys[0]], 'o', color=colour_for_segment, markersize=11)
        # plt.plot([xs[0]], [ys[0]], 'o', color='green', markersize=8, zorder=20)
        # if speakerwise_colouring:
        #     plt.plot([xs[-1]], [ys[-1]], 'o', color=colour_for_segment, markersize=11)
        # plt.plot([xs[-1]], [ys[-1]], 'o', color='red', markersize=8, zorder=20)

        spkr_idx += 1

    # line1 = Line2D(range(1), range(1), color="green", marker='o', markersize=7, linestyle='none')
    # line2 = Line2D(range(1), range(1), color="red", marker='o', markersize=7, linestyle='none')
    # if speakerwise_colouring:
    #     line3 = Line2D([0], [0], color=speakerwise_colours[0], lw=1),
    #     line4 = Line2D([0], [0], color=speakerwise_colours[1], lw=1),
    #
    #     plt.legend((line1, line2, line3, line4), ('Beginning of Conversation', 'End of Conversation',
    #                                               names[0], names[1]), prop={'size': 7})
    # else:
    #     plt.legend((line1, line2), ('Beginning of Conversation', 'End of Conversation'))

    # plt.title(' '.join(save_name.split('_')))
    # if save_fig:
    #     plt.savefig("Saved_Images/{0}/{1}.png".format(transcript_name, save_name), dpi=600)
    # plt.show()

    return

def Animate_3D(segments_info_df_1, keyword_vectors_df, save_name, transcript_name, names,
                                          segments_info_df_2=pd.DataFrame(),
                                          Node_Position='total_average', only_nouns=True, save_fig=False,
                                          speakerwise_colouring=False):
    """
    Note updated yet.
    Taken from my messy code in Inference. Here ready for when I have segmentation info from Jonas' method.
    """
    from matplotlib.animation import FuncAnimation
    from matplotlib import rcParams

    speakerwise_colours = ['cornflowerblue', 'darkorchid']
    spkr_idx = 0
    # set up a figure twice as wide as it is tall
    fig = plt.figure()  # figsize=(22, 11)
    fig.suptitle('Movement of Conversation through Topic Space over Time')

    # set up the axes for the first plot
    ax1 = fig.add_subplot(111, projection='3d')  # ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    dict_list = []
    first_sents_list = []
    node_pos_list = []
    labels_list = []
    for segments_info_df in [segments_info_df_1, segments_info_df_2]:
        if segments_info_df.empty: #i.e. if not speakerwise.
            continue

        print(segments_info_df.head().to_string())
        if Node_Position == 'total_average':
            node_position = segments_info_df['total_average_keywords_wordvec'].values
            first_sents = segments_info_df['first_sent_numbers'].values

        if Node_Position == '1_max_count':
            labels_text = segments_info_df['top_count_keyword'].values
            node_position = segments_info_df['top_count_wordvec'].values
            first_sents = segments_info_df['first_sent_numbers'].values

            # Check the segments all had enough keywords to have taken max(count)...
            first_sents = [i for idx, i in enumerate(first_sents) if node_position[idx]!=None]
            node_position = [pos for pos in node_position if pos != None]
            labels_text = [label for label in labels_text if label != None]
            first_sents_list.append(first_sents)
            node_pos_list.append(node_position)
            labels_list.append(labels_text)

            # now check for any keywords that have >1 connection.. clusters!
            D = defaultdict(list)
            for i, item in enumerate(labels_text):
                D[item].append(i)
            print('D', D) #are these the indices of the node pos for words

            D = {k: len(v) for k, v in D.items() if len(v) > 1}
            print('D', D) #and now just a count of the indices ============== WANT TO PLOT DASHED LINE BETWEEN THEM IN 3D

            max_count = max(D.values())
            print('max count', max_count)

            # Normalise counts
            words_to_highlight_dict = {k: [v, v / max_count] for k, v in D.items()}

            # Decide on font
            fonts_dict = {9: [0, 0.2], 10: [0.2, 0.4], 11: [0.4, 0.6], 12: [0.6, 0.8], 12: [0.8, 10]}

            fonts = []
            for v_norm in np.array(list(words_to_highlight_dict.values()))[:, 1]:
                for key, list_norms in fonts_dict.items():
                    if float(v_norm) >= float(list_norms[0]) and float(v_norm) <= float(list_norms[1]):
                        fonts.append(key)
                    else:
                        continue

            for idx, key in enumerate(words_to_highlight_dict.keys()):
                words_to_highlight_dict[key].append(fonts[idx])
            print('words_to_highlight_dict', words_to_highlight_dict)
            dict_list.append(words_to_highlight_dict)

        if Node_Position == '3_max_count':
            labels_text = segments_info_df['top_3_counts_keywords'].values
            node_position = segments_info_df['top_3_counts_wordvec'].values
            first_sents = segments_info_df['first_sent_numbers'].values

            # Check the segments all had enough keywords to have taken max(count)...
            first_sents = [i for idx, i in enumerate(first_sents) if node_position[idx]!=None]
            node_position = [pos for pos in node_position if str(pos[0]) != 'n']
            labels_text = [label for label in labels_text if str(label) != 'nan']

    # EMBEDDING PART
    keyword_types = ['noun', 'pke', 'bigram', 'trigram']
    colours = ['pink', 'green', 'orange', 'blue']
    labels = ['Nouns', 'PKE Keywords', 'Bigrams', 'Trigrams']

    number_types_toplot = range(len(keyword_types))
    if only_nouns:
        number_types_toplot = [0]

    plt.rc('font', size=6)
    # for words_to_highlight_dict in dict_list:
    #     type = 'noun'
    #     words = keyword_vectors_df['{}_keyw'.format(type)]
    #     Xs, Ys = keyword_vectors_df['{}_X'.format(type)], keyword_vectors_df['{}_Y'.format(type)]
    #     unplotted = list(keyword_vectors_df['unfamiliar_{}'.format(type)].dropna(axis=0))
    #
    #     ax1.scatter([0 for i in range(len(Xs))], Xs, Ys, c=colours[i], label=labels[i])
    #
    #     for label, x, y in zip(words, Xs, Ys):
    #
    #         # First check if label is in word_to_highlight (i.e. if it has a cluster of quiver arrow heads)
    #         if label in words_to_highlight_dict.keys():
    #             font = words_to_highlight_dict[label][2]
    #             plt.rc('font', size=font)
    #             clr = 'k'
    #             zord = 100
    #             wght = 'heavy'
    #             xypos = (-8, -10)
    #             # label = label.title()
    #         elif label in labels_text:
    #             plt.rc('font', size=8)
    #             clr = 'k'
    #             zord = 90
    #             wght = 'normal'
    #             xypos = (-5, -5)
    #
    #         else:
    #             continue
    #             # plt.rc('font', size=6)
    #             # clr = 'darkgrey'
    #             # zord = 5
    #             # wght = 'normal'
    #             # xypos = (-5, 0)
    #
    #
    #         #plt.annotate(label, xy=(x, y), xytext=xypos, textcoords="offset points", color=clr, zorder=zord, weight=wght)
    #         ax1.text(0, x+xypos[0], y+xypos[1], label, size=10, color=clr, zorder=zord, weight=wght, zdir='x') labels_list

    labels = range(len(node_position))

    # Data for a three-dimensional line
    xs_1 = [x[0] for x in node_pos_list[0]]
    ys_1 = [x[1] for x in node_pos_list[0]]

    xs_2 = [x[0] for x in node_pos_list[1]]
    ys_2 = [x[1] for x in node_pos_list[1]]

    # w = [i - j for i, j in zip(labels[1:], labels[:-1])]
    # u = [i - j for i, j in zip(xs[1:], xs[:-1])]
    # v = [i - j for i, j in zip(ys[1:], ys[:-1])]

    #ax.plot3D(xs, segment_numbers, ys, 'bo-')
    ax1.set_xlabel('$Sentence Number$', fontsize=13)
    ax1.set_ylabel('$X$', fontsize=20, rotation=0)
    ax1.set_zlabel('$Y$', fontsize=20)
    ax1.zaxis.set_rotate_label(False)
    #ax1.set_title('Manual')

    # To make sure labels are spread out well, going to mess around with xs and ys
    # xs_, ys_ = xs, ys
    # ppairs = [(i, j) for i, j in zip(xs_, ys_)]
    # repeats = list(set(map(tuple, ppairs)))
    # repeat_num = [0 for i in range(len(repeats))]

    cnt = 0
    if not speakerwise_colouring:
        speaker_colour = 'b'
    if speakerwise_colouring:
        speaker_colour = speakerwise_colours[spkr_idx]
    # (old_x, old_y, old_z) = (0, 0, 0)

    #Plot Embedding on Sentence_Number = 0 plane

    # for x, y, z, label in zip(first_sents, xs, ys, labels):
    #       cnt +=1
    #       ax1.plot([x], [y], [z],'o', markersize=2, alpha=0.2) #markerfacecolor='k', markeredgecolor='k', marker='o', markersize=5, alpha=0.6) # MAYBE REMOVE node so can clear up appearance
    #
    #       # first check location of annotation is unique - if not, update location for the sentence number
    #       if (y, z) in repeats:
    #           idx = repeats.index((y, z))
    #           addition = repeat_num[idx]
    #           y_txt = y + addition
    #           z_txt = z - 1
    #           repeat_num[idx] += 3
    #           pass
    #       else:
    #           y_txt = y
    #           z_txt = z
    #
    #       ax1.text(x, y_txt, z_txt, label+1, size=10)
    #       if cnt ==1:
    #         (old_x, old_y, old_z) = (x, y, z)
    #         continue
    #
    #
    #       a = Arrow3D([old_x, x], [old_y, y], [old_z, z], mutation_scale=10, lw=1, arrowstyle="-|>", color=speaker_colour)
    #       ax1.add_artist(a)
    #
    #       (old_x, old_y, old_z) = (x, y, z)
    #
    # spkr_idx += 1

    def animate(i):
        # x, y, z = first_sents[i], xs[i], ys[i]
        # ax.plot([x], [y], [z], 'o', markersize=2,
        #         alpha=0.2)  # markerfacecolor='k', markeredgecolor='k', marker='o', markersize=5, alpha=0.6) # MAYBE REMOVE node so can clear up appearance
        #
        # # first check location of annotation is unique - if not, update location for the sentence number
        # if (y, z) in repeats:
        #       idx = repeats.index((y, z))
        #       addition = repeat_num[idx]
        #       y_txt = y + addition
        #       z_txt = z - 1
        #       repeat_num[idx] += 3
        #       pass
        # else:
        #       y_txt = y
        #       z_txt = z
        #
        # ax1.text(x, y_txt, z_txt, label+1, size=10)
        # if cnt ==1:
        #     (old_x, old_y, old_z) = (x, y, z)
        #     continue

        # if i == 0:
        #     ax1.plot(first_sents[0], xs[0], ys[0], 'o', color='green', markersize=8, zorder=20)
        # # # ax.cla()  # clear the previous image
        # print(colour_for_segment[i])
        # print('i', i)
        # print('first_sents[:i], xs[:i], ys[:i]', first_sents[:i], xs[:i], ys[:i])
        # if i>0:
        #     a = Arrow3D(first_sents[:i], xs[:i], ys[:i], mutation_scale=10, lw=1, arrowstyle="-|>", color=speaker_colour)
        #     ax1.add_artist(a)

        #ax1.text(0, x + xypos[0], y + xypos[1], label, size=10, color=clr, zorder=zord, weight=wght, zdir='x')
        ax1.text(first_sents_list[0][i], xs_1[i], ys_1[i], labels_list[0][i], size=7, color='k', zorder=100, zdir='x')
        ax1.plot(first_sents_list[0][i - 2:i], xs_1[i - 2:i], ys_1[i - 2:i], color=speakerwise_colours[0], lw=1)  # colour_for_segment[i])  # plot the line
        ax1.plot(first_sents_list[1][i - 2:i], xs_2[i - 2:i], ys_2[i - 2:i], color=speakerwise_colours[1], lw=1)  # colour_for_segment[i])  # plot the line
        ax1.text(first_sents_list[1][i], xs_2[i], ys_2[i], labels_list[1][i], size=7, color='k', zorder=100, zdir='x')
        #ax1.set_xlim([first_sents_list[0][0], first_sents_list[0][-1]])  # fix the x axis
        # ax.set_ylim([1.1 * np.min(y), 1.1 * np.max(y)])  # fix the y axis
        # if i == len(xs) - 1:
        #     ax.plot([xs[-1]], [ys[-1]], 'o', color='red', markersize=8, zorder=20)

    anim = FuncAnimation(fig, animate, frames=len(xs_1)-1, interval=1, repeat=False)  # 1000 ms delay

    # configure full path for ImageMagick
    rcParams['animation.convert_path'] = r'/Users/ShonaCW/Downloads/ImageMagick-7.0.10/bin/convert'

    #anim.save("Animations/example.mp4", fps=1, dpi=150)
    # save animation at 30 frames per second
    anim.save('Animations/myAnimation_4S.gif', writer='imagemagick', fps=5)
    # plt.close()

    if speakerwise_colouring:
        # line1 = Line2D(range(1), range(1), color="green", marker='o', markersize=7, linestyle='none')
        # line2 = Line2D(range(1), range(1), color="red", marker='o', markersize=7, linestyle='none')
        line3 = Line2D([0], [0], color=speakerwise_colours[0], lw=1),
        line4 = Line2D([0], [0], color=speakerwise_colours[1], lw=1),

        plt.legend((line3, line4), (names[0], names[1]), prop={'size': 7})

    # AXIS STUFF
    ax1.dist = 13
    ax1 = plt.gca()
    # ax.xaxis.set_ticklabels([])
    ax1.yaxis.set_ticklabels([])
    ax1.zaxis.set_ticklabels([])

    # for line in ax.xaxis.get_ticklines():
    #     line.set_visible(False)
    for line in ax1.yaxis.get_ticklines():
        line.set_visible(False)
    for line in ax1.zaxis.get_ticklines():
        line.set_visible(False)

    # if save_fig:
    #     plt.savefig("Saved_Images/{0}/{1}.png".format(transcript_name, save_name), dpi=600)

    # plt.show()
    anim.save('Animations/myAnimation_3D.gif', writer='imagemagick', fps=5)
    plt.close()

    return



## The main function putting it all together

def Convert_PDF_to_txt(path_to_pdf):
    """
    Function to convert a saved PDF document to a txt file.

    i.e. used for..
    path_to_pdf = Path('msci-project/transcripts/joe_rogan_kanye_west_2.pdf')
    Convert_PDF_to_txt(path_to_pdf)
    """
    import PyPDF2

    # mypdf = open(path_to_pdf, mode='rb')
    # pdf_document = PyPDF2.PdfFileReader(mypdf)
    # text = pdf_document.extractText()
    #
    # with open("msci-project/transcripts/joe_rogan_kanye_west.txt", "w") as f:
    #     #for item in utterances_speakerwise[1]:
    #     f.write("%s\n" % text)
    # print('extracted text preview: ', text[:200])

    with open(path_to_pdf, 'rb') as pdf_file, open('msci-project/transcripts/joe_rogan_kanye_west.txt', 'w') as text_file:
        read_pdf = PyPDF2.PdfFileReader(pdf_file)
        number_of_pages = read_pdf.getNumPages()
        for page_number in range(number_of_pages):  # use xrange in Py2
            page = read_pdf.getPage(page_number)
            page_content = page.extractText()
            text_file.write(page_content)

    return


def Split_Transcript_By_Speaker(content, names):
    """
    Function to prepare transcript content for speaker-wise analysis.
    """
    if names[1] =='Elon Musk':
        # Get rid of the time marks
        text_1 = re.sub('\([0-9]{2}:[0-9]{2}:[0-9]{2}\)', " ", content)  # \w+\s\w+;
        text_2 = re.sub('\([0-9]{2}:[0-9]{2}\)', " ", text_1)
        text = re.sub('…', "...", text_2)
        # print('\n content without time marks', text[:300])

        # Remove speaker names
        codes = [888123, 888321]  # Just two random codes for the speakers
        for name, code in zip(names, codes):
            text = re.sub(str(name).title() + ':', str(code), text)

    if names[1] =='Jack Dorsey':
        # Change dashes to ...'s when speaker is cut off
        content = re.sub('-\s\s', "...", content)
        # Get rid of the time marks
        text_1 = re.sub('[0-9]{2}:[0-9]{2}:[0-9]{2}', " ", content)  # \w+\s\w+;
        text = re.sub('\([0-9]{2}:[0-9]{2}\)', " ", text_1)

        print('\n content without time marks', text[:300])

        # Remove speaker names
        codes = [888123, 888321]  # Just two random codes for the speakers
        for name, code in zip(names, codes):
            text = re.sub(str(name).title(), str(code), text)

    print('text', text[:500])
    # Strip new-lines
    content_2 = re.sub('\n', " ", text)
    # Strip white spaces
    content_2 = content_2.strip()

    # # Create list of all sentences (utterances) in transcript
    # all_sentences = sent_tokenize(content_2)
    # print('all_sentences', len(all_sentences), all_sentences)

    # OR splitting it into utterances (i.e. even if one person says multiple sentences they'll be put together and counted as only 1 sentence (utterance)
    utterances = content_2.split('888')
    all_utterances = [utt for utt in utterances if utt !='']
    # print('all_sentences', len(all_sentences), all_sentences)

    # Split into two lists of sentences, one for each speaker
    content_speaker_1 = [utt[3:] for utt in all_utterances if utt[:3] == '123']
    content_speaker_2 = [utt[3:] for utt in all_utterances if utt[:3] == '321']

    # print('Number of utterances by speaker 1 (', names[0], ') : ', len(content_speaker_1), '. First few utterances:', content_speaker_1[:3])
    # print('Number of utterances by speaker 2 (', names[1], ') : ', len(content_speaker_2), '. First few utterances:', content_speaker_2[:3])
    # print('total above:', len(content_speaker_1) + len(content_speaker_2))
    # print('\nall_utterances length:', len(all_utterances), 'Preview: ', all_utterances[:20])

    content_speaker_1 = [utt[3:-3] for utt in content_speaker_1]
    content_speaker_2 = [utt[3:-3] for utt in content_speaker_2]

    return all_utterances, [content_speaker_1, content_speaker_2]

## Other analysis stuff
def Interupption_Analysis(save_fig=False):
    """
    Function to look at how often each speaker cuts off the other. Build a profile for each speaker when looking at this
    vs how many Questions they ask/ topics they introduce/ time spoken
    """
    names = ['Joe', 'Rogan', 'Jack', 'Dorsey'] #'Elon', 'Musk'] #'Jack', 'Dorsey']

    with open('txts/Joe_Rogan_{0}/all_utterances.txt'.format('_'.join(names[2:4])), 'r') as f:
        all_utts = f.read()

    names_dict = {'123':[' '.join(names[:2]), 'blue'], '321': [' '.join(names[2:4]), 'green']}
    sents = all_utts.split('\n')

    idxs_sents_with_cutoff = [idx for idx, sent in enumerate(sents) if '...  ' in sent]
    print('', len(idxs_sents_with_cutoff))
    # number of times speaker 123 was interrupted by 321
    idxs_123 = [idx for idx in idxs_sents_with_cutoff if sents[idx][:3] == '123']
    # number of times speaker 123 was interrupted by 123
    idxs_321 = [idx for idx in idxs_sents_with_cutoff if sents[idx][:3] == '321']
    pprint(np.array(sents)[idxs_sents_with_cutoff])

    idxs = np.zeros(len(sents))
    colours = ['k' for i in idxs]
    for i in idxs_sents_with_cutoff:
        idxs[i] = 1
        colours[i] = names_dict[sents[i][:3]][1]

    plt.figure()
    xs = range(len(sents))
    for i in xs:
        plt.plot([xs[i], xs[i]], [0, idxs[i]], '-', color=colours[i], lw=2)

    plt.xlabel('Sentence Number')
    plt.ylabel('Interruption')
    legend_elements = [Line2D([0], [0], color=list(names_dict.values())[0][1], lw=1, label=list(names_dict.values())[1][0]),
                       Line2D([0], [0], color=list(names_dict.values())[1][1], lw=1, label=list(names_dict.values())[0][0])]
    plt.title('Speakers Interrupted - {0} interview'.format(' '.join(names[2:4])))
    plt.legend(handles=legend_elements)
    plt.ylim([0, 1.2])
    if save_fig:
        plt.savefig("Saved_Images/{0}/Interruptions_{1}.png".format('_'.join([n.lower() for n in names]), names[2]), dpi=600)
    plt.show()

    # Info on interruptions
    mini_df = {'Speaker': [], 'Total #Interruptions': []}

    mini_df['Speaker'] = ['Joe Rogan', ' '.join(names[2:4])]
    mini_df['Total #Interruptions'] = [len(idxs_321), len(idxs_123)]
    # NOTE the above ^ 'len's are swapped around as we want to know how many times the speaker interrupted the other,
    # not how many times they themselves were interupptrd

    # Create df and save / print in xml
    mini_df = pd.DataFrame({k: pd.Series(l) for k, l in mini_df.items()})

    print(tabulate.tabulate(mini_df.values, mini_df.columns, tablefmt="pipe"))
    mini_df.to_hdf('Saved_dfs/joe_rogan_{}/interruptions_df.h5'.format('_'.join([n.lower() for n in names[2:4]])), key='df', mode='w')

    return

def Snappyness(name, n_average=True, n = 5, normalised=False):
    """Not sure how I'll define this property of conversation but for now just plot length of speaker turn"""

    import string
    import statistics
    file = pd.read_pickle("processed_transcripts/joe_rogan_{0}.pkl".format(name))
    print(file[:100].to_string())

    length_of_utts = []
    old_sent_coords = [0, 0]
    old_idx = 0
    for idx, row in file.iterrows():
        new_speaker = row['speaker_change']
        old_speaker = file.speaker[old_idx]  # i.e. the speaker who said all utterances between old index and new index

        if not new_speaker:
            # Only want to plot line when the speaker has changed
            old_idx = idx
            continue

        # collect utterances from old_idx to new_idx-1
        utt = ' '.join(list(file.utterance[old_idx:idx]))
        words_in_utt = utt.split(' ')
        # remove punctuation and single-letter strings unless they're "i" (i.e. so ['he', ''', 'd'] (he'd) = ['he']
        words_in_utt = [word for word in words_in_utt if word not in string.punctuation]
        words_in_utt = [word for word in words_in_utt if len(word)>1 or word.lower()=='i']
        num_words_in_utt = len(words_in_utt)

        length_of_utts.append(num_words_in_utt)
        # colour = speakers_map[old_speaker]

        # # Collect Utterances of this speaker
        # Utts = list(file.da_label[old_idx:idx])
        # tag_change = any(x in changer_DAs for x in das_covered)
    if not n_average:
        #plotting all
        if not normalised:
            plt.figure()
            plt.title(f'{name.title()} interview: Length of Utterances')
            plt.bar(range(len(length_of_utts)), length_of_utts)
            plt.ylabel('Utterance length')
            plt.xlabel('Utterance Number')
            plt.show()
        if normalised:
            max_y = max(length_of_utts)
            num_utts = len(length_of_utts)
            xs = range(len(length_of_utts))
            xs_normalised = [val/ num_utts for val in xs]
            length_of_utts_normalised = [val / max_y for val in length_of_utts]

            plt.figure()
            plt.title(f'{name.title()} interview: Normalised Utterances Lengths')
            plt.bar(xs, length_of_utts_normalised)
            plt.ylabel('Utterance length')
            plt.xlabel('Utterance Number')
            plt.show()

    if n_average:
        list1 = list(itertools.chain.from_iterable([statistics.mean(length_of_utts[i:i+n])]*n for i in range(0,len(length_of_utts),n)))
        if not normalised:
            plt.figure()
            plt.title(f'{name.title()} interview: Length of Utterances (averaging every {n} utt lengths)')
            plt.bar(range(len(list1)), list1)
            plt.ylabel('Utterance length')
            plt.xlabel('Utterance Number')
            plt.show()
        if normalised:
            max_y = max(list1)
            num_utts = len(list1)
            xs = range(len(list1))
            xs_normalised = [val / num_utts for val in xs]
            list1_normalised = [val/max_y for val in list1]

            plt.figure()
            plt.title(f'{name.title()} interview: Normalised Utterance Lengths\n(averaged every {n})')
            plt.bar(xs, list1_normalised)
            plt.ylabel('Normalised Utterance Length')
            plt.xlabel('Utterance Number')
            plt.show()

    return

def Snappyness_EvenSegs(name, n=200, normalised=False):
    """
    Split into n even segments and find utterance lengths + normalise
    """

    import string
    import statistics
    file = pd.read_pickle("processed_transcripts/joe_rogan_{0}.pkl".format(name))
    print(file[:100].to_string())

    length_of_utts = []
    old_sent_coords = [0, 0]
    old_idx = 0
    for idx, row in file.iterrows():
        new_speaker = row['speaker_change']
        old_speaker = file.speaker[
            old_idx]  # i.e. the speaker who said all utterances between old index and new index

        if not new_speaker:
            # Only want to plot line when the speaker has changed
            old_idx = idx
            continue

        # collect utterances from old_idx to new_idx-1
        utt = ' '.join(list(file.utterance[old_idx:idx]))
        words_in_utt = utt.split(' ')
        # remove punctuation and single-letter strings unless they're "i" (i.e. so ['he', ''', 'd'] (he'd) = ['he']
        words_in_utt = [word for word in words_in_utt if word not in string.punctuation]
        words_in_utt = [word for word in words_in_utt if len(word) > 1 or word.lower() == 'i']
        num_words_in_utt = len(words_in_utt)

        length_of_utts.append(num_words_in_utt)

        #list1 = list(itertools.chain.from_iterable([statistics.mean(length_of_utts[i:i+n])]*n for i in range(0,len(length_of_utts),n)))

    num_utts = len(length_of_utts)
    print('length_of_utts: ', num_utts, length_of_utts)
    idxs_split = split(range(num_utts), n)
    first_sent_idxs_list = [i[0] for i in idxs_split]
    first_sent_idxs_list.insert(-1, num_utts)
    print('first_sent_idxs_list', len(first_sent_idxs_list), first_sent_idxs_list)
    utt_lengths_split = [length_of_utts[i1:i2] for i1, i2 in zip(first_sent_idxs_list, first_sent_idxs_list[1:])]

    print('utt_lengths_split: ', len(utt_lengths_split), utt_lengths_split)
    average_utts_length = [np.average(sublist) for sublist in utt_lengths_split]
    print('average_utts_length', len(average_utts_length), average_utts_length)

    if not normalised:
        plt.figure()
        plt.title(f'{name.title()} interview: Average Length of Utterances ({n} Even Segments)')
        plt.bar(range(len(average_utts_length)), average_utts_length)
        plt.ylabel('Utterance length')
        plt.xlabel('Utterance Number')

    if normalised:
        max_y = max(average_utts_length)
        num_utts = len(average_utts_length)
        xs = range(n)
        list1_normalised = [val/max_y for val in average_utts_length]
        print('list1_normalised', len(list1_normalised), list1_normalised)

        plt.figure()
        plt.title(f'{name.title()} interview: Normalised Average Length of Utterances ({n} Even Segments)')
        plt.bar(xs, list1_normalised)

        # create nth degree polynomial fit
        # n = 1
        # zn = np.polyfit(xs[::2], list1_normalised[::2], n)
        # pn = np.poly1d(zn)  # construct polynomial

        # create qth degree polynomial fit
        q = 10
        zq = np.polyfit(xs[::2], list1_normalised[::2], q)
        pq = np.poly1d(zq)

        # plot data and fit
        xx = np.linspace(0, max(xs), 50)
        # plt.plot(xx, pn(xx), color='g')
        plt.plot(xx, pq(xx), color='k')
        #plt.plot(xs, list1_normalised, '-')

        plt.ylabel('Normalised Utterance Length')
        plt.xlabel('Utterance Number')

    #plt.savefig("Saved_Images/Stuff/{0}_n:{1}_normalised:{2}.png".format(name, n, normalised), dpi=600)

    plt.show()
    return

#Interupption_Analysis(save_fig=True)
#Snappyness('jack_dorsey', n_average=False, n = 20, normalised=True) #'elon_musk' #'jack_dorsey'
#Snappyness_EvenSegs('jack_dorsey', n=100, normalised=True)




## Discussion Tree Building
def Simple_Line_DA(cutoff_sent=200, Interviewee='elon musk', save_fig=False):
    """
    Function to plot line
    black line moving between points

    step size = 2

    if two utterances in a row  both ave changer DAs but are spoken by the same speaker, only count that as one
    or look at the second Q and check whether the NEXT sentence has a '1' for the speaker change?

    or only add a node when the speaker has changed and check all labels that happened between this one and the last
    node and if ANY of them are in change_DAs then change direction
    """
    changer_DAs = ["Wh-Question", "Yes-No-Question", "Declarative Yes-No-Question", "Declarative Wh-Question"]
    speakers_map = {'joe rogan': 'purple', Interviewee : 'blue'}
    step_size = 1
    transcript_name = "joe_rogan_{}".format('_'.join(list(speakers_map.keys())[1].split(' ')))

    file = pd.read_pickle("processed_transcripts/{}.pkl".format(transcript_name))
    print(file[:100].to_string())

    plt.figure()
    plt.title('Detecting "Changer" Dialogue Acts')
    old_sent_coords = [0, 0]
    old_idx = 0
    for idx, row in file[1:cutoff_sent].iterrows():
        new_speaker = row['speaker_change']
        old_speaker = file.speaker[old_idx] #i.e. the speaker who said all utterances between old index and new index

        if not new_speaker:
            # Only want to plot line when the speaker has changed
            old_idx = idx
            continue

        colour = speakers_map[old_speaker]

        # Check whether any change DAs between last node and now..
        das_covered = list(file.da_label[old_idx:idx])
        tag_change = any(x in changer_DAs for x in das_covered)

        if not tag_change:
            new_dir_x = 1 # 1 bc x direction is for continuing on convo #old_dir_x  # 1 if moving along x, 0 if moving along y
            change_in_coords = [step_size, 0] if new_dir_x else [0, step_size]
            new_sent_coords = list(map(add, old_sent_coords, change_in_coords))

            plt.plot(new_sent_coords[0], new_sent_coords[1], 'o', color='k', ms=3) #plot node
            plt.plot([old_sent_coords[0], new_sent_coords[0]],[old_sent_coords[1], new_sent_coords[1]], '-',
                     color=colour)# plot line

        if tag_change:
            print('\ntopic', row['topics'])
            print('words', row['key_words'])
            new_dir_x = 0 #not(old_dir_x)
            change_in_coords = [step_size, 0] if new_dir_x else [0, step_size]
            new_sent_coords = list(map(add, old_sent_coords, change_in_coords))
            print('new_sent_coords', new_sent_coords)
            print('change_in_coords', change_in_coords)

            try:
                annotation = row['topics'] #[0]

            except:
                annotation = 'Nan'

            plt.plot(new_sent_coords[0], new_sent_coords[1], 'o', color='k', ms=3)  # plot node
            plt.annotate(annotation, xy=(new_sent_coords[0]-10, new_sent_coords[1]+0.1), color='k', zorder=100), # textcoords="offset points" #weight=)
            plt.plot([old_sent_coords[0], new_sent_coords[0]], [old_sent_coords[1], new_sent_coords[1]], '-',
                     color=colour)  # plot line

        old_sent_coords = new_sent_coords
        old_idx = idx

    # If want equal axis sizes
    # largest_dim = max(old_sent_coords) + 5
    # plt.xlim(0, largest_dim)
    # plt.ylim(-10, largest_dim)
    legend_handles = []
    legend_labels = []
    for i in range(len(list(speakers_map.keys()))):
        legend_handles.append(Line2D([0], [0], color=list(speakers_map.values())[i], lw=1))
        legend_labels.append(list(speakers_map.keys())[i])

    # plt.xlabel('Only Statements in Utterance')
    # plt.ylabel('Question in Utterance')
    plt.legend(legend_handles, legend_labels)

    if save_fig:
        plt.savefig("Saved_Images/{0}/Simple_Line_DA.png".format(transcript_name), dpi=600)
    plt.show()

    return

def Simple_Line_Topics(cutoff_sent=200, Interviewee='elon musk', save_fig=False):
    """"
    Function
    """
    speakers_map = {'joe rogan': 'purple', Interviewee: 'blue'}
    step_size = 1
    transcript_name = "joe_rogan_{}".format('_'.join(list(speakers_map.keys())[1].split(' ')))
    file = pd.read_pickle("processed_transcripts/{}.pkl".format(transcript_name))
    print(file[:100].to_string())

    topic_linegraph_dict = {'Idx' :[], 'All_Current_Topics' :[], 'New_Topic':[], 'Speaker':[], 'Sentence':[],
                            'DA_Label': []}

    plt.figure()
    plt.title('Simple Line with Topic Step : Jack Dorsey')
    old_sent_coords = [0, 0]
    old_idx = 0
    old_topics, most_recently_plotted = [], ''
    first_topic_idx = int(file.index[file['topics'].astype(bool)].tolist()[0])
    print('first_topic_idx', first_topic_idx)
    print(file.iloc[first_topic_idx]['topics'])

    for idx, row in file[0:cutoff_sent].iterrows():
        old_speaker = file.speaker[old_idx]  # i.e. the speaker who said all utterances between old index and new index
        colour = speakers_map[old_speaker]

        if str(file.topics[old_idx]) == 'nan':
            old_idx = idx
            continue

        print('\nold_topics', old_topics)
        current_topics = list(row['topics'])

        print('current_topic', current_topics)
        continued_topics = [x for x in old_topics if x in current_topics]
        # if continued_topics != most_recently_plotted:
        #     # i.e. yes we're carrying on topics from last utterance to this one, but it's not been plotted!
        print('continued_topics', continued_topics)

        continued_topic = False if len(continued_topics)==0 else True #new_topic != current_topic else False
        print('continued_topic?', continued_topic)
        new_topic = [x for x in current_topics if x in list(file.topics[idx + 1])]
        if old_sent_coords[0] == 0 and old_sent_coords[1] == 0:
            plt.annotate(', '.join(new_topic), xy=(0, 1 + 0.2), color='k',
                         zorder=100),  # textcoords="offset points" #weight=)
        if continued_topic:
            new_dir_x = 1 # 1 bc x direction is for continuing on convo on current topic
            change_in_coords = [step_size, 0]
            new_sent_coords = list(map(add, old_sent_coords, change_in_coords))

            #plt.plot([old_sent_coords[0], new_sent_coords[0]], [old_sent_coords[1], new_sent_coords[1]], '-', color=colour)
            plt.plot(new_sent_coords[0], new_sent_coords[1], 'o', color='k', ms=3, zorder=100)  # plot node

        if not continued_topic:
            # print('\ntopic', row['topics'])
            # print('words', row['key_words'])
            topic_linegraph_dict['Idx'].append(idx)
            topic_linegraph_dict['All_Current_Topics'].append(current_topics)
            topic_linegraph_dict['New_Topic'].append(new_topic)
            topic_linegraph_dict['Speaker'].append(row['speaker'])
            topic_linegraph_dict['Sentence'].append(row['utterance'])
            topic_linegraph_dict['DA_Label'].append(row['da_label'])

            change_in_coords = [0, step_size]
            new_sent_coords = list(map(add, old_sent_coords, change_in_coords))

            plt.annotate(', '.join(new_topic), xy=(new_sent_coords[0]-15.5, new_sent_coords[1]+0.2), color='k',
                         zorder=100),  # textcoords="offset points" #weight=)
            plt.plot([old_sent_coords[0], new_sent_coords[0]], [old_sent_coords[1], new_sent_coords[1]], '-',
                     color=colour, linewidth=4)  # plot line
            plt.plot(new_sent_coords[0], new_sent_coords[1], 'o', color='k', ms=3, zorder=100)  # plot node

        old_topics = new_topic
        old_sent_coords = new_sent_coords
        old_idx = idx

    # Create df
    topic_linegraph_df = pd.DataFrame({k: pd.Series(l) for k, l in topic_linegraph_dict.items()})

    # Mini df for printing
    mini_df = {'Speaker': [], 'Total #Utterances': [], 'Total #Questions Asked': [], 'Total #Statements': [],
               'Number of Topics Introduced': [],
               '#Topics introduced by Statement' :[], '#Topics introduced by Question':[]} #'Topics Introduced': [],

    mini_df['Speaker'] = ['Joe Rogan', Interviewee.title()]
    joe_total_df = file[file['speaker'] == 'joe rogan']
    interviewee_total_df = file[file['speaker'] == Interviewee]
    joe_total_utts = len(joe_total_df)
    interviewee_total_utts = len(interviewee_total_df)
    mini_df['Total #Utterances'] = [joe_total_utts, interviewee_total_utts]
    joe_total_Qs = len([idx for idx, row in joe_total_df.iterrows() if 'Question' in row['da_label']])
    interviewee_total_Qs = len([idx for idx, row in interviewee_total_df.iterrows() if 'Question' in row['da_label']])
    joe_total_Ss = len([idx for idx, row in joe_total_df.iterrows() if 'Statement' in row['da_label']])
    interviewee_total_Ss = len([idx for idx, row in interviewee_total_df.iterrows() if 'Statement' in row['da_label']])
    mini_df['Total #Questions Asked'] = [joe_total_Qs, interviewee_total_Qs]
    mini_df['Total #Statements'] = [joe_total_Ss, interviewee_total_Ss]

    joe_df = topic_linegraph_df[topic_linegraph_df['Speaker'] == 'joe rogan']
    interviewee_df = topic_linegraph_df[topic_linegraph_df['Speaker'] == Interviewee]
    mini_df['Number of Topics Introduced'].append(len(joe_df))
    mini_df['Number of Topics Introduced'].append(len(interviewee_df))
    topics_introduced_joe = ', '.join(list(itertools.chain.from_iterable(list(joe_df.New_Topic))))
    topics_introduced_interviewee = ', '.join(list(itertools.chain.from_iterable(list(interviewee_df.New_Topic))))
    print('topics_introduced_joe', topics_introduced_joe)
    print('topics_introduced_interviewee', topics_introduced_interviewee)
    mini_df['#Topics introduced by Statement'].append(len([idx for idx, row in joe_df.iterrows() if 'Statement' in row['DA_Label']]))
    mini_df['#Topics introduced by Statement'].append(len([idx for idx, row in interviewee_df.iterrows() if 'Statement' in row['DA_Label']]))
    mini_df['#Topics introduced by Question'].append(len([idx for idx, row in joe_df.iterrows() if 'Question' in row['DA_Label']]))
    mini_df['#Topics introduced by Question'].append(len([idx for idx, row in interviewee_df.iterrows() if 'Question' in row['DA_Label']]))

    # Create df
    mini_df = pd.DataFrame({k: pd.Series(l) for k, l in mini_df.items()})

    print(tabulate.tabulate(topic_linegraph_df.values, topic_linegraph_df.columns, tablefmt="pipe"))
    print(tabulate.tabulate(mini_df.values, mini_df.columns, tablefmt="pipe"))
    topic_linegraph_df.to_hdf('Saved_dfs/{0}/topic_linegraph_df.h5'.format(transcript_name), key='df', mode='w')
    mini_df.to_hdf('Saved_dfs/{0}/mini_df.h5'.format(transcript_name), key='df', mode='w')

    legend_handles = []
    legend_labels = []
    for i in range(len(list(speakers_map.keys()))):
        legend_handles.append(Line2D([0], [0], color=list(speakers_map.values())[i], lw=1))
        legend_labels.append([x.title() for x in list(speakers_map.keys())][i])

    # plt.xlabel('Only Statements in Utterance')
    # plt.ylabel('Question in Utterance')
    plt.ylim(top = new_sent_coords[1]+1)
    plt.legend(legend_handles, legend_labels, loc='upper left')

    if save_fig:
        plt.savefig("Saved_Images/{0}/Simple_Line_Topics.png".format(transcript_name), dpi=600)
    plt.show()
    return

def Shifting_Line_Topics(cutoff_sent=400, Interviewee='elon musk', save_fig=False):
    """"
    Function
    """
    speakers_map = {'joe rogan': 'purple', Interviewee: 'blue'}
    transcript_name = "joe_rogan_{}".format('_'.join(list(speakers_map.keys())[1].split(' ')))
    file = pd.read_pickle("processed_transcripts/{}.pkl".format(transcript_name))
    print(file[:100].to_string())

    step_size_x, step_size_y = 1, 1

    topic_linegraph_dict = {'Idx': [], 'All_Current_Topics': [], 'New_Topic': [], 'Speaker': [], 'Sentence': [],
                            'DA_Label': []}

    plt.figure()
    plt.title('Shifting Topic Line')
    old_sent_coords = [0, 0]
    old_idx = 0
    old_topics, most_recently_plotted = [], ''
    Dict_of_topics, Dict_of_topics_counts = {}, {}

    for idx, row in file[1:cutoff_sent].iterrows():
        old_speaker = file.speaker[old_idx]  # i.e. the speaker who said all utterances between old index and new index
        colour = speakers_map[old_speaker]
        if str(file.topics[old_idx]) == 'nan':
            old_idx = idx
            continue

        current_topics = list(row['topics'])
        continued_topics = [x for x in old_topics if x in current_topics]
        continued_topic = False if len(continued_topics) == 0 else True

        if continued_topic:
            change_in_coords = [0, step_size_y]
            new_sent_coords = list(map(add, old_sent_coords, change_in_coords))

            plt.plot(new_sent_coords[0], new_sent_coords[1], 'o', color='k', ms=3)  # plot node
            plt.plot([old_sent_coords[0], new_sent_coords[0]], [old_sent_coords[1], new_sent_coords[1]], '-',
                     color=colour)

        if not continued_topic:
            new_topic = [x for x in current_topics if x in list(file.topics[idx + 1])]
            topic_linegraph_dict['Idx'].append(idx)
            topic_linegraph_dict['All_Current_Topics'].append(current_topics)
            topic_linegraph_dict['New_Topic'].append(new_topic)
            topic_linegraph_dict['Speaker'].append(row['speaker'])
            topic_linegraph_dict['Sentence'].append(row['utterance'])
            topic_linegraph_dict['DA_Label'].append(row['da_label'])

            change_in_coords = [step_size_x, 0]
            new_sent_coords = list(map(add, old_sent_coords, change_in_coords))


            ## NOW CHECK FOR RETURNS...
            print('new_topic', new_topic)
            print(Dict_of_topics)
            # Check if topic has already been visited.
            the_topic = None
            for topic in new_topic:  # NOTE what if new_topic contains >1 topics which have been mentioned in DIFFERENT PLACES??... WHICH X TO GO TO?
                if topic in Dict_of_topics:
                    X_pos, Y_pos = Dict_of_topics[topic]
                    the_topic = topic
                else:
                    continue


            # check if X_pos has been assigned, if not, assign new position
            if the_topic is None:
                for topic in new_topic:  #  in case it contains multiple, but they all get this x position.
                    Dict_of_topics[topic] = new_sent_coords
                    Dict_of_topics_counts[topic] = 1

                # then plot…
                plt.plot(new_sent_coords[0], new_sent_coords[1], 'o', color='k', ms=3)  # plot node
                plt.annotate(', '.join(current_topics), xy=(new_sent_coords[0]+0.1, new_sent_coords[1]), color='k',
                             zorder=100),  # textcoords="offset points" #weight


            else: # i.e. if we are jumping back to a previously mentioned topic
                Dict_of_topics_counts[the_topic] += 1
                step_size_x += 0.1  # also adjust step size so don't return to other topics
                new_sent_coords = [X_pos, new_sent_coords[1]]  # keep y position but change x.

                plt.plot(new_sent_coords[0], new_sent_coords[1], 'o', color='k', ms=3)  # plot node
                plt.plot([X_pos, X_pos], [Y_pos, new_sent_coords[1]], '--', color='k', linewidth=1)  #  add dashed line between last one and this one

                plt.annotate(', '.join(current_topics), xy=(new_sent_coords[0], new_sent_coords[1]), color='k',
                             zorder=100),  # textcoords="offset points" #weight=

                # if Dict_of_topics_counts[topic] == 2:
                #     # Annotate the line
                #     plt.annotate(the_topic, xy=(Dict_of_topics[topic][0]-2, Dict_of_topics[topic][1]+2), color='k',
                #                  zorder=100, rotation=90, weight='bold'),  # textcoords="offset points" #weight=)

            plt.plot([old_sent_coords[0], new_sent_coords[0]], [old_sent_coords[1], new_sent_coords[1]], '-',
                     color=colour)  # plot line

        old_topics = current_topics #new_topic
        old_sent_coords = new_sent_coords
        old_idx = idx


    print(' the final index of row', idx)
    print('the final y position', new_sent_coords[1])
    # print('Dict_of_topics', Dict_of_topics)
    # print('Dict_of_topics_counts', Dict_of_topics_counts)
    legend_handles = []
    legend_labels = []
    for i in range(len(list(speakers_map.keys()))):
        legend_handles.append(Line2D([0], [0], color=list(speakers_map.values())[i], lw=1))
        legend_labels.append(list(speakers_map.keys())[i])

    # plt.xlabel('Only Statements in Utterance')
    # plt.ylabel('Question in Utterance')
    plt.legend(legend_handles, legend_labels)
    if save_fig:
        plt.savefig("Saved_Images/{0}/Shifting_Line_Topics.png".format(transcript_name), dpi=600)
    plt.show()
    return

def Shifting_Line_Topics_2(cutoff_sent=400, Interviewee='jack dorsey', save_fig=False):
    """"Function """
    speakers_map = {'joe rogan': 'purple', Interviewee: 'blue'}
    transcript_name = "joe_rogan_{}".format('_'.join(list(speakers_map.keys())[1].split(' ')))
    file = pd.read_pickle("processed_transcripts/{}.pkl".format(transcript_name))
    print(file[:100].to_string())

    step_size_x, step_size_y = 1, 1

    topic_linegraph_dict = {'Idx': [], 'All_Current_Topics': [], 'New_Topic': [], 'Speaker': [], 'Sentence': [],
                            'DA_Label': []}

    plt.figure()
    plt.title('Shifting Topic Line 2: {0}'.format(Interviewee.title()))
    old_sent_coords = [0, 0]
    old_idx = 0
    old_topics, most_recently_plotted = [], ''
    Dict_of_topics, Dict_of_topics_counts = {}, {}

    for idx, row in file[1:cutoff_sent].iterrows():
        old_speaker = file.speaker[old_idx]  # i.e. the speaker who said all utterances between old index and new index
        colour = 'k' #speakers_map[old_speaker]
        if str(file.topics[old_idx]) == 'nan':
            old_idx = idx
            continue

        current_topics = list(row['topics'])
        continued_topics = [x for x in old_topics if x in current_topics]
        continued_topic = False if len(continued_topics) == 0 else True

        if continued_topic:

            change_in_coords = [0, step_size_y]
            new_sent_coords = list(map(add, old_sent_coords, change_in_coords))

            plt.plot(new_sent_coords[0], new_sent_coords[1], 'o', color='k', ms=3)  # plot node
            plt.plot([old_sent_coords[0], new_sent_coords[0]], [old_sent_coords[1], new_sent_coords[1]], '-',
                     color=colour)

        if not continued_topic:
            new_topic = [x for x in current_topics if x in list(file.topics[idx + 1])]
            topic_linegraph_dict['Idx'].append(idx)
            topic_linegraph_dict['All_Current_Topics'].append(current_topics)
            topic_linegraph_dict['New_Topic'].append(new_topic)
            topic_linegraph_dict['Speaker'].append(row['speaker'])
            topic_linegraph_dict['Sentence'].append(row['utterance'])
            topic_linegraph_dict['DA_Label'].append(row['da_label'])

            change_in_coords = [step_size_x, 0]
            new_sent_coords = list(map(add, old_sent_coords, change_in_coords))


            ## NOW CHECK FOR RETURNS...
            #print('new_topic', new_topic)
            #print(Dict_of_topics)
            # Check if topic has already been visited.
            the_topic = None
            for topic in new_topic:  # NOTE what if new_topic contains >1 topics which have been mentioned in DIFFERENT PLACES??... WHICH X TO GO TO?
                if topic in Dict_of_topics:
                    X_pos, Y_pos = Dict_of_topics[topic]
                    the_topic = topic
                else:
                    continue


            # check if X_pos has been assigned, if not, assign new position
            if the_topic is None:
                for topic in new_topic:  #  in case it contains multiple, but they all get this x position.
                    Dict_of_topics[topic] = new_sent_coords
                    Dict_of_topics_counts[topic] = 1

                # then plot…
                plt.plot(new_sent_coords[0], new_sent_coords[1], 'o', color='k', ms=1)  # plot node
                # plt.annotate(', '.join(current_topics), xy=(new_sent_coords[0]+1, new_sent_coords[1]), color='k',
                #              zorder=100),  # textcoords="offset points" #weight
                print('the_topic', the_topic, '. Current topics:', current_topics, '. New_topic: ',new_topic)


            else: # i.e. if we are jumping back to a previously mentioned topic
                Dict_of_topics_counts[the_topic] += 1
                step_size_x += 0.5  # also adjust step size so don't return to other topics
                new_sent_coords = [X_pos, new_sent_coords[1]]  # keep y position but change x.

                plt.plot(new_sent_coords[0], new_sent_coords[1], 'o', color='k', ms=1)  # plot node
                plt.plot([X_pos, X_pos], [Y_pos, new_sent_coords[1]], '--', color='k', linewidth=1)  #  add dashed line between last one and this one

                # plt.annotate(', '.join(current_topics), xy=(new_sent_coords[0], new_sent_coords[1]), color='k',
                #              zorder=100),  # textcoords="offset points" #weight=
                print('the_topic', the_topic, '. Current topics:', current_topics, '. New_topic: ',new_topic)

                if Dict_of_topics_counts[the_topic] == 2:
                    # Annotate the line
                    plt.annotate(the_topic, xy=(Dict_of_topics[the_topic][0]-0.7, Dict_of_topics[the_topic][1]+9),
                                 color='k', zorder=100, rotation=90, weight='bold')

            plt.plot([old_sent_coords[0], new_sent_coords[0]], [old_sent_coords[1], new_sent_coords[1]], '-',
                     color=colour)  # plot line

        old_topics = current_topics #new_topic
        old_sent_coords = new_sent_coords
        old_idx = idx


    print(' the final index of row', idx)
    print('the final y position', new_sent_coords[1])
    # print('Dict_of_topics', Dict_of_topics)
    # print('Dict_of_topics_counts', Dict_of_topics_counts)
    # legend_handles = []
    # legend_labels = []
    # for i in range(len(list(speakers_map.keys()))):
    #     legend_handles.append(Line2D([0], [0], color=list(speakers_map.values())[i], lw=1))
    #     legend_labels.append(list(speakers_map.keys())[i])

    # plt.xlabel('Only Statements in Utterance')
    # plt.ylabel('Question in Utterance')
    # plt.legend(legend_handles, legend_labels)
    if save_fig:
        plt.savefig("Saved_Images/{0}/Simpler_Shifting_Line_Topics.png".format(transcript_name), dpi=600)
    plt.show()
    return

def DT_Shifting_Line_Topics(Interviewee='jack dorsey', logscalex=True, save_fig=False):
    """"
    Full transcript
    Log scale on x axis
    simple appearance with red annotations
    """
    speakers_map = {'joe rogan': 'purple', Interviewee: 'blue'}
    transcript_name = "joe_rogan_{}".format('_'.join(list(speakers_map.keys())[1].split(' ')))
    file = pd.read_pickle("processed_transcripts/{}.pkl".format(transcript_name))
    print(file[:100].to_string())

    step_size_x, step_size_y = 1, 1

    topic_linegraph_dict = {'Idx': [], 'All_Current_Topics': [], 'New_Topic': [], 'Speaker': [], 'Sentence': [],
                            'DA_Label': []}

    plt.figure()
    plt.title('Shifting Topic Line {0}, log(x) scale:{1}'.format(Interviewee.title(), logscalex))
    old_sent_coords = [0, 0]
    old_idx = 0
    old_topics, most_recently_plotted = [], ''
    Dict_of_topics, Dict_of_topics_counts = {}, {}

    for idx, row in file[1:].iterrows():
        old_speaker = file.speaker[old_idx]  # i.e. the speaker who said all utterances between old index and new index
        colour = 'k' #speakers_map[old_speaker]
        if str(file.topics[old_idx]) == 'nan':
            old_idx = idx
            continue

        current_topics = list(row['topics'])
        continued_topics = [x for x in old_topics if x in current_topics]
        continued_topic = False if len(continued_topics) == 0 else True

        if continued_topic:

            change_in_coords = [0, step_size_y]
            new_sent_coords = list(map(add, old_sent_coords, change_in_coords))

            plt.plot(new_sent_coords[0], new_sent_coords[1], 'o', color='k', ms=3)  # plot node
            plt.plot([old_sent_coords[0], new_sent_coords[0]], [old_sent_coords[1], new_sent_coords[1]], '-',
                     color=colour)

        if not continued_topic:
            new_topic = [x for x in current_topics if x in list(file.topics[idx + 1])]
            topic_linegraph_dict['Idx'].append(idx)
            topic_linegraph_dict['All_Current_Topics'].append(current_topics)
            topic_linegraph_dict['New_Topic'].append(new_topic)
            topic_linegraph_dict['Speaker'].append(row['speaker'])
            topic_linegraph_dict['Sentence'].append(row['utterance'])
            topic_linegraph_dict['DA_Label'].append(row['da_label'])

            change_in_coords = [step_size_x, 0]
            new_sent_coords = list(map(add, old_sent_coords, change_in_coords))


            ## NOW CHECK FOR RETURNS...
            #print('new_topic', new_topic)
            #print(Dict_of_topics)
            # Check if topic has already been visited.
            the_topic = None
            for topic in new_topic:  # NOTE what if new_topic contains >1 topics which have been mentioned in DIFFERENT PLACES??... WHICH X TO GO TO?
                if topic in Dict_of_topics:
                    X_pos, Y_pos = Dict_of_topics[topic]
                    the_topic = topic
                else:
                    continue


            # check if X_pos has been assigned, if not, assign new position
            if the_topic is None:
                for topic in new_topic:  #  in case it contains multiple, but they all get this x position.
                    Dict_of_topics[topic] = new_sent_coords
                    Dict_of_topics_counts[topic] = 1

                # then plot…
                plt.plot(new_sent_coords[0], new_sent_coords[1], 'o', color='k', ms=1)  # plot node
                # plt.annotate(', '.join(current_topics), xy=(new_sent_coords[0]+1, new_sent_coords[1]), color='k',
                #              zorder=100),  # textcoords="offset points" #weight
                print('the_topic', the_topic, '. Current topics:', current_topics, '. New_topic: ',new_topic)


            else: # i.e. if we are jumping back to a previously mentioned topic
                Dict_of_topics_counts[the_topic] += 1
                step_size_x += 0.5  # also adjust step size so don't return to other topics
                new_sent_coords = [X_pos, new_sent_coords[1]]  # keep y position but change x.

                plt.plot(new_sent_coords[0], new_sent_coords[1], 'o', color='k', ms=1)  # plot node
                plt.plot([X_pos, X_pos], [Y_pos, new_sent_coords[1]], '--', color='k', linewidth=1)  #  add dashed line between last one and this one

                # plt.annotate(', '.join(current_topics), xy=(new_sent_coords[0], new_sent_coords[1]), color='k',
                #              zorder=100),  # textcoords="offset points" #weight=
                print('the_topic', the_topic, '. Current topics:', current_topics, '. New_topic: ',new_topic)

                if Dict_of_topics_counts[the_topic] == 2:
                    # Annotate the line
                    plt.annotate(the_topic, xy=(Dict_of_topics[the_topic][0]-3, Dict_of_topics[the_topic][1]+10), color='darkred',
                                 zorder=100, rotation=90, weight='bold'),  # textcoords="offset points" #weight=)

            plt.plot([old_sent_coords[0], new_sent_coords[0]], [old_sent_coords[1], new_sent_coords[1]], '-',
                     color=colour)  # plot line

        old_topics = current_topics #new_topic
        old_sent_coords = new_sent_coords
        old_idx = idx


    print(' the final index of row', idx)
    print('the final y position', new_sent_coords[1])
    # print('Dict_of_topics', Dict_of_topics)
    # print('Dict_of_topics_counts', Dict_of_topics_counts)
    # legend_handles = []
    # legend_labels = []
    # for i in range(len(list(speakers_map.keys()))):
    #     legend_handles.append(Line2D([0], [0], color=list(speakers_map.values())[i], lw=1))
    #     legend_labels.append(list(speakers_map.keys())[i])

    # plt.xlabel('Only Statements in Utterance')
    # plt.ylabel('Question in Utterance')
    # plt.legend(legend_handles, legend_labels)
    if logscalex:
        plt.xscale('log')
    if save_fig:
        plt.savefig("Saved_Images/{0}/DT_Shifting_Topic_Line_LogX:{1}.png".format(transcript_name, logscalex), dpi=600)
    plt.show()
    return


def DT_Firt_Draft_Mess(cutoff_sent=400, Interviewee='jack dorsey', save_fig=False):
    """"
    Shifting Line Topics 2 but new...
    """
    speakers_map = {'joe rogan': 'purple', Interviewee: 'blue'}
    transcript_name = "joe_rogan_{}".format('_'.join(list(speakers_map.keys())[1].split(' ')))
    file = pd.read_pickle("processed_transcripts/{}.pkl".format(transcript_name))
    print(file[:100].to_string())

    step_size_x, step_size_y = 1, 1

    topic_linegraph_dict = {'Idx': [], 'All_Current_Topics': [], 'New_Topic': [], 'Speaker': [], 'Sentence': [],
                            'DA_Label': []}

    plt.figure()
    plt.title('Shifting Topic Line 2: {0}'.format(Interviewee.title()))
    old_sent_coords = [0, 0]
    old_idx = 0
    old_topics, most_recently_plotted = [], ''
    Dict_of_topics, Dict_of_topics_counts = {}, {}
    branch_number = 1
    for idx, row in file[1:cutoff_sent].iterrows():
        old_speaker = file.speaker[old_idx]  # i.e. the speaker who said all utterances between old index and new index\
        colour = cm.copper(branch_number/20) #'k' #speakers_map[old_speaker]
        if str(file.topics[old_idx]) == 'nan':
            old_idx = idx
            continue

        current_topics = list(row['topics'])
        continued_topics = [x for x in old_topics if x in current_topics]
        continued_topic = False if len(continued_topics) == 0 else True

        if continued_topic:
            change_in_coords = [0, 1]
            new_sent_coords = list(map(add, old_sent_coords, change_in_coords))
            ## ADDED IN THIS next LINE
            for top in continued_topics:
                Dict_of_topics[top] = new_sent_coords # think i dont want to do this for all topics in continued_topics, only for the newest one

            plt.plot(new_sent_coords[0], new_sent_coords[1], 'o', color='k', ms=1)  # plot node
            plt.plot([old_sent_coords[0], new_sent_coords[0]], [old_sent_coords[1], new_sent_coords[1]], '-',
                     color=colour)


        if not continued_topic:
            new_topic = [x for x in current_topics if x in list(file.topics[idx + 1])]
            topic_linegraph_dict['Idx'].append(idx)
            topic_linegraph_dict['All_Current_Topics'].append(current_topics)
            topic_linegraph_dict['New_Topic'].append(new_topic)
            topic_linegraph_dict['Speaker'].append(row['speaker'])
            topic_linegraph_dict['Sentence'].append(row['utterance'])
            topic_linegraph_dict['DA_Label'].append(row['da_label'])

            change_in_coords = [step_size_x, 2]                                                     #make things move up more BUT NOTE this removes time element
            new_sent_coords = list(map(add, old_sent_coords, change_in_coords))


            ## NOW CHECK FOR RETURNS...
            #print('new_topic', new_topic)
            #print(Dict_of_topics)
            # Check if topic has already been visited.
            the_topic = None
            for topic in new_topic:  # NOTE what if new_topic contains >1 topics which have been mentioned in DIFFERENT PLACES??... WHICH X TO GO TO?
                if topic in Dict_of_topics:
                    X_pos, Y_pos = Dict_of_topics[topic]
                    the_topic = topic
                else:
                    continue


            # check if X_pos has been assigned, if not, assign new position
            if the_topic is None:
                for topic in new_topic:  #  in case it contains multiple, but they all get this x position.
                    Dict_of_topics[topic] = new_sent_coords
                    Dict_of_topics_counts[topic] = 1

                # then plot…
                plt.plot(new_sent_coords[0], new_sent_coords[1], 'o', color='k', ms=1)  # plot node
                plt.annotate(', '.join(current_topics), xy=(new_sent_coords[0], new_sent_coords[1]), color='k',
                             zorder=100),  # textcoords="offset points" #weight

                #print('the_topic', the_topic, '. Current topics:', current_topics, '. New_topic: ',new_topic)

                plt.plot([old_sent_coords[0], new_sent_coords[0]], [old_sent_coords[1], new_sent_coords[1]], '-',
                         color=colour)  # plot line


            else: # i.e. if we are jumping back to a previously mentioned topic
                #make a mar of the branch number

                ## When WE ARE HERE we are starting a new topic which we HAVE seen before, so we're about to shift away
                # from the top of our previous topic. we want to save the last coordinate of every topic we visit


                print('-branch_number:', branch_number, '. The Topic', the_topic, '. ENDING position:',
                      old_sent_coords[0], old_sent_coords[1])
                plt.plot(old_sent_coords[0], old_sent_coords[1], 'o', color='orange', ms=10)
                plt.rc('font', size=6)
                plt.annotate(branch_number, xy=(old_sent_coords[0], old_sent_coords[1]), color='darkred', zorder=100, weight='bold')
                plt.rc('font', size=8)
                # Update branch number
                branch_number += 1

                ## NOTE SURE ABOUT THIS BIT.....
                try:
                    Dict_of_topics_counts[the_topic] += 1
                except:
                    Dict_of_topics_counts[the_topic] = 1
                    Dict_of_topics_counts[the_topic] += 1
                #

                step_size_x *= -1  # also adjust step size so don't return to other topics
                if step_size_x <0:
                    step_size_x -= 5
                else:
                    step_size_x += 5

                step_size_y += 5
                new_sent_coords = [X_pos, new_sent_coords[1]] #Ypos  # JUMP BACK TO LAST MENTION ???
                print('Branch_number:', branch_number, '. The Topic', the_topic, '. Starting position:',
                      new_sent_coords)

                plt.plot(new_sent_coords[0], new_sent_coords[1], 'o', color='k', ms=Dict_of_topics_counts[the_topic]*3)  # plot node
                plt.plot([X_pos, X_pos], [Y_pos, new_sent_coords[1]], '--', color='k', linewidth=1)  #  add dashed line between last one and this one

                plt.annotate(', '.join(current_topics), xy=(new_sent_coords[0], new_sent_coords[1]), color='k',
                             rotation=90, zorder=100),  # textcoords="offset points" #weight=

                #print('the_topic', the_topic, '. Current topics:', current_topics, '. New_topic: ', new_topic)

                if Dict_of_topics_counts[the_topic] == 2:
                    # Annotate the line
                    plt.annotate(the_topic, xy=(Dict_of_topics[the_topic][0]-0.2, Dict_of_topics[the_topic][1]+9),
                                 color='k', zorder=100, weight='bold')

        old_topics = current_topics #new_topic
        old_sent_coords = new_sent_coords
        old_idx = idx


    print(' the final index of row', idx)
    print('the final y position', new_sent_coords[1])
    # print('Dict_of_topics', Dict_of_topics)
    # print('Dict_of_topics_counts', Dict_of_topics_counts)
    # legend_handles = []
    # legend_labels = []
    # for i in range(len(list(speakers_map.keys()))):
    #     legend_handles.append(Line2D([0], [0], color=list(speakers_map.values())[i], lw=1))
    #     legend_labels.append(list(speakers_map.keys())[i])

    # plt.xlabel('Only Statements in Utterance')
    # plt.ylabel('Question in Utterance')
    # plt.legend(legend_handles, legend_labels)
    if save_fig:
        plt.savefig("Saved_Images/{0}/Simpler_Shifting_Line_Topics.png".format(transcript_name), dpi=600)
    plt.show()
    return


def DT_First_Draft(cutoff_sent=400, Interviewee='jack dorsey', save_fig=False):
    """"
    Function
    NOTE A:
    Here, we have seen that >=1 topic from the previous Utterance is still being spoken about in this utterance
    so we loop through all the topics currently being spoken about and add on the current node position to ALL of their
    dictionary lists.
    Ideally only want to add on the current positions to the topic which is in the next utterance too. i.e. so if
    'people' AND 'twitter' were both carried on, but only twitter is come back to later (and hence only twitter is
    plotted in bold on the map) then only want to save coords for TWITTER. but the issue is we can't know what will be
    come back to before we get to that utterance in a later stage.

    NOTE B:
        No topics were continued between the most recent utterance and the current one. Now we must decide if
        the new topic is a topic we have seen before (loop back), or a brand new one (just take a step sideways).

    NOTE C:
        Extract the topic that's being continued on in the next utterance also - to avoid looking at one that's only
        mentioned once. i.e. only interested in the more long-term topics
    """

    # Upload df containing Topic + Dialogue Act information...
    transcript_name = "joe_rogan_{}".format('_'.join(Interviewee.split(' ')))
    transcript_df = pd.read_pickle("processed_transcripts/{}.pkl".format(transcript_name))
    #transcript_df = pd.read_pickle("/Users/ShonaCW/Downloads/processed_transcripts (2)/29/joe_rogan_kanye_west.pkl")
    print(transcript_df.head(100).to_string())

    # Define some dictionaries and counters we'll need...
    Dict_of_topics, Dict_of_topics_counts, Dict_of_topics_direction = {}, {}, {}
    (step_size_x, step_size_y) = (1, 0)
    old_sent_coords, old_topic, old_current_topics = [0, 0], '', []
    branch_number, old_idx, topic_direction = 0, 0, +1
    topics_with_stacks = []
    single_stacks_appended_to_last_counter = 0

    first_idx_with_a_topic = int(transcript_df.index[transcript_df['topics'].notna()].tolist()[0]) + 1

    # Instantiate figure
    plt.figure()
    plt.title('Basic Discussion Tree: Joe Rogan & {0}'.format(Interviewee.title()))

    # Loop through Utterances in the dataframe...
    for idx, row in transcript_df[1:cutoff_sent].iterrows():
        print('idx: ', idx)
        #colour = cm.copper(branch_number/20)                          # Added a colour map so later branches are lighter
        colour = 'k'
        no_topic = False

        if str(transcript_df.topics[old_idx]) == 'nan':               # Skip past Utterances which have no topic
            old_idx = idx
            print('was nan, looping w no plot')
            continue

        current_topics = list(row['topics'])           #.pop()                  # All topics contained in this Utt
        # print('current_topics', current_topics)
        continued_topics = [x for x in current_topics if x == old_topic]   # Topics continued from previous Utt
        new_topic = [x for x in current_topics if x in list(transcript_df.topics[idx + 1])]  # NOTE C
        continued_topic = False if len(continued_topics) == 0 else True     # False if no topics were continued on

        print('current_topics', current_topics)
        print('new_topic', new_topic)
        print('continued_topic', continued_topic)

        if not continued_topic and len(new_topic) == 0: # i.e. if the name of the stack isnt in the current topics, but the current topics also don't match up with the NEXT ones (a stand-along utterance in a way...
            # in this case we just match it up with the previous stack (if this happens a lot though should just make them their own node point
            # check if any of the current topics were discussed in the last one
            continued_topics = [x for x in current_topics if x in old_current_topics]
            single_stacks_appended_to_last_counter += 1
            # print('single_stacks_appended_to_last_counter', single_stacks_appended_to_last_counter)
            continued_topic = False if len(continued_topics) == 0 else True  # False if no topics were continued on
            no_topic = True
            print('in 1')

        if continued_topic:                 # If continued on topics from last Utterance, just move up the y axis 1 step
            change_in_coords = [0, 1]
            new_sent_coords = list(map(add, old_sent_coords, change_in_coords))

            # print('-continued_topics', continued_topics)
            # print('-topics_with_stacks', topics_with_stacks)
            # print('-the_topic', the_topic)
            # find which of the continued topics the stack is named after and store the values for it in the dicts

            if step_size_x < 0:  # Save the direction in which this branch is travelling
                current_direction = -1
            else:
                current_direction = 1

            if not no_topic:
                the_topic = [topic for topic in continued_topics if topic in topics_with_stacks][0]
                Dict_of_topics[the_topic] = new_sent_coords
                Dict_of_topics_direction[the_topic] = current_direction
                Dict_of_topics_counts[the_topic] += 1

            # for topic in continued_topics: # Note A
            #     try:
            #         # print('Dict_of_topics_heights_climbed[topic] BEFORE', Dict_of_topics_heights_climbed[topic])
            #         # Dict_of_topics_heights_climbed[topic].append(new_sent_coords[1])
            #         Dict_of_topics[topic] = new_sent_coords
            #         Dict_of_topics_direction[topic] = topic_direction
            #         Dict_of_topics_counts[topic] += 1
            #         # print('Dict_of_topics_heights_climbed [topic] AFTER', Dict_of_topics_heights_climbed[topic])
            #     except:
            #         pass
            #         # Dict_of_topics_counts[topic] = 1

            plt.plot(new_sent_coords[0], new_sent_coords[1], 'o', color='k', ms=3)                  # Plot node
            plt.plot([old_sent_coords[0], new_sent_coords[0]], [old_sent_coords[1], new_sent_coords[1]], '-',
                     color=colour)                                                                  # Plot line
            # plt.annotate(continued_topics, xy=(new_sent_coords[0], new_sent_coords[1]))
            print('2')

        elif not continued_topic:                                                                                       # NOTE B
            # print('\nall new topics', new_topic, 'list(transcript_df.topics[idx + 1])', list(transcript_df.topics[idx + 1]))
            # print('checking if it had its own stack before.. (topics_with_stacks):', topics_with_stacks)
            print('3')
            the_topic = None
            for top in new_topic:
                if top in topics_with_stacks: #Dict_of_topics:
                    X_pos, Y_pos = Dict_of_topics[top]
                    topic_direction = Dict_of_topics_direction[top]
                    the_topic = top
                    break
                else:
                    continue
            #print('the_topic chosen (if it had been seen before): ', the_topic)

            ## Here we just want to shift the branch horizontally and start a new stack, as it's a whole new topic
            if the_topic is None:
                print('4')
                change_in_coords = [step_size_x, step_size_y]                       # Shift horizontally and upwards
                if idx == first_idx_with_a_topic:
                    new_sent_coords = old_sent_coords
                else:
                    new_sent_coords = list(map(add, old_sent_coords, change_in_coords))

                the_topic = new_topic[0]
                #print('new_topic', new_topic, ', the topic to name the stack:', the_topic)
                topics_with_stacks.append(the_topic)

                Dict_of_topics[the_topic] = new_sent_coords
                Dict_of_topics_counts[the_topic] = 1

                if step_size_x < 0:                      # Save the direction in which this branch is travelling
                    current_direction = -1
                else:
                     current_direction = 1

                Dict_of_topics_direction[the_topic] = current_direction

                # Plot: continuing on the same branch, but with a new position to mark a new set of topics
                plt.plot(new_sent_coords[0], new_sent_coords[1], 'o', color=colour, ms=1)                   # Plot node
                plt.plot([old_sent_coords[0], new_sent_coords[0]], [old_sent_coords[1], new_sent_coords[1]], '-',
                         color=colour)
                # Plot line
                plt.rc('font', size=7)
                plt.annotate(the_topic, xy=(new_sent_coords[0] + 0.2*topic_direction, new_sent_coords[1]+1), color='k',
                             rotation = 90, zorder=100, weight='bold') # If want info plotted #f'Current topics: {current_topics}, new_topic: {new_topic}, topic: {the_topic}'
                plt.rc('font', size=8)

            ## Here we are starting a new branch at the position of the topic we've jumped back to
            else:
                print('5')


                plt.plot(old_sent_coords[0], old_sent_coords[1], 'o', color='orange', ms=8)
                plt.rc('font', size=5)
                plt.annotate(branch_number, xy=(old_sent_coords[0], old_sent_coords[1]), color='darkred', zorder=100,
                             weight='bold')
                plt.rc('font', size=8)

                # New branch, and update step sizes
                branch_number += 1
                print('Branch Number: ', branch_number)
                step_size_y += 1

                if topic_direction > 0: # if the branch was moving positively last time, we want to go negative
                    step_size_x = -1 # make it negative
                    step_size_x -= 0.01*branch_number # increase increment
                    topic_direction_updated = -1
                elif topic_direction < 0:
                    step_size_x = 1 # make it positive
                    step_size_x += 0.01*branch_number # increase increment
                    topic_direction_updated = 1

                new_sent_coords = [X_pos, Y_pos] #new_sent_coords[1]]  # + Dict_of_topics_heights_climbed[the_topic][-1]
                Dict_of_topics[the_topic] = new_sent_coords
                Dict_of_topics_direction[the_topic] = topic_direction_updated
                Dict_of_topics_counts[the_topic] += 1

                #print(f'-NEW BRANCH {branch_number} CREATED at topic {the_topic}. topic_direction_updated', topic_direction_updated)

                # sublists_height = []
                # for group in mit.consecutive_groups(list(Dict_of_topics_heights_climbed[the_topic])):
                #     sublists_height.append(list(group))
                # height_since_start_list = [x[-1] - x[0] for x in sublists_height]
                # height = sublists_height[0][0]
                # height_to_add = sum(height_since_start_list)

                # print('\nthe_topic', the_topic)
                # print('branch_number:', branch_number)
                # print('Dict_of_topics[the_topic]', Dict_of_topics[the_topic])
                # print('Y_pos', Y_pos)
                #
                # print('Dict_of_topics_heights_climbed[the_topic]', Dict_of_topics_heights_climbed[the_topic])
                # print('sublist', sublists_height)
                # print('height_since_start_list', height_since_start_list)
                # print('height of [0][0]', height)
                # print('height_to_add', height_to_add)

                # Plot...
                #plt.plot(new_sent_coords[0], new_sent_coords[1], 'o', color='green', ms=5)   # Plot branch-starting node
                plt.plot(new_sent_coords[0], new_sent_coords[1], 'o', color='k', ms=1)                  # Plot node
                plt.plot([X_pos, X_pos], [Y_pos, new_sent_coords[1]], '--', color='k', linewidth=1)     # Dashed line

                # plt.annotate(the_topic, xy=(new_sent_coords[0], new_sent_coords[1]), color='k', zorder=100)

                # if Dict_of_topics_counts[the_topic] == 2:
                #     plt.annotate(the_topic , xy=(Dict_of_topics[the_topic][0], Dict_of_topics[the_topic][1]),
                #                  color='k', zorder=100, rotation=90, weight='bold') # Annotate the line
                print('branch number', branch_number, 'the_topic', the_topic)

                #topics_with_stacks.append(the_topic)

        old_topic = the_topic # current_topics #new_topic
        old_current_topics = current_topics
        old_sent_coords = new_sent_coords
        old_idx = idx

    print('single_stacks_appended_to_last_counter: ', single_stacks_appended_to_last_counter)

    # print('Dict_of_topics', Dict_of_topics)
    # print('Dict_of_topics_counts', Dict_of_topics_counts)
    # legend_handles = []
    # legend_labels = []
    # for i in range(len(list(speakers_map.keys()))):
    #     legend_handles.append(Line2D([0], [0], color=list(speakers_map.values())[i], lw=1))
    #     legend_labels.append(list(speakers_map.keys())[i])

    # plt.xlabel('Only Statements in Utterance')
    # plt.ylabel('Question in Utterance')
    # plt.legend(legend_handles, legend_labels)
    plt.ylim(top=110)
    if save_fig:
        plt.savefig("Saved_Images/{0}/Discussion_Tree1.png".format(transcript_name), dpi=600)
    plt.show()
    return

def Choose_Topics(new_topics, nlp):
    """
    Function to decide which topic word should label the stack.
    Saves the stack name to a new column in the df.
    Once I've run it here can then plot the word embedding trajectory and the DT using it.
    """
    # First of all check if there is only one new_topic
    only_one = True if len(new_topics) == 1 else False
    # print('IN CHOOSE TOPICS, new_topics:', new_topics)

    new_topics_backup = new_topics.copy()

    if only_one:
        the_topic = new_topics[0]
        # print('only one new topic')

    else:
        # nlp = spacy.load("en_core_web_sm")              #en_core_web_sm   #en_core_web_lg# do this outside of this function to speed up time

        # Remove words which are not nouns or proper nouns
        tokens = [nlp(word)[0] for word in new_topics]

        #all_pairs = [(token.text, token.pos_) for token in tokens]      # if interested later
        only_nouns = [token.text for token in tokens if token.pos_ in ['NOUN', 'PROPN']]

        new_topics = only_nouns

        # Remove plural words
        to_remove = []
        for i in new_topics:
            for j in new_topics:
                if i == j:
                    continue
                else:
                    if j == i+'s' or j == i+'es':
                        to_remove.append(j) #remove j (the plural version) from the list
                    else:
                        continue

        # print('to_remove:', to_remove)
        new_topics = [x for x in new_topics if x not in to_remove]

        # check if we now have only one option
        if len(new_topics) == 1:
            the_topic = new_topics[0]
            # print('after removing plurals, only one new topic')

        else:
            # Check if there are any bigrams
            bigrams = [x for x in new_topics if '_' in x]
            if len(bigrams) == 0:
                # If no bigrams, we now just choose the LONGEST noun/ proper noun
                try:
                    the_topic = max(new_topics, key=len)
                except:
                    # i.e. if there are NO new_topics from when we only considered nouns.. i.e. if new_topics=['29', '30']
                    the_topic = new_topics_backup[0]

            else:
                # print('bigrams to choose from:', bigrams)
                the_topic = bigrams[0]


    #the_topic = new_topics[0] #for now...
    # print('the_topic chosen:', the_topic)

    return the_topic




def DT_Second_Draft(path, podcast_name, cutoff_sent=-1, save_fig=False, info=False):
    """
    Work with sets of topics
    Access transcript dfs from newer collection (of 20k+) from preprocessed Spotify Podcast dataset.

    Want to be able to call this function when looping through LOTS of different podcasts, i.e. no longer just considering
    Joe Rogan podcasts.


    NOTE Z:
        In here if the name of the stack isnt in the current topics, but the current topics also don't match up with the
        NEXT ones (a stand-along utterance in a way... in this case we just match it up with the previous stack
        (if this happens a lot though should just make them their own node point and? check if any of the current topics
        were discussed in the last one
    """
    # Upload df containing Topic + Dialogue Act information...
    transcript_name = str(path).split("/spotify_", 1)[1][:-4]
    print('Building DT for...', transcript_name)

    #transcript_df = pd.read_pickle(f"/Users/ShonaCW/Downloads/processed_transcripts (2)/{folder_number}/{transcript_name}.pkl")
    transcript_df = pd.read_pickle(path)
    # print(transcript_df.head(100).to_string())                                        # If want

    # Define some dictionaries and counters we'll need...
    Dict_of_topics, Dict_of_topics_counts, Dict_of_topics_direction = {}, {}, {}
    (step_size_x, step_size_y) = (1, 0)
    old_sent_coords, old_topic, old_current_topics = [0, 0], '', []
    branch_number, topic_direction = 0, +1
    topics_with_stacks = []
    single_stacks_appended_to_last_counter = 0
    first_idx_with_a_topic = int(transcript_df.index[transcript_df['topics'].astype(bool)].tolist()[0])

    if info:
        print(transcript_df.head(100).to_string())
        print('\nfirst_idx_with_a_topic:', first_idx_with_a_topic)

    # Deal with leaf colours quartile-wise
    Num_Total_Utts = len(transcript_df)
    Quartiles = [i for i in split_segs(range(Num_Total_Utts), 4)]
    Colours_Dict = {0: ['palegreen', 5], 1: ['lawngreen', 4], 2: ['forestgreen', 4], 3: ['darkgreen', 4]}

    nlp = spacy.load("en_core_web_sm") ####

    # Instantiate figure
    plt.figure()
    plt.title('Discussion Tree: {}'.format(transcript_name))

    # Loop through Utterances in the dataframe...
    for idx, row in transcript_df[1:cutoff_sent].iterrows():
        quartile = next(i for i, v in enumerate(Quartiles) if idx in v)
        colour_leaves = Colours_Dict[quartile][0] #cm.YlOrRd(branch_number/20)       # Added a colour map so later branches are lighter
        size_leaves = Colours_Dict[quartile][1]
        colour = 'k'
        colour_label = 'k' #'saddlebrown'
        no_topic = False

        set_of_topics = row['topics']
        current_topics = [list(x) for x in set_of_topics if x]               # All topics contained in this Utt #pop if set is not empty

        if idx < 20 and len(current_topics) == 0:                      #Sometimes the first few Utterances have no topic
            if info:
                print('Skipped idx due to no topics')
            continue

        current_topics = [item for sublist in current_topics for item in sublist]
        next_topics = [list(x) for x in transcript_df.topics[idx + 1] if x]
        next_topics = [item for sublist in next_topics for item in sublist]

        continued_topics = [x for x in current_topics if x == old_topic]        # Topics continued from previous Utt
        new_topic = [x for x in current_topics if x in next_topics]  # NOTE C
        continued_topic = False if len(continued_topics) == 0 else True         # False if no topics were continued on

        if info:
            print('\nidx: ', idx)
            print('current_topics', current_topics)
            print('continued_topics', continued_topics)
            print('new_topic', new_topic)
            print('continued_topic', continued_topic)

        if not continued_topic and len(new_topic) == 0:
            if len(old_current_topics) == 0:
                #ie if this is the FIRST line!
                new_topic = current_topics.copy()
            else:
                #if this isn't the first line. Note Z
                continued_topics = [x for x in current_topics if x in old_current_topics]
                single_stacks_appended_to_last_counter += 1
                # print('single_stacks_appended_to_last_counter', single_stacks_appended_to_last_counter)
                continued_topic = False if len(continued_topics) == 0 else True  # False if no topics were continued on
                no_topic = True

        if continued_topic:                 # If continued on topics from last Utterance, just move up the y axis 1 step
            change_in_coords = [0, 1]
            new_sent_coords = list(map(add, old_sent_coords, change_in_coords))

            # print('-continued_topics', continued_topics)
            # print('-topics_with_stacks', topics_with_stacks)
            # print('-the_topic', the_topic)
            # find which of the continued topics the stack is named after and store the values for it in the dicts

            if step_size_x < 0:  # Save the direction in which this branch is travelling
                current_direction = -1
            else:
                current_direction = 1

            if not no_topic:
                the_topic = [topic for topic in continued_topics if topic in topics_with_stacks][0]
                Dict_of_topics[the_topic] = new_sent_coords
                Dict_of_topics_direction[the_topic] = current_direction
                # Dict_of_topics_counts[the_topic] += 1

            # for topic in continued_topics: # Note A
            #     try:
            #         # print('Dict_of_topics_heights_climbed[topic] BEFORE', Dict_of_topics_heights_climbed[topic])
            #         # Dict_of_topics_heights_climbed[topic].append(new_sent_coords[1])
            #         Dict_of_topics[topic] = new_sent_coords
            #         Dict_of_topics_direction[topic] = topic_direction
            #         Dict_of_topics_counts[topic] += 1
            #         # print('Dict_of_topics_heights_climbed [topic] AFTER', Dict_of_topics_heights_climbed[topic])
            #     except:
            #         pass
            #         # Dict_of_topics_counts[topic] = 1

            plt.plot(new_sent_coords[0], new_sent_coords[1], 'o', color=colour, ms=3, zorder=0)              # Plot node
            plt.plot([old_sent_coords[0], new_sent_coords[0]], [old_sent_coords[1], new_sent_coords[1]], '-',
                     color=colour, linewidth=1, zorder=0)                                                 # Plot line
            # plt.annotate(continued_topics, xy=(new_sent_coords[0], new_sent_coords[1]))

        elif not continued_topic:                                                                                       # NOTE B
            # print('\nall new topics', new_topic, 'list(transcript_df.topics[idx + 1])', list(transcript_df.topics[idx + 1]))
            # print('checking if it had its own stack before.. (topics_with_stacks):', topics_with_stacks)
            the_topic = None
            for top in new_topic:
                if top in topics_with_stacks: #Dict_of_topics:
                    X_pos, Y_pos = Dict_of_topics[top]
                    topic_direction = Dict_of_topics_direction[top]
                    the_topic = top
                    break
                else:
                    continue
            #print('the_topic chosen (if it had been seen before): ', the_topic)

            ## Here we just want to shift the branch horizontally and start a new stack, as it's a whole new topic
            if the_topic is None:
                change_in_coords = [step_size_x, step_size_y]                       # Shift horizontally and upwards
                if idx == first_idx_with_a_topic:
                    new_sent_coords = old_sent_coords
                else:
                    new_sent_coords = list(map(add, old_sent_coords, change_in_coords))

                the_topic = Choose_Topics(new_topic, nlp) #Choose_Topics(new_topic, nlp) #new_topic[0]  # change this eventually, this is an arbitary choice for the stack name

                topics_with_stacks.append(the_topic)
                Dict_of_topics[the_topic] = new_sent_coords
                Dict_of_topics_counts[the_topic] = 1

                if step_size_x < 0:                      # Save the direction in which this branch is travelling
                    current_direction = -1
                else:
                     current_direction = 1

                Dict_of_topics_direction[the_topic] = current_direction

                # Plot: continuing on the same branch, but with a new position to mark a new set of topics
                plt.plot(new_sent_coords[0], new_sent_coords[1], 'o', color=colour, ms=3, zorder=0)     # Plot node
                plt.plot([old_sent_coords[0], new_sent_coords[0]], [old_sent_coords[1], new_sent_coords[1]], '-',
                         color=colour, linewidth=1, zorder=0)

                # Annotate each stack
                # plt.rc('font', size=6)
                # plt.annotate(the_topic, xy=(new_sent_coords[0] + 0.2*topic_direction, new_sent_coords[1]+1), color='k',
                #              rotation = 90, zorder=100) #, weight='bold')
                # # for above: If want info plotted #f'Current topics: {current_topics}, new_topic: {new_topic}, topic: {the_topic}'
                # plt.rc('font', size=8)

            ## Here we are starting a new branch at the position of the topic we've jumped back to
            else:
                # Plot and annotate little orange dots indicating the number of branch which just ended
                plt.plot(old_sent_coords[0], old_sent_coords[1], 'o', color=colour_leaves, ms=size_leaves, zorder=100)
                plt.rc('font', size=7) #size_leaves
                plt.annotate(branch_number, xy=(old_sent_coords[0], old_sent_coords[1]), color='saddlebrown', zorder=101,
                             weight='bold')
                plt.rc('font', size=8)

                # New branch, and update step sizes
                branch_number += 1
                step_size_y += 1

                if topic_direction > 0: # if the branch was moving positively last time, we want to go negative
                    step_size_x = -1 # make it negative
                    step_size_x -= 0.01*branch_number # increase increment
                    topic_direction_updated = -1
                elif topic_direction < 0:
                    step_size_x = 1 # make it positive
                    step_size_x += 0.01*branch_number # increase increment
                    topic_direction_updated = 1

                new_sent_coords = [X_pos, Y_pos] #new_sent_coords[1]]  # + Dict_of_topics_heights_climbed[the_topic][-1]
                Dict_of_topics[the_topic] = new_sent_coords
                Dict_of_topics_direction[the_topic] = topic_direction_updated
                Dict_of_topics_counts[the_topic] += 1

                #print(f'-NEW BRANCH {branch_number} CREATED at topic {the_topic}. topic_direction_updated', topic_direction_updated)

                # sublists_height = []
                # for group in mit.consecutive_groups(list(Dict_of_topics_heights_climbed[the_topic])):
                #     sublists_height.append(list(group))
                # height_since_start_list = [x[-1] - x[0] for x in sublists_height]
                # height = sublists_height[0][0]
                # height_to_add = sum(height_since_start_list)

                # print('\nthe_topic', the_topic)
                # print('branch_number:', branch_number)
                # print('Dict_of_topics[the_topic]', Dict_of_topics[the_topic])
                # print('Y_pos', Y_pos)
                #
                # print('Dict_of_topics_heights_climbed[the_topic]', Dict_of_topics_heights_climbed[the_topic])
                # print('sublist', sublists_height)
                # print('height_since_start_list', height_since_start_list)
                # print('height of [0][0]', height)
                # print('height_to_add', height_to_add)

                # Plot...
                #plt.plot(new_sent_coords[0], new_sent_coords[1], 'o', color='green', ms=5)   # Plot branch-starting node
                plt.plot(new_sent_coords[0], new_sent_coords[1], 'o', color=colour, ms=3, zorder=0)                  # Plot node

                # plt.annotate(the_topic, xy=(new_sent_coords[0], new_sent_coords[1]), color='k', zorder=100)

                if Dict_of_topics_counts[the_topic] == 2:
                    plt.annotate(the_topic, xy=(Dict_of_topics[the_topic][0]+0.3, Dict_of_topics[the_topic][1]),
                                 color=colour_label, zorder=150, rotation=0) # weight='bold' #Annotate the line
                #print('branch number', branch_number, 'the_topic', the_topic)

                #topics_with_stacks.append(the_topic)

        old_topic = the_topic # current_topics #new_topic
        old_current_topics = current_topics
        old_sent_coords = new_sent_coords
        # print('idx: ', idx)
        # print('     ', the_topic)

    # print('single_stacks_appended_to_last_counter: ', single_stacks_appended_to_last_counter)

    # Indicate which node finishes off the final branch
    plt.plot(old_sent_coords[0], old_sent_coords[1], 'o', color=colour_leaves, ms=size_leaves)
    plt.rc('font', size=7)
    plt.annotate(branch_number, xy=(old_sent_coords[0], old_sent_coords[1]), color='red', zorder=100,
                 weight='bold')
    plt.rc('font', size=8)

    # print('Dict_of_topics', Dict_of_topics)
    # print('Dict_of_topics_counts', Dict_of_topics_counts)
    # legend_handles = []
    # legend_labels = []
    # for i in range(len(list(speakers_map.keys()))):
    #     legend_handles.append(Line2D([0], [0], color=list(speakers_map.values())[i], lw=1))
    #     legend_labels.append(list(speakers_map.keys())[i])

    # plt.xlabel('Only Statements in Utterance')
    # plt.ylabel('Question in Utterance')
    # plt.legend(legend_handles, legend_labels)
    if save_fig:
        if not os.path.exists('Spotify_Podcast_DataSet/{0}/{1}'.format(podcast_name, transcript_name)):
            os.makedirs('Spotify_Podcast_DataSet/{0}/{1}'.format(podcast_name, transcript_name))

        plt.savefig("Spotify_Podcast_DataSet/{0}/{1}/{1}_DT.png".format(podcast_name, transcript_name), dpi=600)
    plt.show()

    return


def DT_Backbone(path, podcast_name, transcript_name, info=False):
    """
    Function which encapsulates the backbone of DT_Second_draft

    """
    # LOAD df containing Topic + Dialogue Act information...
    print('Collecting backbone info for: ', transcript_name)

    transcript_df = pd.read_pickle(path)

    # create new column we will fill
    Num_Total_Utts = len(transcript_df)
    transcript_df['stack_name'] = [None] * Num_Total_Utts
    transcript_df['branch_num'] = [None] * Num_Total_Utts
    transcript_df['position_X'] = [None] * Num_Total_Utts
    transcript_df['position_Y'] = [None] * Num_Total_Utts
    transcript_df['word_embedding_X'] = [None] * Num_Total_Utts
    transcript_df['word_embedding_Y'] = [None] * Num_Total_Utts
    transcript_df['leaf_colour'] = [None] * Num_Total_Utts

    transcript_df['new_topic'] = [False] * Num_Total_Utts
    transcript_df['new_branch'] = [False] * Num_Total_Utts


    # Define some dictionaries and counters we'll need...
    Dict_of_topics, Dict_of_topics_counts, Dict_of_topics_direction = {}, {}, {}
    (step_size_x, step_size_y) = (1, 3)

    old_sent_coords, old_topic, old_current_topics = [0, 0], '', []
    branch_number, topic_direction = 0, +1
    topics_with_stacks = []
    single_stacks_appended_to_last_counter = 0
    first_idx_with_a_topic = int(transcript_df.index[transcript_df['topics'].astype(bool)].tolist()[0])

    # Deal with leaf colours quartile-wise
    Quartiles = [i for i in split_segs(range(Num_Total_Utts), 4)]
    Colours_Dict = {0: ['palegreen', 5], 1: ['lawngreen', 4], 2: ['forestgreen', 4], 3: ['darkgreen', 4]}

    nlp = spacy.load("en_core_web_sm")  ####

    # Loop through Utterances in the dataframe...
    for idx, row in transcript_df.iterrows():
        quartile = next(i for i, v in enumerate(Quartiles) if idx in v)
        colour_leaves = Colours_Dict[quartile][0]  # cm.YlOrRd(branch_number/20)   # Added so later branches are lighter
        size_leaves = Colours_Dict[quartile][1]
        colour = 'k'
        no_topic = False

        set_of_topics = row['topics']
        current_topics = [list(x) for x in set_of_topics if x]  # All topics contained in this Utt

        if idx < 20 and len(current_topics) == 0:  # Sometimes the first few Utterances have no topic
            if info:
                print('Skipped idx due to no topics')
            continue

        current_topics = [item for sublist in current_topics for item in sublist]
        continued_topics = [x for x in current_topics if x == old_topic]  # Topics continued from previous Utt


        if idx == Num_Total_Utts-1: # i.e. no 'next' topics
            new_topic = current_topics

        else:
            next_topics = [list(x) for x in transcript_df.topics[idx + 1] if x]
            next_topics = [item for sublist in next_topics for item in sublist]

            new_topic = [x for x in current_topics if x in next_topics]  # NOTE C

        continued_topic = False if len(continued_topics) == 0 else True  # False if no topics were continued on

        if info:
            print('\nidx: ', idx)
            print('current_topics', current_topics)
            print('continued_topics', continued_topics)
            print('new_topic', new_topic)
            print('continued_topic', continued_topic)

        if not continued_topic and len(new_topic) == 0:
            if len(old_current_topics) == 0:
                # ie if this is the FIRST line!
                new_topic = current_topics.copy()
            else:
                # if this isn't the first line. Note Z
                continued_topics = [x for x in current_topics if x in old_current_topics]
                single_stacks_appended_to_last_counter += 1
                continued_topic = False if len(continued_topics) == 0 else True  # False if no topics were continued on
                no_topic = True

        if continued_topic:  # If continued on topics from last Utterance, just move up the y axis 1 step
            change_in_coords = [0, 1]
            new_sent_coords = list(map(add, old_sent_coords, change_in_coords))

            if step_size_x < 0:  # Save the direction in which this branch is travelling
                current_direction = -1
            else:
                current_direction = 1

            if not no_topic:
                the_topic = [topic for topic in continued_topics if topic in topics_with_stacks][0]
                Dict_of_topics[the_topic] = new_sent_coords
                Dict_of_topics_direction[the_topic] = current_direction

            # not updating leaf colour as this is not the end of a stack (and hence doesn't have a leaf)
            transcript_df.loc[idx, 'stack_name'] = the_topic
            transcript_df.loc[idx, 'position_X'] = new_sent_coords[0]
            transcript_df.loc[idx, 'position_Y'] = new_sent_coords[1]

        elif not continued_topic:  # NOTE B
            the_topic = None
            for top in new_topic:
                if top in topics_with_stacks:  # Dict_of_topics:
                    X_pos, Y_pos = Dict_of_topics[top]
                    topic_direction = Dict_of_topics_direction[top]
                    the_topic = top
                    break
                else:
                    continue

            ## Here we just want to shift the branch horizontally and start a new stack, as it's a whole new topic
            if the_topic is None:
                #Dict_of_topics_counts[the_topic] = 3 # instantiate it at 3
                #step_size_y = Dict_of_topics_counts[the_topic]
                change_in_coords = [step_size_x, step_size_y]  # Shift horizontally and upwards
                if idx == first_idx_with_a_topic:
                    new_sent_coords = old_sent_coords
                else:
                    new_sent_coords = list(map(add, old_sent_coords, change_in_coords))

                the_topic = Choose_Topics(new_topic, nlp)

                topics_with_stacks.append(the_topic)
                Dict_of_topics[the_topic] = new_sent_coords
                Dict_of_topics_counts[the_topic] = 1

                if step_size_x < 0:  # Save the direction in which this branch is travelling
                    current_direction = -1
                else:
                    current_direction = 1

                Dict_of_topics_direction[the_topic] = current_direction

                transcript_df.loc[idx, 'stack_name'] = the_topic
                transcript_df.loc[idx, 'position_X'] = new_sent_coords[0]
                transcript_df.loc[idx, 'position_Y'] = new_sent_coords[1]
                transcript_df.loc[idx, 'new_topic'] = True

            ## Here we are starting a new branch at the position of the topic we've jumped back to
            else:
                branch_number += 1

                if topic_direction > 0:  # if the branch was moving positively last time, we want to go negative
                    step_size_x = -1  # make it negative
                    step_size_x -= 0.01 * branch_number  # increase increment
                    topic_direction_updated = -1
                elif topic_direction < 0:
                    step_size_x = 1  # make it positive
                    step_size_x += 0.01 * branch_number  # increase increment
                    topic_direction_updated = 1

                new_sent_coords = [X_pos, Y_pos]
                Dict_of_topics[the_topic] = new_sent_coords
                Dict_of_topics_direction[the_topic] = topic_direction_updated
                Dict_of_topics_counts[the_topic] += 1
                #step_size_y += 1

                transcript_df.loc[idx-1, 'leaf_colour'] = colour_leaves

                transcript_df.loc[idx, 'stack_name'] = the_topic
                transcript_df.loc[idx, 'position_X'] = new_sent_coords[0]
                transcript_df.loc[idx, 'position_Y'] = new_sent_coords[1]
                transcript_df.loc[idx, 'new_topic'] = True
                transcript_df.loc[idx, 'new_branch'] = True
                transcript_df.loc[idx, 'branch_num'] = branch_number

        old_topic = the_topic
        old_current_topics = current_topics
        old_sent_coords = new_sent_coords

    # Create new hdf file for given podcast
    try:
        if not os.path.exists('Spotify_Podcast_DataSet_/{0}/{1}'.format(podcast_name, transcript_name)):
            os.makedirs('Spotify_Podcast_DataSet/{0}/{1}'.format(podcast_name, transcript_name))
    except OSError:
        pass

    transcript_df.to_hdf('Spotify_Podcast_DataSet/{0}/{1}/transcript_df.h5'.format(podcast_name, transcript_name),
                         key='df', mode='w')

    print('Saved DF to file')
    if info:
        print(transcript_df.head(-200).to_string())

    return

def Create_ConceptNet_TSNE(podcast_name, transcript_name, configfiles):
    """"""
    # if info:
    #     print('Accessing all episodes for the given show...')

    #configfiles = list(Path("/Users/ShonaCW/Downloads/processed_transcripts (2)/").rglob("**/spotify_{}_*.pkl".format(podcast_name)))
    # if not podcast_name == 'joe_rogan':
    #     configfiles = list(Path("/Users/ShonaCW/Downloads/processed_transcripts (2)/").rglob(
    #         "**/spotify_{}_*.pkl".format(podcast_name)))
    # else:
    #     configfiles = list(Path("/Users/ShonaCW/Downloads/processed_transcripts (2)/").rglob("**/joe_rogan_*.pkl"))

    num_podcasts = len(configfiles)
    print('Number of "{0}" podcasts found: {1}'.format(podcast_name, num_podcasts))

    # Collect all topic keywords from entire podcast show
    all_topical_keywords = []
    for path in configfiles:
        if podcast_name=='joe_rogan':
            transcript_name = str(path).split("/joe_rogan_", 1)[1][:-4]
        else:
            transcript_name = str(path).split("/spotify_", 1)[1][:-4]

        transcript_df = pd.read_hdf('Spotify_Podcast_DataSet/{0}/{1}/transcript_df.h5'.format(podcast_name, transcript_name), key='df')
        topics = list(transcript_df[transcript_df['new_topic'] == True].stack_name)

        # # extract/ save all backbone info
        all_topical_keywords.append(topics)

    print('all_topical_keywords', all_topical_keywords)
    # Remove repeats
    all_topical_keywords = list(dict.fromkeys([item for sublist in all_topical_keywords for item in sublist]))

    # Now find ConceptNet embeddings and reduce using TSNE...
    words_found, reduced_vectors_X, reduced_vectors_Y = get_ConceptNet(all_topical_keywords)
    print('reduced_vectors_X', reduced_vectors_X)

    # Save embeddings to one df
    ConceptNet_TSNE_df = pd.DataFrame(columns=['Topics', 'X', 'Y'])
    ConceptNet_TSNE_df['Topics'] = all_topical_keywords
    ConceptNet_TSNE_df['X'] = reduced_vectors_X
    ConceptNet_TSNE_df['Y'] = reduced_vectors_Y

    # Make sure a folder is set up in which we can save the Info
    if not os.path.exists('Spotify_Podcast_DataSet/{0}'.format(podcast_name)):
        os.makedirs('Spotify_Podcast_DataSet/{0}'.format(podcast_name))

    ConceptNet_TSNE_df.to_hdf('Spotify_Podcast_DataSet/{0}/ConceptNet_Numberbatch_TSNE.h5'.format(podcast_name), key='df')
    # And save embeddings for all the stack_labels in each particular episode
    all_words = list(ConceptNet_TSNE_df['Topics'])
    cnt = 0
    for path in configfiles:
        if podcast_name=='joe_rogan':
            transcript_name = str(path).split("/joe_rogan_", 1)[1][:-4]
        else:
            transcript_name = str(path).split("/spotify_", 1)[1][:-4]

        transcript_df = pd.read_hdf(
            'Spotify_Podcast_DataSet/{0}/{1}/transcript_df.h5'.format(podcast_name, transcript_name), key='df')

        # Now assign word embedding position to utterances based on their topics
        # (or no position if they dont contain a topical keyword)
        for idx, row in transcript_df.iterrows():
            if str(row['stack_name']) in all_words:
                i = all_words.index(str(row['stack_name']))
                transcript_df.loc[idx, 'word_embedding_X'] = list(ConceptNet_TSNE_df['X'])[i]  # assign relevant embedding
                transcript_df.loc[idx, 'word_embedding_Y'] = list(ConceptNet_TSNE_df['Y'])[i]  # assign relevant embedding
            else:
                continue

        # saving again, this time with the reduced word embeddings
        transcript_df.to_hdf('Spotify_Podcast_DataSet/{0}/{1}/transcript_df.h5'.format(podcast_name, transcript_name),
                             key='df', mode='w')

        print('Saved embeddings for file number', cnt+1, '/', num_podcasts, ' : ', str(path))
        cnt += 1

    return

def get_ConceptNet(word_list, info=False):

    # load ConceptNet file into dataframe
    ConceptNet_df = pd.read_hdf('ConceptNet/en_mini_conceptnet.h5', "data")

    words_found, vectors, idxs_of_missing_words = [], [], []

    # Search for words
    for idx, word in enumerate(word_list):
        try:
            vectors.append(ConceptNet_df.loc[word].values)    # embeddings_dict[word])
            words_found.append(word)
        except:
            idxs_of_missing_words.append(idx)
            continue
    if info:
        print('Num words in word_list: ', len(word_list))
        print('Num words with vectors: ', len(words_found))
        print('len(vectors)', len(vectors))
        print('Words not found:', [word for word in word_list if word not in words_found])

        #now perform dimensionality reduction
        print('Performing TSNE dimensionality reduction')
    tsne = TSNE(n_components=2, random_state=0)
    reduced_vectors = list(tsne.fit_transform([l.tolist() for l in vectors]) ) # Note B

    # Now add back in placeholders for the missing words
    for idx in idxs_of_missing_words:
        reduced_vectors.insert(idx, [None, None])

    reduced_vectors = np.array(reduced_vectors)
    reduced_vectors_X, reduced_vectors_Y = reduced_vectors[:, 0], reduced_vectors[:, 1]

    return words_found, reduced_vectors_X, reduced_vectors_Y





def DT_Third_Draft(podcast_name, transcript_name, cutoff_sent=-1, save_fig=False, info=False):
    """
    Function to plot Discussion Trees using backbone data, rather than from scratch

    transcript_df['stack_name']     = [None] * Num_Total_Utts
    transcript_df['branch_num']     = [None] * Num_Total_Utts
    transcript_df['position_X']     = [None] * Num_Total_Utts
    transcript_df['position_Y']     = [None] * Num_Total_Utts
    transcript_df['word_embedding_X'] = [None] * Num_Total_Utts
    transcript_df['word_embedding_Y'] = [None] * Num_Total_Utts
    transcript_df['leaf_colour']        = [None] * Num_Total_Utts
    transcript_df['new_topic']          = [False] * Num_Total_Utts
    transcript_df['new_branch']      = [False] * Num_Total_Utts"""
    import statistics

    #load relevant df
    pod_df = pd.read_hdf('Spotify_Podcast_DataSet/{0}/{1}/transcript_df.h5'.format(podcast_name, transcript_name), key='df')
    colour = 'k'        # colour of tree structure
    colour_label = 'k'  # colour of annotations

    old_sent_coords = [0, 0]
    annotations = []

    # Calculate the height of each stack
    idx_of_new_branch = list(pod_df[pod_df['new_topic'] == True].index)
    idx_of_new_branch.insert(0, 0)
    idx_of_new_branch.insert(-1, len(pod_df))
    #height_of_stack = [idx_of_new_branch[i+1] - idx_of_new_branch[i] for i in range(len(idx_of_new_branch)-1)]

    Topics_Utterances = dict(pod_df['stack_name'].value_counts())
    lists = sorted(Topics_Utterances.items(), key=lambda kv: kv[1])
    words, usage = zip(*lists)

    # Calculate mean / median stack height (only going to label long ones)
    med = statistics.median(usage)
    mean = statistics.mean(usage)

    # Instantiate figure
    plt.figure()
    plt.title('Discussion Tree: {}'.format(transcript_name))
    plt.rc('font', size=7)
    for idx, row in pod_df[0:cutoff_sent].iterrows():
        the_topic = row['stack_name']

        if not the_topic:
            continue

        branch_num = row['branch_num']
        x = row['position_X']
        y = row['position_Y']
        new_topic = row['new_topic']
        new_branch = row['new_branch']

        plt.plot(x, y, 'o', color=colour, ms=3, zorder=0)  # Plot node
        if not new_branch:
            # Plot: continuing on the same branch, but with a new position to mark a new set of topics
            plt.plot([old_sent_coords[0], x], [old_sent_coords[1], y], '-', color=colour, linewidth=1, zorder=0)

        else:
            # Annotate last position with a leaf + branch number label
            leaf_colour = pod_df.iloc[idx-1]['leaf_colour']
            # Plot and annotate little orange dots indicating the number of branch which just ended
            plt.plot(old_sent_coords[0], old_sent_coords[1], 'o', color=leaf_colour, zorder=100) #ms=size_leaves,
            # plt.rc('font', size=7)  # size_leaves
            plt.annotate(branch_num-1, xy=(old_sent_coords[0], old_sent_coords[1]), color='k', zorder=101,
                         weight='bold')
            # plt.rc('font', size=8)

        # x_change = -(x - old_sent_coords[0]) / 4
        shift_posx = 0.25 if cutoff_sent < 500 else 1
        shift_negx = 0.4 if cutoff_sent < 500 else 5
        x_change = +shift_posx if (x - old_sent_coords[0]) < 0 else -shift_negx
        if new_topic and the_topic not in annotations:
            word_popularity = usage[words.index(the_topic)]
            if word_popularity > (mean + 0.5*mean) or pod_df.iloc[idx+1]['new_branch']==True:
                # Annotate
                plt.annotate(the_topic, xy=(x + x_change, y+3), color=colour_label, zorder=150, rotation=90, weight='bold') ###UNHASH
                annotations.append(the_topic)

        old_sent_coords = [x, y]

    total_duration = pod_df.iloc[-1].timestamp
    podcast_duration = pod_df.iloc[cutoff_sent].timestamp
    total_utterances = len(pod_df)
    num_utts = cutoff_sent
    print('Total podcast_duration:', total_duration)
    print('selected podcast duration', podcast_duration)
    print('\nTotal number of utterances', total_utterances)
    print('number_of_utterances selected:', num_utts)

    # save
    if save_fig:
        if not os.path.exists('Spotify_Podcast_DataSet/{0}/{1}'.format(podcast_name, transcript_name)):
            os.makedirs('Spotify_Podcast_DataSet/{0}/{1}'.format(podcast_name, transcript_name))

        plt.savefig("Spotify_Podcast_DataSet/{0}/{1}/{1}_DT3.png".format(podcast_name, transcript_name), dpi=600)

    plt.show()

    return


def Word_Embedding_Layout(podcast_name, transcript_name, cutoff_sent=-1, save_fig=False, info=False):
    """
    Function to plot the word embedding layout, using backbone df.
    """
    # load relevant df
    pod_df = pd.read_hdf('Spotify_Podcast_DataSet/{0}/{1}/transcript_df.h5'.format(podcast_name, transcript_name),
                         key='df')

    # Instantiate figure
    plt.figure()
    plt.title('WE: {}'.format(transcript_name))
    plt.rc('font', size=7)

    for idx, row in pod_df[pod_df['new_topic']==True][0:cutoff_sent].iterrows():
        the_topic = row['stack_name']

        branch_num = row['branch_num']
        new_branch = row['new_branch']
        x = row['word_embedding_X']
        y = row['word_embedding_Y']

        if not x or not y: #i.e. couldn't find an embedding for the given topic, then just ignore and move onto next
            continue

        plt.plot(x, y, 'o', color='lightpink')
        plt.annotate(the_topic, xy=(x, y), xytext=(-5, 0), textcoords="offset points")

    ConceptNet_df = pd.read_hdf('Spotify_Podcast_DataSet/{0}/ConceptNet_Numberbatch_TSNE.h5'.format(podcast_name))
    plt.set_xlim(min(ConceptNet_df['X']), max(ConceptNet_df['X']))
    plt.set_ylim(min(ConceptNet_df['Y']), max(ConceptNet_df['Y']))

    # save
    if save_fig:
        if not os.path.exists('Spotify_Podcast_DataSet/{0}/{1}'.format(podcast_name, transcript_name)):
            os.makedirs('Spotify_Podcast_DataSet/{0}/{1}'.format(podcast_name, transcript_name))

        plt.savefig("Spotify_Podcast_DataSet/{0}/{1}/{1}_WE.png".format(podcast_name, transcript_name), dpi=600)

    plt.show()

    return

def TTTS(podcast_name, transcript_name, cutoff_sent=-1, save_fig=False, heatmap=False):
    """
    Function to plot the Trajectory Through Topic Space Visualisation, using backbone df.
    """

    # load relevant df
    pod_df = pd.read_hdf('Spotify_Podcast_DataSet/{0}/{1}/transcript_df.h5'.format(podcast_name, transcript_name),
                         key='df')
    keyword_df = pd.read_hdf('Spotify_Podcast_DataSet/{0}/ConceptNet_Numberbatch_TSNE.h5'.format(podcast_name),
                         key='df')

    # find popularity of each key topic in terms of
    #A) number of times the topic is linked back to
    Popularity = {}

    for topic in pod_df[0:cutoff_sent][pod_df['new_topic']==True].stack_name:
        try:
            Popularity[topic] += 1
        except:
            Popularity[topic] = 1 # initiate w/ 2: if a new branch was made there that's the 2nd time it's mentioned


    #B) number of utterances spent on the topic
    Topics_Utterances = dict(pod_df['stack_name'].value_counts())
    lists = sorted(Topics_Utterances.items(), key=lambda kv: kv[1])
    x, y = zip(*lists)
    largest = y[-1]
    len_quartiles = int(largest/4)
    quartiles = [list(range(0, len_quartiles)), list(range(len_quartiles, 2*len_quartiles)), list(range(2*len_quartiles, 3*len_quartiles)), list(range(3*len_quartiles, largest+1))]

    # Instantiate figure
    plt.figure()
    # plt.rcParams['axes.facecolor'] = 'mintcream'
    plt.title('TTTS: {}'.format(transcript_name))


    # if heatmap:
    #     a_list = []
    #     for word, usage in zip(x, y): # x is the word, y is the usage
    #         row = keyword_df[keyword_df['Topics']==word]
    #         X, Y = row['X'].values, row['Y'].values
    #         print(X, Y)
    #         a_list.append([float(X), float(Y), usage])
    #     print('a', a_list)
    #     b = np.asarray(a_list)
    #     plt.pcolor(b)


    xs, ys = list(pod_df[:cutoff_sent][pod_df['new_topic']==True].word_embedding_X), list(pod_df[:cutoff_sent][pod_df['new_topic']==True].word_embedding_Y)
    xs, ys = [x for x in xs if x], [y for y in ys if y] # remove Nones

    u = [i - j for i, j in zip(xs[1:], xs[:-1])]
    v = [i - j for i, j in zip(ys[1:], ys[:-1])]

    if len(xs) > 50:
        linewidth = 0.002
    else:
        linewidth = 0.004

    # Plot background
    for idx, row in pod_df[:cutoff_sent][pod_df['new_topic']==True].iterrows():
        the_topic = row['stack_name']

        branch_num = row['branch_num']
        new_branch = row['new_branch']
        x = row['word_embedding_X']
        y = row['word_embedding_Y']

        if not x or not y: #i.e. couldn't find an embedding for the given topic, then just ignore and move onto next
            continue

        plt.plot(x, y, 'o', color='lightpink', zorder=8)
        plt.rc('font', size=7 + [idx for idx, sublist in enumerate(quartiles) if Topics_Utterances[the_topic] in sublist][0])
        weight = 'bold' if Popularity[the_topic] > 1 else 'normal'
        plt.annotate(the_topic, xy=(x, y), xytext=(-5, 0), weight=weight, textcoords="offset points", zorder=200)

    # Plot Quiver
    n = len(xs)
    colour_for_segment = cm.Blues(np.linspace(0.3, 0.8, n))  # colormap so it's easier to follow

    plt.quiver(xs[:-1], ys[:-1], u, v, scale_units='xy',
               angles='xy', width=linewidth, scale=1, color=colour_for_segment, zorder=5)

    # plot special colours for the first and last point
    plt.plot([xs[0]], [ys[0]], 'o', color='green', markersize=8, zorder=20)
    plt.plot([xs[-1]], [ys[-1]], 'o', color='red', markersize=8, zorder=20)

    # Legend
    line1 = Line2D(range(1), range(1), color="green", marker='o', markersize=7, linestyle='none')
    line2 = Line2D(range(1), range(1), color="red", marker='o', markersize=7, linestyle='none')

    plt.legend((line1, line2), ('Beginning of Conversation', 'End of Conversation'))

    # save
    if save_fig:
        if not os.path.exists('Spotify_Podcast_DataSet/{0}/{1}'.format(podcast_name, transcript_name)):
            os.makedirs('Spotify_Podcast_DataSet/{0}/{1}'.format(podcast_name, transcript_name))

        plt.savefig("Spotify_Podcast_DataSet/{0}/{1}/{1}_TTTS.png".format(podcast_name, transcript_name), dpi=600)

    plt.show()
    plt.rc('font', size=8) # resetting
    return

def TTTS_Comparison(podcast_name, transcript_name1, transcript_name2, cutoff_sent=-1, save_fig=False, info=False):
    """Function to plot the Trajectory Through Topic Space Visualisation, using backbone df."""


    # load relevant df
    pod_df_1 = pd.read_hdf('Spotify_Podcast_DataSet/{0}/{1}/transcript_df.h5'.format(podcast_name, transcript_name1),
                         key='df')
    pod_df_2 = pd.read_hdf('Spotify_Podcast_DataSet/{0}/{1}/transcript_df.h5'.format(podcast_name, transcript_name2),
                         key='df')
    keyword_df = pd.read_hdf('Spotify_Podcast_DataSet/{0}/ConceptNet_Numberbatch_TSNE.h5'.format(podcast_name),
                         key='df')

    # Find popularity of each key topic in the two podcasts
    pod_1_topics = list(pod_df_1[pod_df_1['new_topic']==True].stack_name)
    Popularity_1 = {}
    for topic in pod_1_topics:
        try:
            Popularity_1[topic] += 1
        except:
            Popularity_1[topic] = 1 # initiate w/ 2: if a new branch was made there that's the 2nd time it's mentioned

    Popularity_2 = {}
    pod_2_topics = list(pod_df_2[pod_df_2['new_topic'] == True].stack_name)
    for topic in pod_2_topics:
        try:
            Popularity_2[topic] += 1
        except:
            Popularity_2[topic] = 1 # initiate w/ 2: if a new branch was made there that's the 2nd time it's mentioned

    Topics_Utterances1 = dict(pod_df_1['stack_name'].value_counts())

    lists1 = sorted(Topics_Utterances1.items(), key=lambda kv: kv[1])
    x1, y1 = zip(*lists1)
    largest1 = y1[-1]
    len_quartiles1 = int(largest1/4)
    quartiles1 = [list(range(0, len_quartiles1)), list(range(len_quartiles1, 2*len_quartiles1)), list(range(2*len_quartiles1, 3*len_quartiles1)), list(range(3*len_quartiles1, largest1+1))]

    Topics_Utterances2 = dict(pod_df_2['stack_name'].value_counts())
    lists2 = sorted(Topics_Utterances2.items(), key=lambda kv: kv[1])
    x2, y2 = zip(*lists2)
    largest2 = y2[-1]
    len_quartiles2 = int(largest2/4)
    quartiles2 = [list(range(0, len_quartiles2)), list(range(len_quartiles2, 2*len_quartiles2)), list(range(2*len_quartiles2, 3*len_quartiles2)), list(range(3*len_quartiles2, largest2+1))]


    # Find the overall popularity
    shared_topics = [topic for topic in list(pod_1_topics) if topic in list(pod_2_topics)]
    shared_topics = list(dict.fromkeys(shared_topics)) #only unique ones

    # Normalise / to find the ones we want in black (all shared ones in black)


    #Colour of shared topics = black
    Colours = {}
    for topic in pod_1_topics:
        Colours[topic] = 'blue'
    for topic in pod_2_topics:
        Colours[topic] = 'purple'
    for topic in shared_topics:
        Colours[topic] = 'k'

    # Instantiate figure
    plt.figure()
    plt.title('TTTS: {0} & {1}'.format(transcript_name1,transcript_name2))

    xs1, ys1 = list(pod_df_1[pod_df_1['new_topic']==True][0:cutoff_sent]['word_embedding_X']), list(pod_df_1[pod_df_1['new_topic']==True][0:cutoff_sent]['word_embedding_Y'])
    xs1, ys1 = [x for x in xs1 if x], [y for y in ys1 if y] # remove Nones

    xs2, ys2 = list(pod_df_2[pod_df_2['new_topic']==True][0:cutoff_sent]['word_embedding_X']), list(pod_df_2[pod_df_2['new_topic']==True][0:cutoff_sent]['word_embedding_Y'])
    xs2, ys2 = [x for x in xs2 if x], [y for y in ys2 if y] # remove Nones

    u1 = [i - j for i, j in zip(xs1[1:], xs1[:-1])]
    v1 = [i - j for i, j in zip(ys1[1:], ys1[:-1])]

    u2 = [i - j for i, j in zip(xs2[1:], xs2[:-1])]
    v2 = [i - j for i, j in zip(ys2[1:], ys2[:-1])]

    if len(xs1) > 50:
        linewidth = 0.002
    else:
        linewidth = 0.004

    # Plot background
    for idx, row in keyword_df.iterrows():
        the_topic = row['Topics']

        #only want to plot ones relevant to these transcripts
        try:
            color = Colours[the_topic]
        except:
            continue

        x = row['X']
        y = row['Y']

        if not x or not y: #i.e. couldn't find an embedding for the given topic, then just ignore and move onto next
            continue

        # determine which transcript this came from
        quartiles = quartiles1 if the_topic in pod_1_topics else quartiles2
        Topics_Utterances = Topics_Utterances1 if the_topic in pod_1_topics else Topics_Utterances2
        size_to_add = 2 if the_topic in shared_topics else [idx for idx, sublist in enumerate(quartiles) if Topics_Utterances[the_topic] in sublist][0]

        plt.plot(x, y, 'o', color='lightpink')
        plt.rc('font', size = 7 + size_to_add)
        weight = 'bold' if the_topic in shared_topics else 'normal'
        color = Colours[the_topic]
        plt.annotate(the_topic, xy=(x, y), xytext=(-5, 0), color=color, weight=weight, textcoords="offset points", zorder=200)

    # Plot Quiver 1
    colour_for_segment = 'lightblue' # eventually change colour with richness etc or speakerwise

    plt.quiver(xs1[:-1], ys1[:-1], u1, v1, scale_units='xy',
               angles='xy', width=linewidth, scale=1, color='lightskyblue', zorder=6)

    colour_for_segment = 'lightpurple' # eventually change colour with richness etc or speakerwise

    plt.quiver(xs2[:-1], ys2[:-1], u2, v2, scale_units='xy',
               angles='xy', width=linewidth, scale=1, color='plum', zorder=6)

    # plot special colours for the first and last point
    plt.plot([xs1[0]], [ys1[0]], 'o', color='green', markersize=8, zorder=20)
    plt.plot([xs1[-1]], [ys1[-1]], 'o', color='red', markersize=8, zorder=20)

    plt.plot([xs2[0]], [ys2[0]], 'o', color='green', markersize=8, zorder=20)
    plt.plot([xs2[-1]], [ys2[-1]], 'o', color='red', markersize=8, zorder=20)

    # Legend
    line1 = Line2D(range(1), range(1), color="green", marker='o', markersize=7, linestyle='none')
    line2 = Line2D(range(1), range(1), color="red", marker='o', markersize=7, linestyle='none')

    plt.legend((line1, line2), ('Beginning of Conversation', 'End of Conversation'))

    # save
    if save_fig:
        if not os.path.exists('Spotify_Podcast_DataSet/{0}/Combined'.format(podcast_name)):
            os.makedirs('Spotify_Podcast_DataSet/{0}/Combined'.format(podcast_name))

        plt.savefig("Spotify_Podcast_DataSet/{0}//{1}&{2}_Combined.png".format(podcast_name, transcript_name1,
                                                                               transcript_name2), dpi=600)

    plt.show()
    plt.rc('font', size=8) # resetting
    return

###iii
def Info_Collection_Handler(podcast_name):
    """
    Will automatically stop creating DTs once it's created them for 10 episodes (for now).
    """
    # First, find all transcripts for episodes of the given podcast
    if not podcast_name == 'joe_rogan':
        configfiles = list(Path("/Users/ShonaCW/Downloads/processed_transcripts (2)/").rglob(
            "**/spotify_{}_*.pkl".format(podcast_name)))
    else:
        configfiles = list(Path("/Users/ShonaCW/Downloads/processed_transcripts (2)/").rglob("**/joe_rogan_*.pkl"))

    num_podcasts = len(configfiles)
    print('\nNumber of "{0}" podcasts found: {1}'.format(podcast_name, num_podcasts), '\n\n')
    # pprint(configfiles)
    # order them by episode number?
    # order them by number of speakers present?

    # Make sure a folder is set up in which we can save the Info
    if not os.path.exists('Spotify_Podcast_DataSet/{0}'.format(podcast_name)):
        os.makedirs('Spotify_Podcast_DataSet/{0}'.format(podcast_name))

    # Next, collect backbone info
    pod_cnt = 1
    for path in configfiles:
        # print(str(path))
        print('\n', pod_cnt, '/', num_podcasts)
        if not podcast_name=='joe_rogan':
            transcript_name = str(path).split("/spotify_", 1)[1][:-4]
        else:
            transcript_name = str(path).split("/joe_rogan_", 1)[1][:-4]

        DT_Backbone(path, podcast_name, transcript_name, info=False)
        pod_cnt += 1

        # # Now, Create df of ConceptNet embeddings and save all the embeddings in the individual files...
        # for path in configfiles:
        #
        # return

    # Check if the podcast-show has its ConceptNet Numberbatch word embedding file saved...
    conceptnet_path = 'Spotify_Podcast_DataSet_/{0}/ConceptNet_Numberbatch_TSNE.h5'.format(podcast_name)

    if not os.path.exists(conceptnet_path):
        #     ConceptNet_TSNE_df = pd.read_hdf(conceptnet_path, key='df')
        #
        # else:
        Create_ConceptNet_TSNE(podcast_name, transcript_name, configfiles)                        #created df explicitly for the topical word embeddings + Fills in the word embedding part of all transcripts

    return

def DT_Handler(podcast_name, podcast_count=10, cutoff_sent=-1, TTTS_only=False, DT_only=False, save_fig=False, info=False):
    """
    Will automatically stop creating DTs once it's created them for 10 episodes (for now).
    """
    # First, find all transcripts for episodes of the given podcast
    if not podcast_name=='joe_rogan':
        configfiles = list(Path("/Users/ShonaCW/Downloads/processed_transcripts (2)/").rglob("**/spotify_{}_*.pkl".format(podcast_name)))
    else:
        configfiles = list(Path("/Users/ShonaCW/Downloads/processed_transcripts (2)/").rglob("**/joe_rogan_*.pkl"))
    num_podcasts = len(configfiles)

    print('Number of "{0}" podcasts found: {1}'.format(podcast_name, num_podcasts))

    # Next, build Discussion Trees for each episode
    pod_cnt = 0
    for path in configfiles:
        if pod_cnt == podcast_count:
            break
        if not podcast_name=='joe_rogan':
            transcript_name = str(path).split("/spotify_", 1)[1][:-4]
        else:
            transcript_name = str(path).split("/joe_rogan_", 1)[1][:-4]

        print('\nEpisode:', transcript_name, '. Full path: ', str(path))
        print('Plotting DT...')
        if not TTTS_only:
            DT_Third_Draft(podcast_name, transcript_name, cutoff_sent=cutoff_sent, save_fig=save_fig, info=info)
        print('Plotting TTTS...')
        if not DT_only:
            TTTS(podcast_name, transcript_name, cutoff_sent=cutoff_sent, save_fig=save_fig)
        pod_cnt += 1

    return

## Call....

#Simple_Line_DA(cutoff_sent=400, Interviewee='jack dorsey', save_fig=False) #'jack dorsey' # 'elon musk'
#Simple_Line_Topics(cutoff_sent=200, Interviewee='jack dorsey', save_fig=False)

# Shifting_Line_Topics(cutoff_sent=400, Interviewee='jack dorsey', save_fig=False)
# Shifting_Line_Topics_2(cutoff_sent=400, Interviewee='jack dorsey', save_fig=False)
# DT_Shifting_Line_Topics(Interviewee='jack dorsey', logscalex=False, save_fig=False)

#DT_First_Draft(cutoff_sent=200, Interviewee='jack dorsey', save_fig=False) #'jack dorsey' #'elon musk' #kanye west
#DT_Second_Draft('/Users/ShonaCW/Downloads/processed_transcripts (2)/186/spotify_heavy_topics_fuckboys_and_44643.pkl', 'heavy_topics', cutoff_sent=-1, save_fig=False, info=False)

#DT_Backbone('/Users/ShonaCW/Downloads/processed_transcripts (2)/186/spotify_heavy_topics_fuckboys_and_44643.pkl', 'heavy_topics', 'fuckboys_and', info=False)

#DT_Third_Draft('heavy_topics', 'heavy_topics_being_a_71410', cutoff_sent=-1, save_fig=False, info=False)
#TTTS('heavy_topics', 'heavy_topics_i_killed_94201', cutoff_sent=100, save_fig=False, heatmap=False) #'heavy_topics_fuckboys_and_44643'

#Info_Collection_Handler('joe_rogan')
DT_Handler('joe_rogan', podcast_count=12, cutoff_sent=1000, save_fig=False, TTTS_only=False, DT_only=True) #'wall_street' #'5_star' (football one) #'confessions_of'

#TTTS_Comparison('heavy_topics', 'heavy_topics_being_a_71410', 'heavy_topics_create_the_54285') #wall_street_e2_madrid_34282', 'wall_street_e7_germany_65827') # heavy_topics_i_killed_94201   being_a_71410

def Looking_at_metadata():
    DataFrame = pd.read_csv('/Users/ShonaCW/Desktop/Imperial/YEAR 4/MSci Project/metadata.csv')
    short_shows = list(DataFrame[DataFrame['duration'] <30].show_name)
    short_shows_tech = [x for x in short_shows if 'tech' in x]
    short_shows_science = [x for x in short_shows if 'Neuroscience for Success ' in x]

    # mask = DataFrame[['show_name']].apply(lambda x: x.contains('Neuroscience for Success')).any(1)
    # neuro_df = df[mask]
    neuro_df = DataFrame[DataFrame[['show_name']].stack().str.contains('confessions of', case=False, na=False).any(level=0)]
    # print(short_shows_tech)
    # print(short_shows_science)

    print(neuro_df.head().to_string())
    return

#Looking_at_metadata()


##

def Go(path_to_transcript, use_combined_embed, speakerwise, use_saved_dfs, embedding_method, seg_method,
       node_location_method, Even_number_of_segments,
       InferSent_cos_sim_limit, Plot_Segmentation, saving_figs, put_underscore_grams, shift_ngrams, just_analysis,
       plot_hist_too, colour_quiver_plots):
    """
    Mother Function.
    names = ['Joe Rogan', 'Jack Dorsey'] or names = ['Joe Rogan', 'Elon Musk']
    """
    # Extract name of transcript under investigation
    transcript_name = Path(path_to_transcript).stem
    print('Transcript: ', transcript_name)

    # Extract names of speakers featured in transcript
    names = Extract_Names(transcript_name)
    print(names)

    ## Load + Pre-process Transcript
    with open(path_to_transcript, 'r') as f:
        content = f.read()

    all_utterances, utterances_speakerwise = Split_Transcript_By_Speaker(content, names)

    # If want to save speaker-split utterances
    with open("txts/Joe_Rogan_{}/all_utterances.txt".format('_'.join(names[1].split(' '))), "w") as f:
        for item in all_utterances:
            f.write("%s\n" % item)

    with open("txts/Joe_Rogan_{}/utterances_speakerwise_Joe.txt".format('_'.join(names[1].split(' '))), "w") as f:
        for item in utterances_speakerwise[0]:
            f.write("%s\n" % item)

    with open("txts/Joe_Rogan_{0}/utterances_speakerwise_{1}.txt".format('_'.join(names[1].split(' ')), names[1].split(' ')[0]), "w") as f:
        for item in utterances_speakerwise[1]:
            f.write("%s\n" % item)

    if just_analysis:  # not interested in plotting etc
        return

    # Lemmatize utterances (TODO: change name "content_sentences" to "content_utterances" to be more precise)
    # content_sentences = Preprocess_Content(all_utterances)
    # Remove tags
    content_sentences = [utt[6:-3] for utt in all_utterances]
    from pprintpp import pprint
    pprint(content_sentences)


    # if not speakerwise:
    #     content_onestring = Process_Transcript(content, names, Info=False)
    #     content = Preprocess_Content(content_onestring)
    #     content_sentences = sent_tokenize(content)
    #     print('number of sentences when not-split', len(content_sentences))


    if put_underscore_ngrams:
        und = 'underscore'
    else:
        und = 'nounderscore'

    # If doing podcast-specific word embedding
    if use_combined_embed:
        folder_name = 'combined_podcast'
    else:
        folder_name = transcript_name

    # ## Segmentation
    # if not speakerwise:
    #     first_sent_idxs_list = Peform_Segmentation(content_sentences, segmentation_method=seg_method,
    #                                                 Num_Even_Segs=Even_number_of_segments,
    #                                                 cos_sim_limit=InferSent_cos_sim_limit, Plot=Plot_Segmentation,
    #                                                save_fig=saving_figs)
    #     print('first_sent_idxs_list', first_sent_idxs_list)
    #
    # elif speakerwise:
    #     first_sent_idxs_list_1 = Peform_Segmentation(utterances_speakerwise[0], segmentation_method=seg_method,
    #                                                Num_Even_Segs=Even_number_of_segments,
    #                                                cos_sim_limit=InferSent_cos_sim_limit, Plot=Plot_Segmentation,
    #                                                save_fig=saving_figs)
    #     first_sent_idxs_list_2 = Peform_Segmentation(utterances_speakerwise[1], segmentation_method=seg_method,
    #                                                  Num_Even_Segs=Even_number_of_segments,
    #                                                  cos_sim_limit=InferSent_cos_sim_limit, Plot=Plot_Segmentation,
    #                                                  save_fig=saving_figs)
    #     print('first_sent_idxs_list_1', first_sent_idxs_list_1)
    #     print('first_sent_idxs_list_2', first_sent_idxs_list_2)


    ## Keyword Extraction
    if not use_saved_dfs:
        Extract_Keyword_Embeddings(content, content_sentences, embedding_method, folder_name,
                                   put_underscore_ngrams=put_underscore_ngrams, shift_ngrams=shift_ngrams, Info=True)
    # OR just load the dataframe
    keyword_vectors_df = pd.read_hdf('Saved_dfs/{0}/keyword_vectors_{1}_{2}_df.h5'.format(folder_name, und,
                                                                                          embedding_method), key='df')

    ## Segment-Wise Information Extraction
    if seg_method == 'Even':
        save_name = '{0}_{1}_segments_info_df'.format(Even_number_of_segments, seg_method)
    if seg_method == 'InferSent':
        save_name = 'InferSent_{0}_segments_info_df'.format(InferSent_cos_sim_limit)
    if seg_method == 'SliceCast':
        save_name = 'SliceCast_segments_info_df'

    # Create dataframe with the information about the segments
    # if not use_saved_dfs:
    #if use_combined_embed then want to create segments in one transcript, but gather keyword analysis using embeddings from COMBINED
    #SO want to save this info in a subfolder in combined podcasts

    if use_combined_embed:
        sub_folder_name = folder_name + '/' + transcript_name
    else:
        sub_folder_name = transcript_name

    if not speakerwise:
        # segments_info_df = get_segments_info(first_sent_idxs_list, content_sentences, keyword_vectors_df, sub_folder_name,
        #                                         save_name=save_name, Info=True)

        # OR just load the dataframe
        segments_info_df = pd.read_hdf('Saved_dfs/{0}/{1}.h5'.format(sub_folder_name, save_name), key='df')

    elif speakerwise:
        # for idx_list, utterances, name in zip([first_sent_idxs_list_1, first_sent_idxs_list_1], utterances_speakerwise, names):
        #     get_segments_info(idx_list, utterances, keyword_vectors_df, sub_folder_name, save_name=save_name+name, Info=True)


        segments_info_df_1 = pd.read_hdf('Saved_dfs/{0}/{1}.h5'.format(sub_folder_name, save_name + names[0]), key='df')
        segments_info_df_2 = pd.read_hdf('Saved_dfs/{0}/{1}.h5'.format(sub_folder_name, save_name + names[1]), key='df')

    # Topical Analysis section
    # Analysis.Analyse(content, content_sentences, keyword_vectors_df, segments_info_df)
    ## transcript_name, embedding_method, seg_method, node_location_method, Even_number_of_segments,
    ## InferSent_cos_sim_limit, saving_figs, und, shift_ngrams, save_name)
    save_name = 'ANIMATION_{0}_{1}_Segments_{2}_NodePosition'.format(Even_number_of_segments,
                                                                                seg_method, node_location_method)
    # Animate(segments_info_df, keyword_vectors_df, transcript_name = sub_folder_name, save_name = save_name,  names = names,
    #         Node_Position = node_location_method, only_nouns=True, save_fig = saving_figs,
    #         colour_quiver_plots = True) #, speakerwise_coloring = False)
    Animate_3D(segments_info_df_1, keyword_vectors_df, segments_info_df_2=segments_info_df_2, save_name = save_name,
               transcript_name = sub_folder_name, names = names, Node_Position = node_location_method, only_nouns=True,
               save_fig = saving_figs, speakerwise_colouring=True)


    ## Plot Word Embedding
    # Plot_Embeddings(keyword_vectors_df, embedding_method, folder_name, shifted_ngrams=shift_ngrams, save_fig=saving_figs)

    ## Plot Quiver Plot
    if seg_method == 'Even':
        save_name = '{0}_{1}_Segments_Quiver_Plot_With_{2}_NodePosition'.format(Even_number_of_segments,
                                                                                seg_method, node_location_method)
    if seg_method == 'InferSent':
        save_name = 'Infersent_{0}_Segments_Quiver_Plot_With_{1}_NodePosition'.format(InferSent_cos_sim_limit,
                                                                                      node_location_method)
    if seg_method == 'SliceCast':
        save_name = 'SliceCast_Segments_Quiver_Plot_With_{0}_NodePosition'.format(node_location_method)

    if not speakerwise:
        Plot_2D_Topic_Evolution_SegmentWise(segments_info_df_1=segments_info_df, save_name=save_name, transcript_name=sub_folder_name,
                                            Node_Position=node_location_method,  save_fig=saving_figs, plot_hist_too=plot_hist_too,
                                            colour_quiver_plots=True, speakerwise_coloring=False, names=names)
    if speakerwise:
        # Plot 2D Topic Evolution for each speaker separately
        save_name_spkrwise = save_name + '_SpeakerWise'
        Plot_2D_Topic_Evolution_SegmentWise(segments_info_df_1=segments_info_df_1, save_name=save_name_spkrwise, transcript_name=sub_folder_name,
                                            segments_info_df_2=segments_info_df_2, names=names,
                                            Node_Position=node_location_method,  save_fig=saving_figs, plot_hist_too=plot_hist_too,
                                            colour_quiver_plots=False, speakerwise_coloring=True)


    ## Plot Quiver + Embedding
    if seg_method == 'Even':
        save_name = '{0}_{1}_Segments_Quiver_and_Embeddings_Plot_With_{2}_NodePosition'.format(Even_number_of_segments,
                                                                                    seg_method, node_location_method)
    if seg_method == 'InferSent':
        save_name = 'Infersent_{0}_Segments_Quiver_and_Embeddings_Plot_With_{1}_NodePosition'.format(
                                                                        InferSent_cos_sim_limit, node_location_method)
    if seg_method == 'SliceCast':
        save_name = 'SliceCast_Segments_Quiver_and_Embeddings_Plot_With_{0}_NodePosition'.format(node_location_method)

    if not speakerwise:
        print('Plot_Quiver_And_Embeddings', Plot_Quiver_And_Embeddings)
        Plot_Quiver_And_Embeddings(segments_info_df_1=segments_info_df, keyword_vectors_df=keyword_vectors_df, names=names,
                                   transcript_name= sub_folder_name, save_name=save_name,
                                   Node_Position=node_location_method, only_nouns=True, save_fig=saving_figs,
                                   colour_quiver_plots=True, speakerwise_colouring=False)
    if speakerwise:
        save_name_spkrwise = save_name + '_SpeakerWise'
        Plot_Quiver_And_Embeddings(segments_info_df_1=segments_info_df_1, segments_info_df_2=segments_info_df_2,
                                   keyword_vectors_df=keyword_vectors_df, transcript_name=sub_folder_name,
                                   save_name=save_name_spkrwise, names=names,
                                   Node_Position=node_location_method, only_nouns=True, save_fig=saving_figs,
                                   colour_quiver_plots=False, speakerwise_colouring=True)


    ## Plot 3D Quiver Plot
    if seg_method == 'Even':
        save_name = '{0}_{1}_Segments_3D_Quiver_With_{2}_NodePosition'.format(Even_number_of_segments,
                                                                                seg_method, node_location_method)
    if seg_method == 'InferSent':
        save_name = 'Infersent_{0}_Segments_3D_Quiver_With_{1}_NodePosition'.format(
                                                                        InferSent_cos_sim_limit, node_location_method)
    if seg_method == 'SliceCast':
        save_name = 'SliceCast_Segments_3D_Quiver_With_{0}_NodePosition'.format(node_location_method)

    if not speakerwise:
        Plot_3D_Trajectory_through_TopicSpace(segments_info_df_1=segments_info_df, keyword_vectors_df=keyword_vectors_df,
                                              save_name=save_name, transcript_name=sub_folder_name, names=names,
                                              Node_Position=node_location_method, only_nouns= True, save_fig=saving_figs)
    if speakerwise:
        Plot_3D_Trajectory_through_TopicSpace(segments_info_df_1=segments_info_df_1, keyword_vectors_df=keyword_vectors_df,
                                              save_name=save_name + '_SpeakerWise', transcript_name=sub_folder_name,
                                              names=names, segments_info_df_2=segments_info_df_2,
                                              Node_Position=node_location_method, only_nouns= True,
                                              save_fig=saving_figs, speakerwise_colouring=True)

    return



## CODE...
if __name__ == '_/_main__':
    path_to_transcript = Path('msci-project/transcripts/joe_rogan_elon_musk.txt') #'data/shorter_formatted_plain_labelled.txt') #'msci-project/transcripts/joe_rogan_jack_dorsey.txt' #msci-project/transcripts/joe_rogan_elon_musk.txt

    embedding_method = 'fasttext'                       #'word2vec'         #'fasttext'

    seg_method = 'Even'                            #'Even'      # 'InferSent'       #'SliceCast'
    node_location_method = '1_max_count'                # 'total_average'    # '1_max_count'     # '3_max_count'

    Even_number_of_segments = 50                       # for when seg_method = 'Even'
    InferSent_cos_sim_limit = 0.52                      # for when seg_method = 'InferSent' 52

    put_underscore_ngrams = False                    # For keywords consisting of >1 word present them with '_' between (did this bc was investigating whether any of the embeddings would recognise key phrases like 'United States' better in that form or 'United_States' form)
    shift_ngrams = True                              # Shift embedded position of ngrams to be the position of the composite noun (makes more sense as the word embeddnigs don't recognise most ngrams and hence plot them all together in a messy cluster)

    Plotting_Segmentation = True
    saving_figs = False

    use_saved_dfs = True                             # i.e. don't extract keywords/ their embeddings, just used saved df

    just_analysis = True

    use_combined_embed = True
    speakerwise = True

    colour_quiver_plots = False
    plot_hist_too = False                            # Plot a histogram indicating the number of keywords contained in each segment (and defined colour schemes for

    Go(path_to_transcript, use_combined_embed, speakerwise, use_saved_dfs, embedding_method, seg_method, node_location_method, Even_number_of_segments,
       InferSent_cos_sim_limit, Plotting_Segmentation, saving_figs, put_underscore_ngrams, shift_ngrams,
       just_analysis, plot_hist_too, colour_quiver_plots)
