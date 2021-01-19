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

import itertools
from pathlib import Path
import operator
from functools import reduce
import numpy as np
from scipy import spatial
import pandas as pd
import sys
import unicodedata
from collections import defaultdict
from pprint import pprint
import random

import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from InferSent.models import InferSent
import importlib
topics = importlib.import_module("msci-project.src.topics")
Analysis = importlib.import_module("Analysis")
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

## Functions for pre-processing...
def Process_Transcript(text, names, Info=False):
    """
    Function to remove names of speakers / times of Utterances from transcript, returning all spoken utterances as a
    single string.
    """
    if Info:
        print('Text format before preprocessing:\n', text[:300])

    # Remove speaker names
    for name in names:
        text = re.sub(name, "", text)
    # Get rid of the time marks
    content_1 = re.sub('[0-9]{2}:[0-9]{2}:[0-9]{2}', " ", text)  # \w+\s\w+;
    # Strip new-lines
    content_2 = re.sub('\n', " ", content_1)
    # Strip white spaces
    content_2.strip()

    if Info:
        print('\nText format after preprocessing:\n', content_2)

    return content_2


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

def Preprocess_Content(content):
    """
    Function to perform Lemmatization of the whole transcript when it is first imported.
    """
    nlp = spacy.load('en', disable=['parser', 'ner'])
    doc = nlp(content)
    content_lemma = " ".join([token.lemma_ for token in doc])
    content_lemma = re.sub(r'-PRON-', "", content_lemma)

    return content_lemma

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
    for word in word_list:
        count = sents_in_subsection_flat.lower().split().count(word)
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
def split(a, n):
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
        idx_split = split(range(num_sents), Num_Even_Segs)
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
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points")
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


def find_colour(numb_keyw, min_keyw, max_keyw):
    # define 3 colours first
    yellow = 'y' #(255, 255, 0)
    orange = 'orange' #(255, 127, 0)
    red = 'r' #(255, 0, 0)
    # colours = [yellow, orange, red]
    #
    # colour = (0, 0, 0) #instantiate
    #
    # # define 3 keyword count
    # counts = [i[0] for i in split(range(max_keyw), 3)][1:]
    # counts.insert(0, 0)
    # counts.insert(len(counts), 100000)
    # print('counts: ', counts)
    # print('numb_keyw: ', numb_keyw)
    #
    # for i in range(len(counts)):
    #     if int(numb_keyw) >= counts[i] and int(numb_keyw) <= counts[i+1]:
    #         colour = colours[i]
    #         break
    #     else:
    #         continue
    # print('colour: ', colour)
    for i in range(len(counts)):
        if counts[i] == i:
                colour = colours[i]
                break
        else:
            continue

    return colour

def Plot_2D_Topic_Evolution_SegmentWise(segments_info_df, save_name, transcript_name, Node_Position='total_average', save_fig=False):
    """
    Plots the 2D word embedding space with a Quiver arrow following the direction of the topics discussed in each
    segment of the transcript.
    """

    if Node_Position =='total_average':
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

    # define colours for the segment
    number_keywords = [sum([int(num) for num in x]) for x in list(segments_info_df['keyword_counts'].values)]
    number_keywords_sorted = number_keywords # make a copy that we can rearrange
    number_keywords_sorted.sort()
    groups = list(split(number_keywords_sorted, 3))
    colours = [(255/255, 255/255, 0), (255/255, 125/255, 0), (240/255, 0, 0)]
    idxs = [next(index for index, sublist in enumerate(groups) if number in sublist) for number in number_keywords]
    colour_for_segment = [colours[i] for i in idxs]


    xs = [x[0] for x in node_position]
    ys = [x[1] for x in node_position]

    u = [i-j for i, j in zip(xs[1:], xs[:-1])]
    v = [i-j for i, j in zip(ys[1:], ys[:-1])]

    plt.figure()
    plt.quiver(xs[:-1], ys[:-1], u, v, scale_units='xy',
               angles='xy', scale=1, color = colour_for_segment, width=0.005)


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
                     xytext=(0, 10), # distance from text to points (x,y)
                     ha='center') # horizontal alignment can be left, right or center

    plt.rc('font', size=10)  # putting font back to normal
    #plot special colours for the first and last point
    plt.plot([xs[0]], [ys[0]], 'o', color='green', markersize=10, label='Beginning of Conversation')
    plt.plot([xs[-1]], [ys[-1]], 'o', color='red', markersize=10, label='End of Conversation')
    plt.legend()
    plt.title(save_name)
    if save_fig:
        plt.savefig("Saved_Images/{0}/{1}.png".format(transcript_name, save_name), dpi=600)
    plt.show()
    return

def Plot_Quiver_And_Embeddings(segments_info_df, keyword_vectors_df, transcript_name, save_name, Node_Position='total_average',
                               only_nouns=True, save_fig=False):
    """
    Plots BOTH a background of keywords + the 2D quiver arrow following the direction of the topics discussed in each
    segment of the transcript.
    """
    # EMBEDDING PART
    keyword_types = ['noun', 'pke', 'bigram', 'trigram']
    colours = ['pink', 'green', 'orange', 'blue']
    labels = ['Nouns', 'PKE Keywords', 'Bigrams', 'Trigrams']

    number_types_toplot = range(len(keyword_types))
    if only_nouns:
        number_types_toplot = [0]

    plt.figure()
    plt.rc('font', size=6)
    for i in number_types_toplot:
        type = keyword_types[i]
        words = keyword_vectors_df['{}_keyw'.format(type)]
        Xs, Ys = keyword_vectors_df['{}_X'.format(type)], keyword_vectors_df['{}_Y'.format(type)]
        unplotted = list(keyword_vectors_df['unfamiliar_{}'.format(type)].dropna(axis=0))

        plt.scatter(Xs, Ys, c=colours[i], label=labels[i], zorder=0)
        for label, x, y in zip(words, Xs, Ys):
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points", color='darkgrey', zorder=5)


    # QUIVER PART
    if Node_Position == 'total_average':
        node_position = segments_info_df['total_average_keywords_wordvec'].values

    if Node_Position == '1_max_count':
        labels_text = segments_info_df['top_count_keyword'].values
        node_position = segments_info_df['top_count_wordvec'].values

        # Check the segments all had enough keywords to have taken max(count)...
        node_position = [pos for pos in node_position if pos != None]
        labels_text = [label for label in labels_text if label != None]

    if Node_Position == '3_max_count':
        labels_text = segments_info_df['top_3_counts_keywords'].values
        node_position = segments_info_df['top_3_counts_wordvec'].values

        # Check the segments all had enough keywords to have taken max(count)...
        node_position = [pos for pos in node_position if str(pos[0]) != 'n']
        labels_text = [label for label in labels_text if str(label) != 'nan']
    labels = range(len(node_position))
    # define colours for the segment
    number_keywords = [sum([int(num) for num in x]) for x in list(segments_info_df['keyword_counts'].values)]
    number_keywords_sorted = number_keywords # make a copy that we can rearrange
    number_keywords_sorted.sort()
    groups = list(split(number_keywords_sorted, 3))
    colours = [(255/255, 255/255, 0), (255/255, 125/255, 0), (240/255, 0, 0)]
    idxs = [next(index for index, sublist in enumerate(groups) if number in sublist) for number in number_keywords]
    colour_for_segment = [colours[i] for i in idxs]

    xs = [x[0] for x in node_position]
    ys = [x[1] for x in node_position]

    u = [i-j for i, j in zip(xs[1:], xs[:-1])]
    v = [i-j for i, j in zip(ys[1:], ys[:-1])]

    # To make sure labels are spread out well, going to mess around with xs and ys
    xs_, ys_ = xs, ys
    ppairs = [(i, j) for i, j in zip(xs_, ys_)]
    repeats = list(set(map(tuple, ppairs)))
    repeat_num = [0 for i in range(len(repeats))]
    plt.rc('font', size=8)  # putting font back to normal
    for x, y, label in zip(xs, ys, labels):
        # first check location of annotation is unique
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
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center
    plt.rc('font', size=10)  # putting font back to normal
    plt.quiver(xs[:-1], ys[:-1], u, v, scale_units='xy',
               angles='xy', scale=1, color= colour_for_segment, width=0.005, zorder=10)

    #plot special colours for the first and last point
    plt.plot([xs[0]], [ys[0]], 'o', color='green', markersize=10, label='Beginning of Conversation')
    plt.plot([xs[-1]], [ys[-1]], 'o', color='red', markersize=10, label='End of Conversation')
    plt.title(save_name)
    plt.legend()
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

def Plot_3D_Trajectory_through_TopicSpace(segments_info_df, keyword_vectors_df, save_name, transcript_name,
                                          Node_Position='total_average', save_fig=False):
    """
    Note updated yet.
    Taken from my messy code in Inference. Here ready for when I have segmentation info from Jonas' method.
    """

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

    if Node_Position == '3_max_count':
        labels_text = segments_info_df['top_3_counts_keywords'].values
        node_position = segments_info_df['top_3_counts_wordvec'].values
        first_sents = segments_info_df['first_sent_numbers'].values

        # Check the segments all had enough keywords to have taken max(count)...
        first_sents = [i for idx, i in enumerate(first_sents) if node_position[idx]!=None]
        node_position = [pos for pos in node_position if str(pos[0]) != 'n']
        labels_text = [label for label in labels_text if str(label) != 'nan']

    labels = range(len(node_position))

    # Data for a three-dimensional line
    xs = [x[0] for x in node_position]
    ys = [x[1] for x in node_position]

    # set up a figure twice as wide as it is tall
    fig = plt.figure() #figsize=(22, 11)
    fig.suptitle('Movement of Conversation through Topic Space over Time')

    # set up the axes for the first plot
    ax1 = fig.add_subplot(111, projection='3d') #ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    #ax.plot3D(xs, segment_numbers, ys, 'bo-')
    ax1.set_xlabel('$Sentence Number$', fontsize=13)
    ax1.set_ylabel('$X$', fontsize=20, rotation = 0)
    ax1.set_zlabel('$Y$', fontsize=20)
    ax1.zaxis.set_rotate_label(False)
    #ax1.set_title('Manual')

    cnt = 0
    # (old_x, old_y, old_z) = (0, 0, 0)
    for x, y, z, label in zip(first_sents, xs, ys, labels):
      cnt +=1
      ax1.plot([x], [y], [z],'o') #markerfacecolor='k', markeredgecolor='k', marker='o', markersize=5, alpha=0.6)
      ax1.text(x, y, z, label+1, size=10)
      if cnt ==1:
        (old_x, old_y, old_z) = (x, y, z)
        continue

      a = Arrow3D([old_x, x], [old_y,y], [old_z, z], mutation_scale=20, lw=1, arrowstyle="-|>", color="b")
      ax1.add_artist(a)

      (old_x, old_y, old_z) = (x, y, z)

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

## The main function putting it all together

def Split_Transcript_By_Speaker(content, names):
    """
    Function to prepare transcript content for speaker-wise analysis.
    """
    # Get rid of the time marks
    text = re.sub('[0-9]{2}:[0-9]{2}:[0-9]{2}', " ", content)  # \w+\s\w+;

    # Remove speaker names
    codes = [123, 321]                  # Just two random codes for the speakers
    for name, code in zip(names, codes):
        text = re.sub(str(name + "\n"), str(code), text)

    # Strip new-lines
    content_2 = re.sub('\n', " ", text)
    # Strip white spaces
    content_2.strip()

    # Create list of all sentences (utterances) in transcript
    all_sentences = sent_tokenize(content_2)

    # Split into two lists of sentences, one for each speaker
    content_speaker_1 = [sent[3:] for sent in all_sentences if sent[:3] == '123']
    content_speaker_2 = [sent[3:] for sent in all_sentences if sent[:3] == '321']

    print('Number of utterances by speaker 1 (', names[0], ') : ', len(content_speaker_1), '. First few utterances:', content_speaker_1[:3])
    print('Number of utterances by speaker 2 (', names[1], ') : ', len(content_speaker_2), '. First few utterances:', content_speaker_2[:3])

    return [content_speaker_1, content_speaker_2]



def Go(path_to_transcript, use_combined_embed, speakerwise, use_saved_dfs, embedding_method, seg_method,
       node_location_method, Even_number_of_segments,
       InferSent_cos_sim_limit, Plot_Segmentation, saving_figs, put_underscore_grams, shift_ngrams, just_analysis):
    """
    Mother Function.
    names = ['Joe Rogan', 'Jack Dorsey'] or names = ['Joe Rogan', 'Elon Musk']
    """
    # Extract name of transcript under investigation
    transcript_name = Path(path_to_transcript).stem
    print('Transcript: ', transcript_name)

    # Extract names of speakers featured in transcript
    names = Extract_Names(transcript_name)

    ## Load + Pre-process Transcript
    with open(path_to_transcript, 'r') as f:
        content = f.read()
    if speakerwise:
        content_speakerwise = Split_Transcript_By_Speaker(content, names)

    if not speakerwise:
        content_onestring = Process_Transcript(content, names, Info=False)
        content = Preprocess_Content(content_onestring)
        content_sentences = sent_tokenize(content)

    if put_underscore_ngrams:
        und = 'underscore'
    else:
        und = 'nounderscore'

    # If doing podcast-specific word embedding
    if use_combined_embed:
        folder_name = 'combined_podcast'
    else:
        folder_name = transcript_name

    ## Segmentation
    first_sent_idxs_list = Peform_Segmentation(content_sentences, segmentation_method=seg_method,
                                                Num_Even_Segs=Even_number_of_segments,
                                                cos_sim_limit=InferSent_cos_sim_limit, Plot=Plot_Segmentation,
                                               save_fig=saving_figs)

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

    segments_info_df = get_segments_info(first_sent_idxs_list, content_sentences, keyword_vectors_df, sub_folder_name,
                                            save_name=save_name, Info=True)

    # OR just load the dataframe
    segments_info_df = pd.read_hdf('Saved_dfs/{0}/{1}.h5'.format(sub_folder_name, save_name), key='df')

    # Topical Analysis section
    # Analysis.Analyse(content, content_sentences, keyword_vectors_df, segments_info_df)
    ## transcript_name, embedding_method, seg_method, node_location_method, Even_number_of_segments,
    ## InferSent_cos_sim_limit, saving_figs, und, shift_ngrams, save_name)


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

    Plot_2D_Topic_Evolution_SegmentWise(segments_info_df, save_fig=saving_figs, transcript_name=sub_folder_name,
                                        Node_Position=node_location_method, save_name=save_name)

    ## Plot Quiver + Embedding
    if seg_method == 'Even':
        save_name = '{0}_{1}_Segments_Quiver_and_Embeddings_Plot_With_{2}_NodePosition'.format(Even_number_of_segments,
                                                                                    seg_method, node_location_method)
    if seg_method == 'InferSent':
        save_name = 'Infersent_{0}_Segments_Quiver_and_Embeddings_Plot_With_{1}_NodePosition'.format(
                                                                        InferSent_cos_sim_limit, node_location_method)
    if seg_method == 'SliceCast':
        save_name = 'SliceCast_Segments_Quiver_and_Embeddings_Plot_With_{0}_NodePosition'.format(node_location_method)

    Plot_Quiver_And_Embeddings(segments_info_df, keyword_vectors_df, sub_folder_name, save_name=save_name,
                               Node_Position=node_location_method, only_nouns=True, save_fig=saving_figs)

    if just_analysis:               # not interested in plotting etc
        return

    ## Plot 3D Quiver Plot
    if seg_method == 'Even':
        save_name = '{0}_{1}_Segments_3D_Quiver_With_{2}_NodePosition'.format(Even_number_of_segments,
                                                                                seg_method, node_location_method)
    if seg_method == 'InferSent':
        save_name = 'Infersent_{0}_Segments_3D_Quiver_With_{1}_NodePosition'.format(
                                                                        InferSent_cos_sim_limit, node_location_method)
    if seg_method == 'SliceCast':
        save_name = 'SliceCast_Segments_3D_Quiver_With_{0}_NodePosition'.format(node_location_method)

    Plot_3D_Trajectory_through_TopicSpace(segments_info_df, keyword_vectors_df, save_name, sub_folder_name,
                                          Node_Position='total_average', save_fig=True)

    return


## CODE...
if __name__=='__main__':
    path_to_transcript = Path('msci-project/transcripts/joe_rogan_elon_musk.txt') #'data/shorter_formatted_plain_labelled.txt') #'msci-project/transcripts/joe_rogan_jack_dorsey.txt' #msci-project/transcripts/joe_rogan_elon_musk.txt

    embedding_method = 'fasttext'                       #'word2vec'         #'fasttext'

    seg_method = 'Even'                            #'Even'      # 'InferSent'       #'SliceCast'
    node_location_method = '1_max_count'                # 'total_average'    # '1_max_count'     # '3_max_count'

    Even_number_of_segments = 100                       # for when seg_method = 'Even'
    InferSent_cos_sim_limit = 0.52                      # for when seg_method = 'InferSent' 52

    put_underscore_ngrams = False                    # For keywords consisting of >1 word present them with '_' between (did this bc was investigating whether any of the embeddings would recognise key phrases like 'United States' better in that form or 'United_States' form)
    shift_ngrams = True                              # Shift embedded position of ngrams to be the position of the composite noun (makes more sense as the word embeddnigs don't recognise most ngrams and hence plot them all together in a messy cluster)

    Plotting_Segmentation = True
    saving_figs = False

    use_saved_dfs = True                             # i.e. don't extract keywords/ their embeddings, just used saved df

    just_analysis = True

    use_combined_embed = False
    speakerwise = False

    Go(path_to_transcript, use_combined_embed, speakerwise, use_saved_dfs, embedding_method, seg_method, node_location_method, Even_number_of_segments,
       InferSent_cos_sim_limit, Plotting_Segmentation, saving_figs, put_underscore_ngrams, shift_ngrams,
       just_analysis)