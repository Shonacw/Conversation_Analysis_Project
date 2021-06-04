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
from datetime import datetime

import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib import cm


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
    Function to extract the names of the speakers given the title of the Joe Rogan podcast transcript title.

    Returns the two names in a list.
    """
    # Split up title and capitalise words
    upper_names = []
    for i in transcript_name.split("_"):
        upper_names.append(i.title())

    # Combine names
    first_name = " ".join(upper_names[:2])
    second_name = " ".join(upper_names[2:])

    return [first_name, second_name]


def Preprocess_Content(content_utterances):
    """
    Function to perform Lemmatization of the whole transcript when it is first imported.
    """
    nlp = spacy.load('en', disable=['parser', 'ner'])               # Load spacy model

    content_utterances_cleaned = []
    for utterance in content_utterances:
        utt = nlp(utterance)
        content_lemma = " ".join([token.lemma_ for token in utt])   # Join lemmatized version of words with spaces
        content_lemma = re.sub(r'-PRON-', "", content_lemma)        # Removing annoying pronoun tags
        content_utterances_cleaned.append(content_lemma)

    return content_utterances_cleaned

def Replace_ngrams_In_Text(content, bigrams_list, trigrams_list):
    """
    Function to replace all cases of individual words from detected n-grams with their respective bi/trigram.
    Returns the document content as a list of words.

    Note that '-' placeholders are used maintain index numbering, but are removed later on.

    Note why trigrams were replaced first:
    ['brain', 'simulation'] was a detected bigram, and ['deep', 'brain', 'simulation'] was a detected trigram. If
     all the cases of 'brain' and 'simulation' are condensed into 'brain_simulation' first, then there would be no cases
     of 'deep', 'brain', and 'simulation' left in the transcript and therefore no trigrams would be found.
    """
    list_of_condensed_grams = []
    content_tokenized = word_tokenize(content)

    # Replace Trigrams...
    for trigram in trigrams_list:
        trigram_0, trigram_1, trigram_2 = trigram                       # Extract individual trigram words
        trigram_condensed = str(trigram_0.capitalize() + '_' + trigram_1.capitalize() + '_' + trigram_2.capitalize())
        list_of_condensed_grams.append(trigram_condensed)
        indices = [i for i, x in enumerate(content_tokenized) if x.lower() == trigram_0
                   and content_tokenized[i+1].lower() == trigram_1
                   and content_tokenized[i+2].lower() == trigram_2]     # Finding indices of trigram words in list
        for i in indices:
            content_tokenized[i] = trigram_condensed
            content_tokenized[i+1] = '-'                                # Placeholders - removed later on
            content_tokenized[i+2] = '-'

    # Replace Bigrams...
    for bigram in bigrams_list:
        bigram_0, bigram_1 = bigram                                     # Extract individual Bigram words
        bigram_condensed = str( bigram_0.capitalize() + '_' + bigram_1.capitalize())
        list_of_condensed_grams.append(bigram_condensed)
        indices = [i for i, x in enumerate(content_tokenized) if x.lower() == bigram_0
                   and content_tokenized[i+1].lower() == bigram_1]      # Finding indices of bigram words in list
        for i in indices:
            content_tokenized[i] = bigram_condensed
            content_tokenized[i+1] = '-'                                # Placeholders - removed later on

    return content_tokenized

def Preprocess_Sentences(content_sentences):
    """
    Function to preprocess sentences such that they are ready for keyword extraction.
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

    Rapid Automatic Keyword Extraction algorithm. Determines key phrases in a body of text by analyzing
    the frequency of word appearance and its co-occurance with other words in the text.
    """
    rake_object = rake.Rake("data/Rake_SmartStoplist.txt")
    keywords = rake_object.run(content)
    if Info:
        print("\nRAKE Keywords:", keywords)

    return keywords


def Counter_Keywords(content_sentences, Info=False):
    """
    Function to extract the top words used in a document using a simple counter.
    Note these are not really 'keywords', just the most common words.
    """
    sents_preprocessed = Preprocess_Sentences(content_sentences)        # Pre-process sentences
    sents_preprocessed_flat = reduce(operator.add, sents_preprocessed)  # Flatten into one long sentence
    keywords = Counter(sents_preprocessed_flat).most_common(10)         # Extract keywords
    if Info:
        print('\nCounter Keywords: ', keywords)

    return keywords



def get_word_from_ngram(ngram, nlp):
    """
    Function to find coordinates for n_grams in topic space using the word vectors of nouns they contain.
    """
    # Loop through PKE keywords and look at the n-grams
    if '_' in ngram:                                                # word_tokenize requires separation by spaces
        ngram = ngram.split("_")
        ngram = ' '.join(ngram)

    pos_list = [word.pos_ for word in nlp(str(ngram))]              # Generate list of Part-of-Speech tags for ngram

    if len(pos_list) > 1:
        counter = Counter(pos_list)
        words = word_tokenize(ngram)
        if 'NOUN' in counter:
            word_to_use = words[pos_list.index('NOUN')]             # TODO: deal with case of two nouns

        elif 'PROPN' in counter:
            word_to_use = words[pos_list.index('PROPN')]

        else:
            return 'nan'                                            # If phrase doesn't contain a noun, discard

    return word_to_use



def Extract_Embeddings(words_to_extract, word2vec_embeddings_dict, fasttext_model, embedding_method,
                       shift_ngrams=False, Info=False):
    """
    Function for extracting the word vectors for the given keywords.

    Note A:
        Enter here if wanting to generate coordinate for n-grams based on word vectors of words they contain.
    Note B:
        Here we save original version of ngram string... but altered version of embedding.
    Note C:
        Must check all possible versions of given word (all-capitals, non-capitals, etc) as Google Embeddings
        are inconsistent in form.
    Note D:
        TODO: Maybe add a checker of whether the STEM / LEMMA of a word exists
    Note E:
        If multiple possible versions of a given word exists in the embedding vocab, take only the first instance.
    """
    words, vectors, words_unplotted = [], [], []
    nlp = spacy.load("en_core_web_sm")                                              # Load model here for efficiency

    # Word2Vec
    if embedding_method == 'word2vec':
        for word in words_to_extract:
            if shift_ngrams and (len(word_tokenize(word)) > 1 or '_' in word):      # Note A
                if len(word_tokenize(word)) > 1 or '_' in word:
                    word_to_use = get_word_from_ngram(word, nlp)
                    if word_to_use == 'nan':
                        words_unplotted.append(word)
                        continue
                    words.append(word)                                              # Note B
                    vectors.append(word2vec_embeddings_dict[word_to_use])
            else:
                if "_" in word:                                                     # Note C
                    new_word = []
                    for i in word.split("_"):
                        new_word.append(i.title())
                    capitalised_phrase = "_".join(new_word)
                    possible_versions_of_word = [word, capitalised_phrase, word.upper()]
                else:
                    possible_versions_of_word = [word, word.title(), word.upper()]  # Note D

                if Info:
                    print('possible_versions_of_word: ', possible_versions_of_word)

                boolean = [x in word2vec_embeddings_dict for x in possible_versions_of_word]
                if any(boolean):
                    idx = int(list(np.where(boolean)[0])[0])                        # Note E
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

    if Info:
        print('Words lacking an embedding:', words_unplotted)

    return words, vectors, words_unplotted



def Keywords_Embeds_Extraction(content, content_sentences, embedding_method, transcript_name,
                               put_underscore_ngrams=True, shift_ngrams=False, return_all=False, Info=False):
    """
    Function to extract all types of keywords from transcript + obtain their word embeddings. Only needs to be run once
    then all the keywords + their embeddings are stored in a dataframe 'keyword_vectors_df' which is saved to hdf
    for easy loading in future tasks.


    Note A:
        Currently using GoogleNews pretrained word vectors, but could also use Glove. The benefit of the Google model is
        that it contains vectors for some 'phrases' (bigrams/ trigrams) which is helpful for the plot being meaningful!
    Note B:
        The TSNE vector dimensionality must be done all together, but in a way that I can then split the vectors back
        into groups based on keyword type. Hence why code a little more fiddly.
    """

    if Info:
        print('\n-Extracting keywords + obtaining their word vectors using GoogleNews pretrained model...')

    content_tokenized = word_tokenize(content)
    words = [w.lower() for w in content_tokenized]

    if Info:
        print("-Extracted content/sentences/words from transcript.")

    # Collect keywords
    nouns_list    = Extract_Nouns(content_sentences, Info=False)
    pke_list      = PKE_Keywords(content, number=30, put_underscore=put_underscore_ngrams, Info=False,)
    bigrams_list  = Extract_bigrams(words, n=20, put_underscore=put_underscore_ngrams, Info=False)
    trigrams_list = Extract_trigrams(words, put_underscore=put_underscore_ngrams, Info=False)
    all_keywords  = list(itertools.chain(nouns_list, pke_list, bigrams_list, trigrams_list))

    if Info:
        print("-Extracted all keywords.")

    if embedding_method == 'word2vec':
        # Choose pre-trained model...  Note A
        Glove_path = r'GloVe/glove.840B.300d.txt'                               # GloVe
        Google_path = r'Google_WordVectors/GoogleNews-vectors-negative300.txt'  # Word2Vec
        path_to_vecs = Google_path

        # Get embeddings dictionary of word vectors from pre-trained word embedding
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
        nouns_set   = Extract_Embeddings_For_Keywords(nouns_list, embeddings_dict, None, embedding_method='word2vec')
        pke_set     = Extract_Embeddings_For_Keywords(pke_list, embeddings_dict, None, embedding_method='word2vec', shift_ngrams=shift_ngrams)
        bigram_set  = Extract_Embeddings_For_Keywords(bigrams_list, embeddings_dict, None,  embedding_method='word2vec', shift_ngrams=shift_ngrams)
        trigram_set = Extract_Embeddings_For_Keywords(trigrams_list, embeddings_dict, None, embedding_method='word2vec', shift_ngrams=shift_ngrams)

    if embedding_method == 'fasttext':
        ft = fasttext.load_model(
            '/Users/ShonaCW/Desktop/Imperial/YEAR 4/MSci Project/Conversation_Analysis_Project/FastText/cc.en.300.bin')
        fasttext.util.reduce_model(ft, 100) # Reduce dimensionality of FastText vectors from 300->100

        # Extract words to plot
        nouns_set   = Extract_Embeddings_For_Keywords(nouns_list, None, ft, embedding_method='fasttext')
        pke_set     = Extract_Embeddings_For_Keywords(pke_list, None, ft, embedding_method='fasttext', shift_ngrams=shift_ngrams)
        bigram_set  = Extract_Embeddings_For_Keywords(bigrams_list, None, ft, embedding_method='fasttext', shift_ngrams=shift_ngrams)
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

    reduced_vectors = tsne.fit_transform(all_vectors)                           # Note B

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


## Functions for Plotting...


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


## Discussion Trees...

def DT_Backbone(path, podcast_name, transcript_name, info=False):
    """
    Function which takes a given transcript and determines the position of the node for each utterance in
    Discussion-Tree (DT) space. All gathered information is saved to the 'transcript_df' dataframe in the
    relevant folder.
    """

    # Load df containing topic and Dialogue Act information...
    print('Collecting backbone info for: ', transcript_name)

    transcript_df = pd.read_pickle(path)

    # Prepare new dataframe columns
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
    topics_with_stacks = []                                                 # Will store all labels of stacks in convo
    single_stacks_appended_to_last_counter = 0
    first_idx_with_a_topic = int(transcript_df.index[transcript_df['topics'].astype(bool)].tolist()[0])

    # Deal with leaf colours quartile-wise
    Quartiles = [i for i in split_segs(range(Num_Total_Utts), 4)]
    Colours_Dict = {0: ['palegreen', 5], 1: ['lawngreen', 4], 2: ['forestgreen', 4], 3: ['darkgreen', 4]}

    # Load language model
    nlp = spacy.load("en_core_web_sm")

    # Loop through Utterances in the dataframe...
    for idx, row in transcript_df.iterrows():
        transcript_df.loc[idx, 'branch_num'] = branch_number                # Fill branch number
        quartile = next(i for i, v in enumerate(Quartiles) if idx in v)     # Find which quartile of the convo we are in
        colour_leaves = Colours_Dict[quartile][0]                           # Colour leaves according to the quartile
        no_topic = False

        # Obtain list of current active topics
        set_of_topics = row['topics']
        current_topics = [list(x) for x in set_of_topics if x]              # All topics contained in this Utt

        if idx < 20 and len(current_topics) == 0:                     # Sometimes the first few Utterances have no topic
            if info:
                print('Skipped idx due to no topics')
            continue

        current_topics = [item for sublist in current_topics for item in sublist]   # Expand list of lists into 1 list
        continued_topics = [x for x in current_topics if x == old_topic]            # Topics continued from previous Utt

        # Obtain list of active topics in the next utterance
        if idx == Num_Total_Utts - 1:                                               # Deal with case of final utterance
            new_topic = current_topics

        else:
            next_topics = [list(x) for x in transcript_df.topics[idx + 1] if x]
            next_topics = [item for sublist in next_topics for item in sublist]

            new_topic = [x for x in current_topics if x in next_topics]             # NOTE C

        continued_topic = False if len(continued_topics) == 0 else True             # False if no topics were continued

        if info:
            print('\nidx: ', idx)
            print('current_topics', current_topics)
            print('continued_topics', continued_topics)
            print('new_topic', new_topic)
            print('continued_topic', continued_topic)

        # Deal with case where no topics have been continued, and no NEW topics are being discusses
        if not continued_topic and len(new_topic) == 0:
            if len(old_current_topics) == 0:             # If this is the FIRST utterance
                new_topic = current_topics.copy()
            else:                                        # If this is not the first utterance
                continued_topics = [x for x in current_topics if x in old_current_topics] # look for ANY matching topics

                # Document the fact no new topics were started
                single_stacks_appended_to_last_counter += 1
                continued_topic = False if len(continued_topics) == 0 else True  # False if no topics were continued on
                no_topic = True

        elif continued_topic:             # If continued on topics from last Utterance, just move up the y axis 1 step
            change_in_coords = [0, 1]
            new_sent_coords = list(map(add, old_sent_coords, change_in_coords))

            if step_size_x < 0:           # Store direction in which this branch is travelling
                current_direction = -1
            else:
                current_direction = 1

            if not no_topic:
                the_topic = [topic for topic in continued_topics if topic in topics_with_stacks][0]
                Dict_of_topics[the_topic] = new_sent_coords
                Dict_of_topics_direction[the_topic] = current_direction

            # Not updating leaf colour as this is not the end of a stack (and hence doesn't have a leaf)
            transcript_df.loc[idx, 'stack_name'] = the_topic
            transcript_df.loc[idx, 'position_X'] = new_sent_coords[0]
            transcript_df.loc[idx, 'position_Y'] = new_sent_coords[1]

        # If we are not simply just shifting upwards due to a continued topic...
        elif not continued_topic:                                         # NOTE B

            # Check whether any of the new topics already have stacks, and if so, store their positions
            the_topic = None
            for top in new_topic:
                if top in topics_with_stacks:                             # Dict_of_topics
                    X_pos, Y_pos = Dict_of_topics[top]
                    topic_direction = Dict_of_topics_direction[top]
                    the_topic = top
                    break
                else:
                    continue

            ## If no familiar topics were found, simply shift the branch horizontally and start a new stack
            if the_topic is None:
                change_in_coords = [step_size_x, step_size_y]             # Shift horizontally and upwards
                if idx == first_idx_with_a_topic:
                    new_sent_coords = old_sent_coords
                else:
                    new_sent_coords = list(map(add, old_sent_coords, change_in_coords))

                the_topic = Choose_Topics(new_topic, nlp)

                topics_with_stacks.append(the_topic)
                Dict_of_topics[the_topic] = new_sent_coords
                Dict_of_topics_counts[the_topic] = 1

                if step_size_x < 0:                 # Save the direction in which this branch is travelling
                    current_direction = -1
                else:
                    current_direction = 1

                Dict_of_topics_direction[the_topic] = current_direction

                transcript_df.loc[idx, 'stack_name'] = the_topic      # Save DT-space coordinate for this utterance
                transcript_df.loc[idx, 'position_X'] = new_sent_coords[0]
                transcript_df.loc[idx, 'position_Y'] = new_sent_coords[1]
                transcript_df.loc[idx, 'new_topic'] = True

            ## If we have instead returned to a familiar topic, we jump back to it's stack and grow from there
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

                # Store DT-space coordinate for this utterance
                transcript_df.loc[idx-1, 'leaf_colour'] = colour_leaves
                transcript_df.loc[idx, 'stack_name'] = the_topic
                transcript_df.loc[idx, 'position_X'] = new_sent_coords[0]
                transcript_df.loc[idx, 'position_Y'] = new_sent_coords[1]
                transcript_df.loc[idx, 'new_topic'] = True
                transcript_df.loc[idx, 'new_branch'] = True
                transcript_df.loc[idx, 'branch_num'] = branch_number

        if (transcript_df.iloc[idx]['new_topic'] == True) and (transcript_df.iloc[idx-1]['position_Y'] in [None, 'None']):
            transcript_df.loc[idx, 'new_branch'] = True

        old_topic = the_topic
        old_current_topics = current_topics
        old_sent_coords = new_sent_coords


    # Create new hdf file for given podcast
    try:
        if not os.path.exists('Spotify_Podcast_DataSet_/{0}/{1}'.format(podcast_name, transcript_name)):
            os.makedirs('Spotify_Podcast_DataSet/{0}/{1}'.format(podcast_name, transcript_name))
    except OSError:
        pass

    # Save dataframe
    transcript_df.to_hdf('Spotify_Podcast_DataSet/{0}/{1}/transcript_df.h5'.format(podcast_name, transcript_name),
                         key='df', mode='w')

    print('Saved DF to file')
    if info:
        print(transcript_df.head(-200).to_string())

    return


def Info_Collection_Handler(podcast_name):
    """
    This function will accesses all transcripts we have from the given podcast series, and creates the 'transcript_df'
    dataframe with all information needed to plot the Discussion Tree of the podcast.
    """
    # First, find all transcripts for episodes of the given podcast
    if not podcast_name == 'joe_rogan':
        configfiles = list(Path("/Users/ShonaCW/Downloads/processed_transcripts (2)/").rglob(
            "**/spotify_{}_*.pkl".format(podcast_name)))
    else:
        configfiles = list(Path("/Users/ShonaCW/Downloads/processed_transcripts (2)/").rglob("**/joe_rogan_*.pkl"))

    # Print how many transcripts we have for the given podcast series
    num_podcasts = len(configfiles)
    print('\nNumber of "{0}" podcasts found: {1}'.format(podcast_name, num_podcasts), '\n\n')

    # Make sure a folder is set up in which we can save the Info
    if not os.path.exists('Spotify_Podcast_DataSet/{0}'.format(podcast_name)):
        os.makedirs('Spotify_Podcast_DataSet/{0}'.format(podcast_name))

    # Next, collect DT-backbone information
    pod_cnt = 1
    for path in configfiles:
        # Print the number of podcast we are currently processing
        print('\n', pod_cnt, '/', num_podcasts)

        if not podcast_name == 'joe_rogan':
            transcript_name = str(path).split("/spotify_", 1)[1][:-4]
        else:
            transcript_name = str(path).split("/joe_rogan_", 1)[1][:-4] # Joe Rogan podcasts have different title format

        DT_Backbone(path, podcast_name, transcript_name, info=False)
        pod_cnt += 1

    # Check if the podcast-show has its ConceptNet Numberbatch word embedding file saved...
    conceptnet_path = 'Spotify_Podcast_DataSet_/{0}/ConceptNet_Numberbatch_TSNE.h5'.format(podcast_name)

    if not os.path.exists(conceptnet_path):
        # Creates unique df for the topical word embeddings + fills in the word embedding part of all transcripts
        Create_ConceptNet_TSNE(podcast_name, transcript_name, configfiles)

    return


def DT_Handler(podcast_name, podcast_count=10, cutoff_sent=-1,
               TTTS_only=False, DT_only=False, Animate_only=False, save_fig=False, info=False):
    """
    Plots Discussion Tree for the podcast using data collected in DT_Backbone function (stored in transcript_df).
    Will automatically stop creating DTs once it's created them for 10 episodes (for now).

    podcast_name:   Title of podcast series we would like to plot Discussion Trees for
    podcast_count:  Number of episodes we would like to plot Discussion Trees for
    cutoff_sent:    Max number of utterances to plot
    TTTS_only:      True if only want to plot Trajectory Through Topic Space
    DT_only:        True if only want to plot Discussion Tree
    Animate_only:   True if only want to plot 2D TTTS Animation

    """
    # First, find all transcripts of the given podcast
    if not podcast_name=='joe_rogan':
        configfiles = list(Path("/Users/ShonaCW/Downloads/processed_transcripts (2)/").rglob("**/spotify_{}_*.pkl".format(podcast_name)))
    else:
        configfiles = list(Path("/Users/ShonaCW/Downloads/processed_transcripts (2)/").rglob("**/joe_rogan_*.pkl"))
    num_podcasts = len(configfiles)

    print('Number of "{0}" podcasts found: {1}'.format(podcast_name, num_podcasts))

    # Next, build Discussion Trees for each episode
    pod_cnt = 0
    for path in configfiles:
        if pod_cnt == podcast_count:        # Break if already plotted max number of DTs
            break
        if not podcast_name=='joe_rogan':   # Joe Rogan podcasts had different titles
            transcript_name = str(path).split("/spotify_", 1)[1][:-4]
        else:
            transcript_name = str(path).split("/joe_rogan_", 1)[1][:-4]

        # Check if we have the transcript_df
        try:
            pod_df = pd.read_hdf('Spotify_Podcast_DataSet/{0}/{1}/transcript_df.h5'.format(podcast_name, transcript_name),
                             key='df')
        except:
            continue

        print('\nEpisode:', transcript_name, '. Full path: ', str(path))

        if Animate_only:
            print('Plotting TTTS Animation...')
            Animate_TTTS(podcast_name, transcript_name, cutoff_sent=cutoff_sent)

        if not TTTS_only or not Animate_only:
            DT_Third_Draft(podcast_name, transcript_name, cutoff_sent=cutoff_sent, save_fig=save_fig, duration=False, labels=False)

        if TTTS_only:
            print('Plotting TTTS...')
            TTTS(podcast_name, transcript_name, cutoff_sent=cutoff_sent, save_fig=save_fig)

        # Update counter
        pod_cnt += 1

    return





def DT_Third_Draft(podcast_name, transcript_name, cutoff_sent=-1, save_fig=False, duration=False, labels=False):
    """
    Function to plot Discussion Trees using backbone data.

    cutoff_sent:    Max number of utterances to plot
    """

    # Load relevant df
    pod_df = pd.read_hdf('Spotify_Podcast_DataSet/{0}/{1}/transcript_df.h5'.format(podcast_name, transcript_name), key='df')
    colour = 'k'        # colour of tree structure
    colour_label = 'k'  # colour of annotations

    # Instantiate
    old_sent_coords = [0, 0]
    annotations = []
    Topics_Utterances = dict(pod_df['stack_name'].value_counts())
    sorted_stack_counts = sorted(Topics_Utterances.items(), key=lambda kv: kv[1], reverse=True)
    print(sorted_stack_counts)
    lists = sorted(Topics_Utterances.items(), key=lambda kv: kv[1])
    words, usage = zip(*lists)

    # Calculate the height of each stack
    idx_of_new_branch = list(pod_df[pod_df['new_topic'] == True].index)
    idx_of_new_branch.insert(0, 0)
    idx_of_new_branch.insert(-1, len(pod_df))

    # Calculate mean stack height, as only going to label longest ones
    mean = statistics.mean(usage)

    # Instantiate figure
    plt.figure()
    plt.title('Discussion Tree: {0}'.format(transcript_name.title()), fontsize=15)
    plt.rc('font', size=6)

    # Loop through utterances and plot nodes according to backbone data
    for idx, row in pod_df[0:cutoff_sent].iterrows():
        # Access information from the transcript backbone dataframe
        the_topic = row['stack_name']
        branch_num = row['branch_num']
        x = row['position_X']
        y = row['position_Y']
        new_topic = row['new_topic']
        new_branch = row['new_branch']

        plt.plot(x, y, 'o', color=colour, ms=2, zorder=0)  # Plot node for utterance

        if not new_branch:
            # Plot: continuing on the same branch, but with a new position to mark a new set of topics
            plt.plot([old_sent_coords[0], x], [old_sent_coords[1], y], '-', color=colour, linewidth=1, zorder=0)

        elif new_branch and branch_num != 0:
            # Annotate last position with a leaf + branch number label
            leaf_colour = 'yellowgreen'
            # Plot and annotate little orange dots indicating the number of branch which just ended
            plt.plot(old_sent_coords[0], old_sent_coords[1], 'o', ms=7, color=leaf_colour, zorder=100)

            if not cutoff_sent == -1:
                plt.annotate(branch_num-1, xy=(old_sent_coords[0]-0.13, old_sent_coords[1]-1), color='k', zorder=101,
                             weight='bold')

        # Add topic annotations
        if labels:
            if new_topic and the_topic not in annotations:
                x_change = 0.2 if (x - old_sent_coords[0]) < 0 else -0.5
                y_change = 2
                word_popularity = usage[words.index(the_topic)]

                if word_popularity > (mean) or pod_df.iloc[idx+1]['new_branch']==True or the_topic=='hashtag':
                    plt.rc('font', size=11)
                    plt.annotate(the_topic, xy=(x + x_change, y+y_change), color=colour_label, zorder=150,
                                 rotation=90, weight='bold')
                    annotations.append(the_topic)
                    plt.rc('font', size=11)

        old_sent_coords = [x, y]

    total_duration   = pod_df.iloc[-1].timestamp
    podcast_duration = pod_df.iloc[cutoff_sent].timestamp
    total_utterances = len(pod_df)
    total_branches   = pod_df.iloc[-1].branch_num
    total_stacks     = len(Counter(pod_df.stack_name.values).keys())
    num_utts         = cutoff_sent

    if info:
        print('Total podcast_duration:', total_duration)
        print('selected podcast duration', podcast_duration)
        print('\nTotal number of utterances', total_utterances)
        print('number_of_utterances selected:', num_utts)


    # Add podcast-level annotations
    plt.rc('font', size=11)
    if duration:
        plt.annotate(f'Podcast Duration: {podcast_duration}\nUtterances:   {total_utterances}\nBranches:      '
                     f'{total_branches}\nStacks:          {total_stacks}', xy=(-13.5, 140),
                     bbox=dict(facecolor='none', edgecolor='black', boxstyle='round, pad=0.3'))

        plt.annotate(f'First {cutoff_sent} Utterances', xy=(-5.8, 20), bbox=dict(facecolor='none', edgecolor='black',
                                                                                 boxstyle='round, pad=0.8'))
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    if cutoff_sent == -1 and podcast_name=='joe_rogan':
        plt.xlim([-14, 9.3])
        plt.ylim([0, 351])

    # Save Discussion Tree fig
    if save_fig:
        if not os.path.exists('Spotify_Podcast_DataSet/{0}/{1}'.format(podcast_name, transcript_name)):
            os.makedirs('Spotify_Podcast_DataSet/{0}/{1}'.format(podcast_name, transcript_name))

        plt.savefig("Spotify_Podcast_DataSet/{0}/{1}/{1}_DT3_16th.png".format(podcast_name, transcript_name), dpi=600)

    plt.show()

    return
