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

def Extract_Embeddings(words_to_extract, word2vec_embeddings_dict, fasttext_model, embedding_method, shift_ngrams=False,
                       Info=False):
    """
    Function for extracting the word vectors for the given keywords.

    Note A:
        If wanting to generate coordinate for ngrams based on word vectors of words they contain
    Note B:
        Must check all possible versions of given word (all-capitals, non-capitals, etc) as Google Embeddings
        are inconsistent in form.
    Note C:
        TODO: Maybe add a checker of whether the STEM / LEMMA of a word exists
    Note D:
        If multiple possible versions of a given word exists in the embedding vocab, take only the first instance
    """
    words, vectors, words_unplotted = [], [], []
    nlp = spacy.load("en_core_web_sm")                  # loading here for efficiency

    # Word2Vec
    if embedding_method == 'word2vec':
        for word in words_to_extract:
            # Note A
            if shift_ngrams and (len(word_tokenize(word)) > 1 or '_' in word):
                if len(word_tokenize(word)) > 1 or '_' in word:
                    word_to_use = get_word_from_ngram(word, nlp)
                    if word_to_use == 'nan':
                        words_unplotted.append(word)
                        continue
                    words.append(word) # save original version of ngram string... but altered version of embedding
                    vectors.append(word2vec_embeddings_dict[word_to_use])
            else:
                if "_" in word:  # Note B
                    new_word = []
                    for i in word.split("_"):
                        new_word.append(i.title())
                    capitalised_phrase = "_".join(new_word)
                    possible_versions_of_word = [word, capitalised_phrase, word.upper()]
                else:
                    possible_versions_of_word = [word, word.title(), word.upper()]  # Note C

                if Info:
                    print('possible_versions_of_word: ', possible_versions_of_word)

                boolean = [x in word2vec_embeddings_dict for x in possible_versions_of_word]
                if any(boolean):
                    idx = int(list(np.where(boolean)[0])[0])                        # Note D
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

def Keywords_Embeds_Extraction(content, content_sentences, embedding_method, transcript_name, put_underscore_ngrams=True,
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
