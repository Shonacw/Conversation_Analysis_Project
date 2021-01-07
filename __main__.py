"""
#Breakdown of Code...

#Notes mentioned in code...

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
from gensim.models import Phrases

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
import string

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import get_test_data
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from collections import defaultdict
from pprint import pprint
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.cluster import DBSCAN
import importlib
topics = importlib.import_module("msci-project.src.topics")

from InferSent.models import InferSent
import torch
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import unicodedata
# from msci-project.src.topics import make_similarity_matrix, plot_similarity

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
    rake_object = rake.Rake("SmartStoplist.txt")
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

def Gensim_Phrase():
    """This stuff taken from Inference when was messing around with phrase extraction"""
    phrases = Phrases(content, min_count=2, threshold=3)
    for phrase in phrases[content]:
        print(phrase)
    # Export a FrozenPhrases object that is more efficient but doesn't allow any more training.
    # frozen_phrases = phrases.freeze()
    # print(frozen_phrases[sent]) #give it a sentence like

    ## N-GRAM EXTRACTION
    from gensim.models.phrases import Phrases, Phraser

    def build_phrases(sentences):
        phrases = Phrases(sentences,
                          min_count=2,
                          threshold=3,
                          progress_per=1000)
        return Phraser(phrases)

    phrases_model.save('phrases_model.txt')

    phrases_model = Phraser.load('phrases_model.txt')

    def sentence_to_bi_grams(phrases_model, sentence):
        return ' '.join(phrases_model[sentence])

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
        if Info:
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
        print('Number of Words from Document without an embedding: ', len(words_unplotted))
        print('List of Words lacking an embedding:', words_unplotted)
    return words, vectors, words_unplotted

# def Plot_Word_Embedding(path_to_transcript, path_to_pretrained_vecs):
#     """
#     Function for plotting word embeddings.
#     """

## Code...
def Extract_Keyword_Vectors(content, content_sentences, Info=False):
    """
    (made on 5th January, works well haven't tested function today (6th) so far though ======)
    Function to extract all types of keywords from transcript + obtain their word embeddings. Only needs to be run once
    then all the keywords + their embeddings are stored in a dataframe 'keyword_vectors_df which is then saved to hdf
    for easy loading in future tasks.

    Note A:
        Currently using GoogleNews pretrained word vectors, but could also use Glove. The benefit of the Google model is
        that it contains vectors for some 'phrases' (bigrams/ trigrams) which is helpful for the plot being meaningful!
    Note B:
        The tsne vector dimensionality must be done all together, but in a way that I can then split the vectors back
        into groups based on keyword type. Hence a little more fiddly.

    """
    if Info:
        print('-\nExtracting keywords + obtaining their word vectors using GoogleNews pretrained model...')

    # Choose pre-trained model... Note A
    Glove_path = r'GloVe/glove.840B.300d.txt'
    Google_path = r'Google_WordVectors/GoogleNews-vectors-negative300.txt'
    path_to_vecs = Google_path

    words = Prep_Content_for_Ngram_Extraction(content)
    if Info:
        print("-Extracted content/sentences/words from transcript.")

    # Get embeddings dictionary of word vectors  from pre-trained word embedding
    embeddings_dict = {}
    if Info:
        print("-Obtaining embeddings...")
    with open(path_to_vecs, 'r', errors='ignore', encoding='utf8') as f:
        try:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector
        except:
            f.__next__()
    if Info:
        print("-Obtained all GoogleNews embeddings.")

    # Extract words to plot

    nouns_set = Extract_Embeddings_For_Keywords(Extract_Nouns(content_sentences), embeddings_dict)
    pke_set = Extract_Embeddings_For_Keywords(PKE_keywords(content), embeddings_dict)
    bigram_set = Extract_Embeddings_For_Keywords(Extract_bigrams(words), embeddings_dict)
    trigram_set = Extract_Embeddings_For_Keywords(Extract_trigrams(words), embeddings_dict)
    if Info:
        print('-Extracted embeddings for all keywords.')

    # Reduce dimensionality of word vectors such that we can store X and Y positions.
    tsne = TSNE(n_components=2, random_state=0)
    sets_to_plot = [nouns_set, pke_set, bigram_set, trigram_set]

    last_noun_vector = len(nouns_set[0])
    last_pke_vector = last_noun_vector + len(pke_set[0])
    last_bigram_vector = last_pke_vector + len(bigram_set[0])
    last_trigram_vector = last_bigram_vector + len(trigram_set[0])
    all_vectors = list(itertools.chain(nouns_set[1], pke_set[1], bigram_set[1], trigram_set[1]))
    all_keywords = list(itertools.chain(nouns_set[0], pke_set[0], bigram_set[0], trigram_set[0]))

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

    # Store keywords + embeddings in a hd5 file for easy accessing in future tasks.
    keyword_vectors_df.to_hdf('Saved_dfs/keyword_vectors_df.h5', key='df', mode='w')
    if Info:
        print('-Created and saved keyword_vectors_df dataframe.')

    return nouns_set, pke_set, bigram_set, trigram_set


def Find_Keywords_in_Segment(sents_in_segment, all_keywords, Info=False):
    """
    should use regular expressions to make faster
    go segment by segment and look for all the keywords/phrases in it, make it into a list
    """
    keywords_contained_in_segment = {} #keys are keywords, values are the number of ocurrences of the keyword in the seg

    sents_in_subsection_flat = ''.join(list(itertools.chain.from_iterable(sents_in_segment))) # one string
    words_in_subsection = word_tokenize(sents_in_subsection_flat)   #might need to remove punctuation before doing this

    #convert to strings
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



def Sentence_Wise_Keyword_Averaging():
    """
    Sentence by sentence analysis (this would look super messy?) or maybe not as some sentences wouldnt contain a keyword
    so would be a reasonable amount of space (along sentence_number axis) between points
    """

def Segment_Wise_Keyword_Averaging(content_in_sentences, list_of_segment_starting_points, keyword_vectors_df):
    """
    Function to perform calculate average topic position during each segment of the conversation.

    If segments are small enough, i.e. only 2-5 utterances at a time, should only really contain a couple of keywords
    that will be (hopefully) semantically similar.

    Requires the sentence-tokenized full transcript, list of sentences indices corresponding the start of each segment,

    need to a) find which keywords are used in each segment and turn into a dict, i.e. {'segment 1': ['brain', 'body']}
    then using the vector information about each keyword (which can be searched up in the keyword_vectors_df)
    can find average position in topic space for each segment of the transcript.
    """

    #haven't yet cleaned/ updated=======

    Keywo_Embed_df_manual = pd.read_hdf('./SGCW/Keyword_Embeddings_df_manual.h5', key='dfs')
    segs_manual_info_df = pd.read_hdf('./SGCW/segs_manual_info.h5', key='dfs')
    keywords_manual = segs_manual_info_df['keyword_list'].values
    (av_xs_manual, av_ys_manual, segment_numbs_manual) = ([], [], [])

    for idx, list_manual in enumerate(keywords_manual):
        ## MANUAL
        # Access XY coord for each word in the sublist
        word_list_X = []
        word_list_Y = []
        for word in list_manual:
            # print(word)
            word_list_X.append(Keywo_Embed_df_manual[Keywo_Embed_df_manual['Word'] == word].X.values[0])
            word_list_Y.append(Keywo_Embed_df_manual[Keywo_Embed_df_manual['Word'] == word].Y.values[0])

        # Finding average positions
        av_xs_manual.append(np.mean(word_list_X))
        av_ys_manual.append(np.mean(word_list_Y))

        segment_numbs_manual.append(idx)

    Topic_avs_df_manual = pd.DataFrame()
    Topic_avs_df_manual['Av_X'] = av_xs_manual
    Topic_avs_df_manual['Av_Y'] = av_ys_manual
    Topic_avs_df_manual.to_hdf('./SGCW/Topic_avs_df_manual.h5', key='dfs', mode='w')

def Plot_2D_Topic_Evolution_SegmentWise():
    """ Plots the nice 2D word embedding space with an arrow following the direction of the topics discussed in each
    segment of the transcript. """
    labels = df_manual['Topic_Num'].values
    xs = df_manual['Av_X'].values
    ys = df_manual['Av_Y'].values

    ax1.quiver(xs[:-1], ys[:-1], xs[1:]-xs[:-1], ys[1:]-ys[:-1], scale_units='xy',
               angles='xy', scale=1, color='b', width=0.005)

    # zip joins x and y coordinates in pairs
    for x, y, label in zip(xs, ys, labels):

        ax1.annotate(label+1, # this is the text
                     (x,y), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,10), # distance from text to points (x,y)
                     ha='center') # horizontal alignment can be left, right or center

    #plot special colours for the first and last point
    ax1.plot([xs[0]], [ys[0]], 'o', color='green', markersize=10, label='Beginning of Conversation')
    ax1.plot([xs[-1]], [ys[-1]], 'o', color='red', markersize=10, label='End of Conversation')
    ax1.set_title('Manual')

    fig.show()



## Now for 3d version (?)

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

def Plot_3D_Trajectory_through_TopicSpace():
    """
    Taken from my messy code in Inference. Here ready for when I have segmentation info from Jonas' method.
    """
    df_manual = pd.read_hdf('./SGCW/Topic_avs_df_manual.h5', key='dfs')
    labels_manual = df_manual['Topic_Num'].values
    df_slice = pd.read_hdf('./SGCW/Topic_avs_df_slice.h5', key='dfs')
    labels_slice = df_slice['Topic_Num'].values

    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=(22, 11))  # plt.figaspect(0.5))
    fig.suptitle('Movement through topic space over time')

    # set up the axes for the first plot
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    ##MANUAL ----------------------------------------------------------
    # Data for a three-dimensional line
    word_emb_xs = df_manual['Av_X'].values
    word_emb_ys = df_manual['Av_Y'].values
    label_seg_df = pd.read_hdf('./SGCW/segs_manual_info.h5', key='dfs')
    segment_numbers = label_seg_df['first_sent_numbers'].values
    print('segment_numbers', segment_numbers)

    #ax.plot3D(xs, segment_numbers, ys, 'bo-')
    ax1.set_xlabel('$time (Segment Number)$', fontsize=13)
    ax1.set_ylabel('$X$', fontsize=20, rotation = 0)
    ax1.set_zlabel('$Y$', fontsize=20)
    ax1.zaxis.set_rotate_label(False)
    ax1.set_title('Manual')

    cnt = 0
    # (old_x, old_y, old_z) = (0, 0, 0)
    for x, y, z, label in zip(segment_numbers, word_emb_xs, word_emb_ys, labels_manual):
      cnt +=1
      ax1.plot([x], [y], [z],'o') #markerfacecolor='k', markeredgecolor='k', marker='o', markersize=5, alpha=0.6)
      ax1.text(x, y, z, label+1, size=10)
      if cnt ==1:
        (old_x, old_y, old_z) = (x, y, z)
        continue

      a = Arrow3D([old_x, x], [old_y,y], [old_z, z], mutation_scale=20, lw=3, arrowstyle="-|>", color="r")
      ax1.add_artist(a)

      (old_x, old_y, old_z) = (x, y, z)


    ## AXIS STUFF
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

    fig.show()

def PlotWord_Embeddings(keyword_vectors_df, save_fig=False):
    """
    Plots the Word2Vec layout of all the keywords from the podcast. Keywords include those extracted using TopicRank,
    all potentially-interesting nouns, and all extracted bigrams and trigrams. Includes colour coordination with respect
    to the type of keywords.
    """
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
        print('\nPlotted', labels[i])
        print(labels[i],'which were not plotted due to lack of embedding: ', list(unplotted))

    plt.legend()
    plt.show()

    if save_fig:
        plt.savefig("Saved_Images/Keyword_Types_WordEmbedding.png", dpi=900)
    return

def Cluster_Transcript(content_sentences):
    """
    Taken from Msci project work that Jonas did on 5th January

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


## Functions for Segmentation with InferSent...

def Obtain_Sent_Embeddings_InferSent(sentences, V=1, Info=False):
    """
    V = 1 for GloVe, 2 for FastText
    """
    all_combinations = False  # True to compare ALL sentences, False to compare only consecutive sentences
    cutoff = 0.5  # Cutoff value for weighted graph
    # STEP 1
    # Build sentences list...
    # Load pre-processed transcript of interview between Elon Musk and Joe Rogan
    # with open(path, 'r') as f:
    #     content = f.read()
    #     sentences = nltk.sent_tokenize(content)
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

def Calc_CosSim_Long_Sents(content_sentences, embeddings, cos_sim_limit=0.52, Info=False):
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

    # def remove_punctuation(text):
    #     return text.translate(tbl)
    if Info:
        print('\n-Creating cos_sim_df dataframe...')

    idxs_blank = []
    content_sentences_copy = content_sentences
    cos_sim_df = pd.DataFrame(columns=['Sentence1', 'Sentence1_idx', 'Sentence2', 'Sentence2_idx', 'Cosine_Similarity',
                                       'New_Section'])

    # Firstly, put all sentences of length <5 to blank in a new content_sentences_object (preliminary method to deal with filler sents)
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

def get_segments_info(df, content_sentences, all_keywords, Info=False):
    """
    df = cos_sim_df
    the sentence indices for which a new sentence are just the values of 'Sentence2_idx' column which also have a '1'
    in the 'New_Section' column
    """
    if Info:
        print('\n-Obtaining information about each segment using cos_sim_df...')

    segments_dict = {'first_sent_numbers': [], 'length_of_segment': [], 'keyword_list': [], 'keyword_counts' : []}

    df_mini = df[df['New_Section'] == 1]
    old_idx = 0
    for idx, row in df_mini.iterrows():
        segments_dict['first_sent_numbers'].append(row['Sentence2_idx'])  # POSITION of each section
        length = np.int(row['Sentence2_idx']) - np.int(old_idx)  #  LENGTH of each section
        segments_dict['length_of_segment'].append(length)

        sentences_in_segment = content_sentences[old_idx : row['Sentence2_idx']]
        keywords_dict = Find_Keywords_in_Segment(sentences_in_segment, all_keywords, Info=False)
        segments_dict['keyword_list'].append(list(keywords_dict.keys()))
        segments_dict['keyword_counts'].append(list(keywords_dict.values()))

        old_idx = row['Sentence2_idx']

    # Convert dictionary to dataframe
    segments_info_df = pd.DataFrame({k: pd.Series(l) for k, l in segments_dict.items()})

    if Info:
        print('-Created segments_info_df. Preview: ')
        print(segments_info_df.head().to_string())
        print('Lengths of Segments:', segments_dict['length_of_segment'])
        print([len(words_list) for words_list in segments_dict['keyword_list']])
        print('Number of segments with zero keywords:', [len(words_list) for words_list in list(keywords_dict.keys())].count(0))

    return segments_info_df


## Umbrella (?) (parent?) (Mother?) (BIG BOY?) Functions...
def Peform_Segmentation(Evenly=False, Num_Even_Segs=10, Infersent=False, cos_sim_limit=0.52):
    """
    Function to segment up a transcript. By default will segment up the transcript into 'Num_Even_Segs' even segments.
    """
    if Infersent:
        # Obtain sentence embeddings using InferSent + create dataframe of consec sents cosine similarity + predict segmentation
        embeddings = Obtain_Sent_Embeddings_InferSent(content_sentences, Info=False)
        cos_sim_df = Calc_CosSim_Long_Sents(content_sentences, embeddings, cos_sim_limit, Info=True)

        # [OR if embeddings were already obtained, simply load dataframe]
        # cos_sim_df = pd.read_hdf('Saved_dfs/InferSent_cos_sim_df.h5', key='df')


if __name__=='__main__':
    # Load pre-processed transcript of interview between Elon Musk and Joe Rogan...
    path_to_transcript = Path(
        '/Users/ShonaCW/Desktop/Imperial/YEAR 4/MSci Project/Conversation_Analysis_Project/data/shorter_formatted_plain_labelled.txt')

    # Get content from transcript
    with open(path_to_transcript, 'r') as f:
        content = f.read()
        content_sentences = nltk.sent_tokenize(content)

    ## Step One: SEGMENTATION - this uses Infersent sentence embedding + cosine similarity for cutoff
    # Obtain sentence embeddings using InferSent + create dataframe of consec sents cosine similarity + predict segmentation
    # embeddings = Obtain_Sent_Embeddings_InferSent(content_sentences, Info=False)
    ##cos_sim_df = Calc_CosSim_Long_Sents(content_sentences, embeddings, cos_sim_limit, Info=True)

    # [OR if embeddings were already obtained, simply load dataframe]
    cos_sim_df = pd.read_hdf('Saved_dfs/InferSent_cos_sim_df.h5', key='df')

    ## Step TWO Obtain all keywords
    # nouns_set, pke_set, bigram_set, trigram_set = Extract_Keyword_Vectors(content, content_sentences, Info=True)

    # [OR if keywords already extracted and saved together in keyword_vectors_df, simply load dataframe]
    keyword_vectors_df = pd.read_hdf('Saved_dfs/keyword_vectors_df.h5', key = 'df')
    all_keywords = list(itertools.chain(keyword_vectors_df['noun_keyw'].values, keyword_vectors_df['pke_keyw'].values,
                                        keyword_vectors_df['bigram_keyw'].values, keyword_vectors_df['trigram_keyw'].values
                                        ))
    """NOTE this is INEFFICIENT. i'm combining all the keywords here, then in the next function i'm separating them again
    but just feel like this is more generalisable... so will do this for now until I decide which way to store them"""

    # Step THREE: Collect information about the keywords contained in each segment
    segments_info_df = get_segments_info(cos_sim_df, content_sentences, all_keywords, Info=True)

    ## Step FOUR: Calculate average keyword position for each segment

    ## Step FIVE: Plots.
