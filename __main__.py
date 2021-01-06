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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import get_test_data
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

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
def Extract_Keyword_Vectors():
    """
    (made on 5th January, works well haven't tested function today (6th) so far though ======)
    Function to extract all types of keywords from transcript + obtain their word embeddings. Only needs to be run once
    then all the keywords + their embeddings are stored in a dataframe 'keyword_vectors_df which is then saved to hdf
    for easy loading in future tasks.

    Note A:
        Currently using GoogleNews pretrained word vectors, but could also use Glove. The benefit of the Google model is
         that it contains vectors for some 'phrases' (bigrams/ trigrams) which is helpful for the plot being meaningful!
    """
    # Load pre-processed transcript of interview between Elon Musk and Joe Rogan...
    path_to_transcript = Path(
        '/Users/ShonaCW/Desktop/Imperial/YEAR 4/MSci Project/Conversation_Analysis_Project/data/shorter_formatted_plain_labelled.txt')

    # Choose which pretrained model to use. GoogleNews is better (Note A)
    Glove_path = r'GloVe/glove.840B.300d.txt'
    Google_path = r'Google_WordVectors/GoogleNews-vectors-negative300.txt'
    path_to_vecs = Google_path

    # Get content from transcript
    with open(path_to_transcript, 'r') as f:
        content = f.read()
        content_sentences = nltk.sent_tokenize(content)

    words = Prep_Content_for_Ngram_Extraction(content)
    print("-Extracted content/sentences/words from transcript.")

    # Get embeddings dictionary of word vectors  from pre-trained word embedding
    embeddings_dict = {}
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
    print("-Obtained embeddings.")

    # Extract words to plot
    nouns_set = Extract_Embeddings_For_Keywords(Extract_Nouns(content_sentences, Info=True), embeddings_dict, Info=True)
    print('-Extracted embeddings for nouns.')
    pke_set = Extract_Embeddings_For_Keywords(PKE_keywords(content, Info=True), embeddings_dict, Info=True)
    print('-Extracted embeddings for pke keywords.')
    bigram_set = Extract_Embeddings_For_Keywords(Extract_bigrams(words), embeddings_dict, Info=True)
    print('-Extracted embeddings for bigrams.')
    trigram_set = Extract_Embeddings_For_Keywords(Extract_trigrams(words), embeddings_dict, Info=True)
    print('-Extracted embeddings for trigrams.')

    # Store keywords + embeddings in a pandas data-frame
    keyword_vectors_df = pd.Dataframe(columns = ['noun_keyw',   'noun_X',    'noun_Y',
                                                 'pke_keyw',     'pke_X',    'pke_Y',
                                                 'bigram_keyw',  'bigram_X', 'bigram_Y',
                                                 'trigram_keyw', 'trigram_X','trigram_Y'])
    keyword_vectors_df['noun_keyw'] = nouns_set[0]
    keyword_vectors_df['noun_X'], keyword_vectors_df['noun_Y'] = nouns_set[1], nouns_set[2]
    keyword_vectors_df['pke_keyw'] = pke_set[0]
    keyword_vectors_df['pke_X'], keyword_vectors_df['pke_Y'] = pke_set[1], pke_set[2]
    keyword_vectors_df['bigram_keyw'] = bigram_set[0]
    keyword_vectors_df['bigram_X'], keyword_vectors_df['bigram_Y'] = bigram_set[1], bigram_set[2]
    keyword_vectors_df['trigram_keyw'] = trigram_set[0]
    keyword_vectors_df['trigram_X'], keyword_vectors_df['trigram_Y'] = trigram_set[1], trigram_set[2]

    # Store keywords + embeddings in a hd5 file for easy accessing in future tasks.
    keyword_vectors_df.to_hdf('Saved_dfs/keyword_vectors_df.h5', key='df', mode='w')

    # Print segment of data-frame to ensure it was formatted correctly
    print(keyword_vectors_df.head())

    return nouns_set, pke_set, bigram_set, trigram_set

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



# Plot Word Embeddings (done 5th Jan. Works nicely but takes a while!)
def PlotWord_Embeddings(nouns_set, pke_set, bigram_set, trigram_set):
    """
    (done 5th Jan. Works nicely but takes a while!)

    Plots the Word2Vec layout of all the keywords from the podcast. Keywords include those extracted using TopicRank,
    all potentially-interesting nouns, and all extracted bigrams and trigrams. Includes colour coordination with respect
    to the type of keywords.
    """
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

    all_vectors = list(itertools.chain(nouns_set[1], pke_set[1], bigram_set[1], trigram_set[1]))
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
        cnt += 1

    plt.legend()
    plt.show()



