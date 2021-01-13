import pandas as pd
from pathlib import Path
import numpy as np

def Analyse(transcript_name, embedding_method, seg_method, node_location_method, Even_number_of_segments,
            InferSent_cos_sim_limit, saving_figs, und, shift_ngrams, seg_save_name):
    """
    Function to

    - general analysis of keyword usage for topic inference
    - speaker oriented analysis

    == QUESTIONS ==
    Question 1:
        Where do most of the keywords get extracted from? Are the top PKE (TopicRank) keywords extracted in an evenly-
        distributed manner or are they mainly extracted from the beginning/middle/end of the transcript?
        What does that say about the focus of the convo in these sections? i.e. are less important topics typically
        discussed at the same times (fraction-wise) during a conversation/ during a podcast?

    Question 2:
        Find the top keywords revisited during the podcast, i.e. the top keywords in terms of a Counter for the
        entire transcript. Then plot segment number vs. # uses of word (in that segment), to
        show which parts of the convo they're brought up in most.

    Question 3:
        Does the number of keywords contained in a section tell us about the richness of conversation?

    == ISSUES ==
    Issue 1:
        How can I ignore un-interesting keywords like

    ==NOTES==
    keyword_vectors_df = pd.DataFrame(columns = ['noun_keyw',   'noun_X',    'noun_Y', 'unfamiliar_noun',
                                                 'pke_keyw',     'pke_X',    'pke_Y', ' unfamiliar_pke',
                                                 'bigram_keyw',  'bigram_X', 'bigram_Y', 'unfamiliar_bigram',
                                                 'trigram_keyw', 'trigram_X','trigram_Y', 'unfamiliar_trigram'])

    segments_dict = {'first_sent_numbers': [], 'length_of_segment': [], 'keyword_list': [], 'keyword_counts': [],
                 'total_average_keywords_wordvec': [],
                 'top_count_keyword': [], 'top_count_wordvec': [],
                 'top_3_counts_keywords': [], 'top_3_counts_wordvec': []}
    """
    desired_width = 320

    pd.set_option('display.width', desired_width)

    np.set_printoptions(linewidth=desired_width)

    pd.set_option('display.max_columns', 10)

    keyword_vectors_df = pd.read_hdf('Saved_dfs/keyword_vectors_{}_{}_df.h5'.format(und, embedding_method), key='df')
    segments_info_df = pd.read_hdf('Saved_dfs/{0}/{1}.h5'.format(transcript_name, seg_save_name), key='df')

    # Question 1



    return
