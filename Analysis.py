import pandas as pd
from pathlib import Path

def Analyse(transcript_name, embedding_method, seg_method, node_location_method, Even_number_of_segments,
            InferSent_cos_sim_limit, saving_figs, und, shift_ngrams, seg_save_name):
    """
    Function to

    - general analysis of keyword usage for topic inference
    - speaker oriented analysis

    Question 1:
        Where do most of the keywords get extracted from? Are the top PKE (TopicRank) keywords evenly distributed or
        were they mainly extracted from the beginning/middle/end of the transcript?

    Question 2:
        Find the top keywords that are revisited during the podcast, i.e. the top keywords in terms of Counter for the
        entire transcript. Then plot segment number vs. # uses of word (in that segment), to
        show which parts of the convo they're brought up in most.

    Question 3:

    """

    keyword_vectors_df = pd.read_hdf('Saved_dfs/keyword_vectors_{}_{}_df.h5'.format(und, embedding_method), key='df')
    segments_info_df = pd.read_hdf('Saved_dfs/{0}/{1}.h5'.format(transcript_name, seg_save_name), key='df')

    # Question 1



    return
