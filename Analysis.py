import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import re

def colored(r, g, b, text):
    """
    Function to print coloured text.

    yellow RGB = (255, 255, 0)
    orange RGB = (255, 127, 0)
    red RGB = (255, 0, 0)
    """
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

def create_heat_list():
    """
    Function to create
    """

def Analyse(content, content_sentences, keyword_vectors_df, segments_info_df):
    """
    (transcript_name, embedding_method, seg_method, node_location_method, Even_number_of_segments,
            InferSent_cos_sim_limit, saving_figs, und, shift_ngrams, seg_save_name):
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
        Does the conversation jump around or do topics slowly change?
            Measure of adjacent keyword similarity
            or measure of similarity between lists of keywords in consecutive sections
            OR Measure of how many keywords are shared between lists of section keywords

        ORRRR maybe for every key noun in the lists I should generate 3 nearest words (in general, in english vocab,
        from word embedding dictionary) and then add them on to the list (and do
        same for the list of keywords corresponding to the next section) and then measure the count of shared words in
        THESE expanded lists... that way, would hopefully see better correlation when the same keywords are NOT
        explicitly used i.e. if section one keywords were ['child', 'person', 'travel', 'craziness', 'kid']
        and then ['appreciate', 'baby', 'form', 'adult'] ... even though they dont share words, the lists are similar!
        so hopefully if i generated the 3 nearest words I'd get 'child' in both!

    Question 4:
        Does the number of keywords contained in a section tell us about the richness of conversation? (?)
        o	Which speaker ends most of their utterances with a question
        o	Once defined what a topic is, see who is the person who first introduces a topic – is this the same person
        as above?


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
                 'top_3_counts_keywords': [], 'top_3_counts_wordvec': [],
                 'noun_list' : [], 'noun_counts' : [], 'top_3_counts_nouns':[], 'top_3_counts_nounwordvec':[]}
    """
    desired_width = 600

    pd.set_option('display.width', desired_width)

    np.set_printoptions(linewidth=desired_width)

    pd.set_option('display.max_columns', 10)

    # segments_info_df = pd.read_hdf('Saved_dfs/joe_rogan_elon_musk/200_Even_segments_info_df.h5', key='df')
    # keyword_vectors_df = pd.read_hdf('Saved_dfs/joe_rogan_elon_musk/keyword_vectors_nounderscore_fasttext_df.h5', key='df')

    # noun_list_section1 = list(segments_info_df['noun_list'][0].values)
    # noun_cnt_list_section1 = list(segments_info_df['noun_counts'][0].values)
    # noun_list_section1 = np.array(noun_list_section1)
    #
    # idxs_of_top_3_keywords = sorted(range(len(noun_cnt_list_section1)), key=lambda i: noun_cnt_list_section1[i])[-3:]
    # top_3_keywords = noun_list_section1[idxs_of_top_3_keywords]
    # print(segments_info_df['noun_list'].values)
    # print('\n\n')
    # print(segments_info_df[['length_of_segment', 'top_3_counts_nouns', 'noun_list', 'top_3_counts_keywords']].head(20))


    '''Question 1'''
    #make text lowercase to catch every example
    content = content.lower()

    #PKE first
    counter_dict = {}
    list_PKE_keywords = list(keyword_vectors_df['pke_keyw'].values)
    list_PKE_keywords = [w for w in list_PKE_keywords if w != 'nan']
    print('list_PKE_keywords: ', list_PKE_keywords)
    for word in list_PKE_keywords:
        count = len(re.findall(' ' + str(word) + ' ', content))
        counter_dict[str(word)] = count
    print('counter_dict:', counter_dict)

    plt.figure()
    ax = plt.axes()
    plt.bar(list(counter_dict.keys()), list(counter_dict.values()))

    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    plt.xlabel('PKE Keyword')
    plt.ylabel('Overall Count')
    plt.show()

    # Nouns
    noun_counter_dict = {}
    list_noun_keywords = list(keyword_vectors_df['noun_keyw'].values)
    list_noun_counts = [len(re.findall(' ' + str(noun) + ' ', content)) for noun in list_noun_keywords]

    full_dict = {}
    for x, y in zip(list_noun_keywords, list_noun_counts):
        full_dict[x] = y

    ordered_full_dict = {k: v for k, v in sorted(full_dict.items(), key=lambda item: item[1])}


    plt.figure()
    ax = plt.axes()
    print('Top n nouns in terms of usage: ', list(ordered_full_dict.items())[-20:])
    plt.bar(list(full_dict.keys())[-20:], list(full_dict.values())[-20:])

    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    plt.xlabel('Noun Keyword')
    plt.ylabel('Overall Count')
    plt.show()

    for sentence in content_sentences:
        if ' y ' in sentence:
            print(sentence)

    # Then do this for each speaker

    return


# Analyse()



