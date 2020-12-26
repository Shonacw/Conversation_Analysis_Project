
from nltk.collocations import BigramCollocationFinder
from nltk.corpus import stopwords
from nltk.metrics import BigramAssocMeasures
from pathlib import Path
from nltk.tokenize import word_tokenize, sent_tokenize

from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import TrigramAssocMeasures

from itertools import groupby
import string

from gensim.models import Word2Vec

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import re

import RAKE as rake
import operator
from functools import reduce

path = Path('/Users/ShonaCW/Desktop/Imperial/YEAR 4/MSci Project/Conversation_Analysis_Project/data/shorter_formatted_plain_labelled.txt')
with open(path) as f:
    content = f.read()
print('content', content)
content_tokenized = word_tokenize(content)
print('content tokenized', content_tokenized)

#Extract bigrams
words = [w.lower() for w in content_tokenized]
print('\nwords:', words)

bcf = BigramCollocationFinder.from_words(words)
stopset = set(stopwords.words('english'))
filter_stops = lambda w: len(w) < 3 or w in stopset
bcf.apply_word_filter(filter_stops)
bigram_list = list(list(set) for set in bcf.nbest(BigramAssocMeasures.likelihood_ratio, 20))
print('Bigrams: ', bigram_list)

#Extract trigrams
# words = [w.lower() for w in content_tokenized]
tcf = TrigramCollocationFinder.from_words(words)
tcf.apply_word_filter(filter_stops)
tcf.apply_freq_filter(3)
trigram_list =  list(list(set) for set in tcf.nbest(TrigramAssocMeasures.likelihood_ratio, 20))
print('Trigrams: ', trigram_list)

list_of_condensed_grams = []

# Replace Trigrams first
for trigram in trigram_list:
    trigram_0, trigram_1, trigram_2 = trigram
    trigram_condensed = str(trigram_0.capitalize() + trigram_1.capitalize() + trigram_2.capitalize())
    list_of_condensed_grams.append(trigram_condensed)
    indices = [i for i, x in enumerate(content_tokenized) if x.lower() == trigram_0 and content_tokenized[i+1].lower() == trigram_1
               and content_tokenized[i+2].lower() == trigram_2]
    # print('\n =====trigram', trigram)
    # print('trigram_condensed', trigram_condensed)
    # print(indices)
    for i in indices:
        # print(content_tokenized[i], content_tokenized[i+1], content_tokenized[i+2])
        content_tokenized[i] = trigram_condensed
        content_tokenized[i+1] = '-' # doing this to maintain index numbers! after replaces all bi/trigrams remove these
        content_tokenized[i+2] = '-'
        # print(content_tokenized[i], content_tokenized[i+1], content_tokenized[i+2])

# Replace Bigrams
for bigram in bigram_list:
    bigram_0, bigram_1 = bigram
    bigram_condensed = str( bigram_0.capitalize() + bigram_1.capitalize())
    list_of_condensed_grams.append(bigram_condensed)
    indices = [i for i, x in enumerate(content_tokenized) if x.lower() == bigram_0 and content_tokenized[i+1].lower() == bigram_1]

    # print('\n =====bigram', bigram)
    # print('bigram_condensed', bigram_condensed)
    # print(indices)
    for i in indices:
        # print(content_tokenized[i], content_tokenized[i+1])
        content_tokenized[i] = bigram_condensed
        content_tokenized[i+1] = '-' # doing this to maintain index numbers! after replaces all bi/trigrams remove these
        # print(content_tokenized[i], content_tokenized[i + 1])

"""
NOTES

# make sure detecting even if they're capitalised in the text
when two words inside a trigram are also a bigram... 
want to firstly detect all the trigrams and then detect remaining bigrams
"""

## Now we create a word embedding on the new set of words (containg bi and trigrams

"""
going to try using preprocessed, however this won't work on the transcript as it has non-word 'words' like
NeuralNetworks and 'ConstructionZone'
"""

# Split into sentences
sents = [list(g) for k, g in groupby(content_tokenized, lambda x:x == '.') if not k]
print('Sents before Preprocessing: ', sents)

sents_preprocessed = []

for sent in sents:
    # Make all words -except those found to be ngrams- lowercase
    sent_lower = [w if w in list_of_condensed_grams else w.lower() for w in sent] #for w in sent if w not in list_of_condensed_grams]
    # Join into one string so can use reg expressions
    my_lst_str = ' '.join(map(str, sent_lower))
    # Remove numbers
    result1 = re.sub(r'\d+', '', my_lst_str)
    # Remove Punctuation
    result2 = re.sub(r'[^\w\s]', '', result1)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(result2)
    result = [i for i in tokens if i not in stop_words]

    # Stemming?
    # Lemmatization ?

    sents_preprocessed.append(result)

print('sents_preprocessed: ', sents_preprocessed)



##Extract Keywords (Ones we'll be interested in plotting)
#RAKE
rake_object = rake.Rake("SmartStoplist.txt") #, 2, 3, 2 #min characters in word, max number of words in phrase, min number of times it's in text
keywords = rake_object.run(content)
print("\nRAKE Keywords:", keywords)

#Counter
from collections import Counter
sents_preprocessed_flat = reduce(operator.add, sents_preprocessed)
keywords_from_counter = Counter(sents_preprocessed_flat).most_common(10)
print('Counter Keywords: ', keywords_from_counter, '\n')


## Plot WordCloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
def Plot_Wordcloud(sents_preprocessed_flat, save=False):
    wordcloud = WordCloud(
                              background_color='white',
                              stopwords=stop_words,
                              max_words=100,
                              max_font_size=50,
                              random_state=42
                             ).generate(str(sents_preprocessed_flat))
    fig = plt.figure(1)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

    if save:
        fig.savefig("WordCloud.png", dpi=900)
    return

sents_preprocessed_flat_onestring = ' '.join(sents_preprocessed_flat)
# Plot_Wordcloud(sents_preprocessed_flat_onestring)

## TD-IDF     https://medium.com/analytics-vidhya/automated-keyword-extraction-from-articles-using-nlp-bfd864f41b34
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction.text import CountVectorizer
#
#
# cv = CountVectorizer(max_df=0.8, stop_words=stop_words, max_features=10000, ngram_range=(1,3))
# X = cv.fit_transform(sents_preprocessed_flat_onestring)
#
# tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
# tfidf_transformer.fit(X)
# # get feature names
# feature_names = cv.get_feature_names()
#
# # generate tf-idf for the given document
# tf_idf_vector = tfidf_transformer.transform(cv.transform([X]))
#
# # Function for sorting tf_idf in descending order
# from scipy.sparse import coo_matrix
#
#
# def sort_coo(coo_matrix):
#     tuples = zip(coo_matrix.col, coo_matrix.data)
#     return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
#
#
# def extract_topn_from_vector(feature_names, sorted_items, topn=10):
#     """get the feature names and tf-idf score of top n items"""
#
#     # use only topn items from vector
#     sorted_items = sorted_items[:topn]
#
#     score_vals = []
#     feature_vals = []
#
#     # word index and corresponding tf-idf score
#     for idx, score in sorted_items:
#         # keep track of feature name and its corresponding score
#         score_vals.append(round(score, 3))
#         feature_vals.append(feature_names[idx])
#
#     # create a tuples of feature,score
#     # results = zip(feature_vals,score_vals)
#     results = {}
#     for idx in range(len(feature_vals)):
#         results[feature_vals[idx]] = score_vals[idx]
#
#     return results
#
#
# # sort the tf-idf vectors by descending order of scores
# sorted_items = sort_coo(tf_idf_vector.tocoo())
# # extract only the top n; n here is 10
# keywords = extract_topn_from_vector(feature_names, sorted_items, 5)
#
# # now print the results
#
# print("\nKeywords:")
# for k in keywords:
#     print(k, keywords[k])

##Plot


##TD-IDF #2
# print('========================================')
# from nltk import tokenize
# from operator import itemgetter
# import math
#
# # Step 1 : Find total words in the document
# doc = ' '.join(words) #sents_preprocessed_flat_onestring
# total_words = doc.split()
# print('total_words:', total_words)
# total_word_length = len(total_words)
# print('total_word_length')
# print(total_word_length)
#
# # Step 2 : Find total number of sentences
# total_sentences = tokenize.sent_tokenize(doc)
# print('total_sentences:', total_sentences)
# total_sent_len = len(total_sentences)
# print('total_sent_len')
# print(total_sent_len)
#
# # Step 3: Calculate TF for each word
# tf_score = {}
# for each_word in total_words:
#     each_word = each_word.replace('.', '')
#     each_word = each_word.replace(',', '')
#     each_word = each_word.replace('?', '')
#     if each_word not in stop_words:
#         if each_word in tf_score:
#             tf_score[each_word] += 1
#         else:
#             tf_score[each_word] = 1
# print('tf_score')
# print(tf_score)
#
# # Dividing by total_word_length for each dictionary element
# tf_score.update((x, y/int(total_word_length)) for x, y in tf_score.items())
#
# print('tf_score after divided by total word length')
# print(tf_score)
#
# # Check if a word is there in sentence list
# def check_sent(word, sentences):
#     final = [all([w in x for w in word]) for x in sentences]
#     sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
#     return int(len(sent_len))
#
# # Step 4: Calculate IDF for each word
# idf_score = {}
# for each_word in total_words:
#     each_word = each_word.replace('.','')
#     if each_word not in stop_words:
#         if each_word in idf_score:
#             idf_score[each_word] = check_sent(each_word, total_sentences)
#         else:
#             idf_score[each_word] = 1
#
# print('idf_score')
# print(idf_score)
#
# # Performing a log and divide
# idf_score.update((x, math.log(int(total_sent_len)/y)) for x, y in idf_score.items())
#
# print('idf_score after log and divide')
# print(idf_score)
#
# # Step 5: Calculating TF*IDF
# tf_idf_score = {key: tf_score[key] * idf_score.get(key, 0) for key in tf_score.keys()}
# print('tf_idf_score')
# print(tf_idf_score)
#
# # Get top N important words in the document
# def get_top_n(dict_elem, n):
#     result = dict(sorted(dict_elem.items(), key = itemgetter(1), reverse = True)[:n])
#     return result
#
# print('get_top_n(tf_idf_score, 5)')
# print(get_top_n(tf_idf_score, 5))
#
# # RUBBISH. PICKS UP: {'’': 0.037814859794761284, 'like': 0.017613957156698272, 'yeah': 0.008016156649331016, '…': 0.0064758945935541054, 'know': 0.005281128785821303}
# print('========================================')

##Model


##
model = Word2Vec(sents_preprocessed, min_count=1)
print('Model Info: ', model)
words = list(model.wv.vocab)
print('Words in Model: ', words)
X = model[model.wv.vocab]
pca = PCA(n_components=2)
results = pca.fit_transform(X)
xs = results[:, 0]
ys = results[:, 1]

plt.figure()
plt.title('Word2Vec Word Embedding Plots')
plt.scatter(xs, ys)
for i, word in enumerate(words):
    if word in list_of_condensed_grams:
        plt.annotate(word, xy=(results[i, 0], results[i, 1]))
plt.show()






