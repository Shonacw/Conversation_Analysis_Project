
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
from wordcloud import WordCloud
import re

import RAKE as rake
from collections import Counter
import pke
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




## Plot WordCloud

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

#Important, define
sents_preprocessed_flat = reduce(operator.add, sents_preprocessed)
sents_preprocessed_flat_onestring = ' '.join(sents_preprocessed_flat)


## Extract Keywords (Ones we'll be interested in plotting)
#RAKE
rake_object = rake.Rake("SmartStoplist.txt") #, 2, 3, 2 #min characters in word, max number of words in phrase, min number of times it's in text
keywords = rake_object.run(content)
print("\nRAKE Keywords:", keywords)

#Counter (rubbish)
keywords_from_counter = Counter(sents_preprocessed_flat).most_common(10)
print('Counter Keywords: ', keywords_from_counter)

# pke
extractor = pke.unsupervised.TopicRank()
extractor.load_document(input=sents_preprocessed_flat_onestring)
extractor.candidate_selection()
extractor.candidate_weighting()
keywords = extractor.get_n_best(30)
top_keywords = []
for i in range(len(keywords)):
  top_keywords.append(keywords[i][0])
print('Pke top_keywords: ', top_keywords, '\n')

# Plot_Wordcloud(sents_preprocessed_flat_onestring)



##Model

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

