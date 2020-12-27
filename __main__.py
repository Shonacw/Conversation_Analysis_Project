"""
#Breakdown of Code...
Step 1:
Step 2: Now we create a word embedding on the new set of words (containg bi and trigrams)
Step 3: Extract Keywords (Ones we'll be interested in plotting)

#Notes mentioned in code...
NOTE A
    Example of why I detect trigrams first: ['brain', 'simulation'] was a detected bigram, and
    ['deep', 'brain', 'simulation'] was a detected trigram. If I condensed all the words 'brain' and 'simulation' into
    'brain_simulation' then once I searched for the trigrams there would be none left, as it would instead have
    'deep', 'brain_simulation'.

NOTE B-
    I'm using the CBOW version of Word2Vec due to this paper
    https://www.cs.cornell.edu/~schnabts/downloads/schnabel2015embeddings.pdf

#Other stuff (for me)...
pip uninstall numpy
pip install -U numpy

# Tasks...
TODO: How does one evaluate the success of a Word Embedding? Then play around with params to optimise.. evaluation metrics
TODO: Maybe explore different methods of dimensionality reduction for plotting wordvecs (might improve layout?)
"""
from nltk.collocations import BigramCollocationFinder
from nltk.corpus import stopwords
from nltk.metrics import BigramAssocMeasures
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import TrigramAssocMeasures
from gensim.models import Word2Vec
import nltk  # Importing nltk as "import nltk.pos_tag" wasn't working (?)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from itertools import groupby
import re
from pathlib import Path
import RAKE as rake
from collections import Counter
import pke
import operator
from functools import reduce

# Load pre-processed transcript of interview between Elon Musk and Joe Rogan...
path = Path('/Users/ShonaCW/Desktop/Imperial/YEAR 4/MSci Project/Conversation_Analysis_Project/data/shorter_formatted_plain_labelled.txt')
with open(path) as f:
    content = f.read()
content_tokenized = word_tokenize(content)
print('content tokenized: ', content_tokenized)

## Step 1
words = [w.lower() for w in content_tokenized]
stopset = set(stopwords.words('english'))
filter_stops = lambda w: len(w) < 3 or w in stopset

# Extract bigrams...
bcf = BigramCollocationFinder.from_words(words)
bcf.apply_word_filter(filter_stops)                 # Ignore bigrams whose words contain < 3 chars / are stopwords
bcf.apply_freq_filter(3)                            # Ignore trigrams which occur fewer than 3 times in the transcript
bigram_list = list(list(set) for set in bcf.nbest(BigramAssocMeasures.likelihood_ratio, 20)) # Considering the top 20
print('Bigrams: ', bigram_list)

# Extract trigrams...
tcf = TrigramCollocationFinder.from_words(words)
tcf.apply_word_filter(filter_stops)                 # Ignore trigrams whose words contain < 3 chars / are stopwords
tcf.apply_freq_filter(3)                            # Ignore trigrams which occur fewer than 3 times in the transcript
trigram_list =  list(list(set) for set in tcf.nbest(TrigramAssocMeasures.likelihood_ratio, 20)) # Considering the top 20
print('Trigrams: ', trigram_list)

list_of_condensed_grams = []

# Replace Trigrams... (NOTE A)
for trigram in trigram_list:
    trigram_0, trigram_1, trigram_2 = trigram
    trigram_condensed = str(trigram_0.capitalize() + '_' + trigram_1.capitalize() + '_' + trigram_2.capitalize())
    list_of_condensed_grams.append(trigram_condensed)
    indices = [i for i, x in enumerate(content_tokenized) if x.lower() == trigram_0
               and content_tokenized[i+1].lower() == trigram_1
               and content_tokenized[i+2].lower() == trigram_2]
    for i in indices:
        content_tokenized[i] = trigram_condensed
        content_tokenized[i+1] = '-'                # Placeholders to maintain index numbering - are removed later on
        content_tokenized[i+2] = '-'

# Replace Bigrams...
for bigram in bigram_list:
    bigram_0, bigram_1 = bigram
    bigram_condensed = str( bigram_0.capitalize() + '_' + bigram_1.capitalize())
    list_of_condensed_grams.append(bigram_condensed)
    indices = [i for i, x in enumerate(content_tokenized) if x.lower() == bigram_0
               and content_tokenized[i+1].lower() == bigram_1]
    for i in indices:
        content_tokenized[i] = bigram_condensed
        content_tokenized[i+1] = '-'                # Placeholders to maintain index numbering - are removed later on


## Step 2
# Group individual words into sentences...
sents = [list(g) for k, g in groupby(content_tokenized, lambda x:x == '.') if not k]
print('\nSents before Preprocessing: ', sents)
sents_preprocessed = []

for sent in sents:
    # Make all words -except the detected bi/trigrams- lowercase
    sent_lower = [w if w in list_of_condensed_grams else w.lower() for w in sent]
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

print('Sents after Preprocessing: ', sents_preprocessed, '\n')


# Useful forms of the transcript for keyword extraction...
sents_preprocessed_flat = reduce(operator.add, sents_preprocessed)
print('\n---> sents_preprocessed_flat: ', sents_preprocessed_flat)
sents_preprocessed_flat_onestring = ' '.join(sents_preprocessed_flat)
print('---> sents_preprocessed_flat_onestring: ', sents_preprocessed_flat_onestring, '\n')


## Step 3
# RAKE...
rake_object = rake.Rake("SmartStoplist.txt") #, 2, 3, 2 #min characters in word, max number of words in phrase, min number of times it's in text
keywords = rake_object.run(content)
print("\nRAKE Keywords:", keywords)

# Counter (rubbish)...
keywords_from_counter = Counter(sents_preprocessed_flat).most_common(10)
print('Counter Keywords: ', keywords_from_counter)

# PKE...
extractor = pke.unsupervised.TopicRank()
extractor.load_document(input=sents_preprocessed_flat_onestring)
extractor.candidate_selection()
extractor.candidate_weighting()
keywords = extractor.get_n_best(30)
top_keywords = []
for i in range(len(keywords)):
  top_keywords.append(keywords[i][0])
print('Pke Keywords: ', top_keywords, '\n')

# Extract nouns (for plotting)...
words_to_plot = [word for (word, pos) in nltk.pos_tag(word_tokenize(sents_preprocessed_flat_onestring))
                 if pos[0] == 'N' and word not in ['yeah', 'yes', 'oh']]
words_to_plot = list(dict.fromkeys(words_to_plot))                      # Remove duplicate words
print('\nWords_to_plot: ', words_to_plot, '\n')

## Step 3
# Define Word2Vec Model...
model = Word2Vec(sents_preprocessed, window=20, min_count=1, workers=4, sg=0) #sg=0 for CBOW, =1 for Skig-gram
words = list(model.wv.vocab)
X = model[model.wv.vocab]
pca = PCA(n_components=2)
results = pca.fit_transform(X)
xs = results[:, 0]
ys = results[:, 1]

# Print information...
# print('Model Info: ', model)
# print('Words in Model: ', words)

# Plot Embedding...
plt.figure()
plt.title('Word2Vec Word Embedding Plots')
plt.scatter(xs, ys)
for i, word in enumerate(words):
    if word in list_of_condensed_grams or word in words_to_plot:
        plt.annotate(word, xy=(results[i, 0], results[i, 1]))
plt.show()

## Extras... (below)

# def Plot_Wordcloud(sents_preprocessed_flat, save=False):
#     wordcloud = WordCloud(
#                               background_color='white',
#                               stopwords=stop_words,
#                               max_words=100,
#                               max_font_size=50,
#                               random_state=42
#                              ).generate(str(sents_preprocessed_flat))
#     fig = plt.figure(1)
#     plt.imshow(wordcloud)
#     plt.axis('off')
#     plt.show()
#
#     if save:
#         fig.savefig("WordCloud.png", dpi=900)
#     return

# Plot_Wordcloud(sents_preprocessed_flat_onestring)
