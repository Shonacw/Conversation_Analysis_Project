from top2vec import Top2Vec
from pathlib import Path
import re
from sklearn.datasets import fetch_20newsgroups

newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
documents = newsgroups.data
print('==documents: ', documents)
print('==len(documents): ', len(documents))

print('==documents[0]: ', documents[0])
print('==len of doc 0: ', len(documents[0]))
print(type(documents))


def Create_Docs():
    # Documents must be a list of strings. For me, each is a segment of the dialogue.
    path = Path('/Users/ShonaCW/Desktop/Imperial/YEAR 4/MSci Project/Conversation_Analysis_Project/data/shorter_formatted_plain_labelled.txt')
    with open(path) as f:
        content = f.read()
    content_list = re.split('========,9,title.', content)
    print('Generated list of strings..')
    print('number of "Documents": ', len(content_list))

    return content_list
#
# Documents = Create_Docs()
# print('Documents', Documents)
# print(len(Documents))
print('Building Model...')
model = Top2Vec(documents, embedding_model='universal-sentence-encoder')
model.get_num_topics()
print('Obtained Number of Topics')
topic_sizes, topic_nums = model.get_topic_sizes()
print('topic_sizes, topic_nums:', topic_sizes, topic_nums)