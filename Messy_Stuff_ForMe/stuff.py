def find_closest_embeddings(embeddings_dict):
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))

def Check_Embedding(embeddings_dict):
    print(find_closest_embeddings(embeddings_dict["king"])[1:6])
    print(find_closest_embeddings(embeddings_dict["twig"] - embeddings_dict["branch"] + embeddings_dict["hand"])[:5])
    #model.most_similar("woman")   model.similarity("girl", "woman")
    return

def Convert_bin_to_txt(look_at_vocab=False):
    """
    Function to convert pre-trained word vector files from .bin to .txt format so that can explore vocabulary.
    Only used this function once, but keeping in case I need to perform a similar conversion .
    """
    path_in = 'Google_WordVectors/GoogleNews-vectors-negative300.bin'
    path_out = 'Google_WordVectors/GoogleNews-vectors-negative300.txt'
    model = KeyedVectors.load_word2vec_format(path_in, binary=True)
    model.save_word2vec_format(path_out, binary=False)

    if look_at_vocab:
        model_google = KeyedVectors.load_word2vec_format(path_out, binary=True)
        words = list(model_google.wv.vocab)[:100000]            # Only looking at first hundred thousand
        phrases = [word for word in words if '_' in word]
        print('\nVocab containing an underscore from Google model:', phrases)
    return


def Gensim_Phrase(content):
    """
    Not in use. Taken from Inference.ipnb, when I was messing around with phrase extraction.
    """
    phrases = Phrases(content, min_count=2, threshold=3)
    for phrase in phrases[content]:
        print(phrase)
    # Export a FrozenPhrases object that is more efficient but doesn't allow any more training.
    # frozen_phrases = phrases.freeze()
    # print(frozen_phrases[sent]) #give it a sentence like

    # N-GRAM EXTRACTION

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