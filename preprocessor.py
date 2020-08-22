import string
import re
from stopwords import stops
# import nltk
# from langdetect import detect


def tokenizer(text):
    """
    Tokenizes text strings to a list of words.

    :param str text: Text of documents
    :return: List of words from document
    """
    text = text.rstrip().lower().replace("\r\n", " ")
    tokenized_text = text.split(" ")
    return tokenized_text


def remove_punctuation(text):
    """
    Removes punctuations inside tokens.

    :param list text: List of tokens of document
    :return: Punctuation-free list of tokens
    """
    text = [words.translate(str.maketrans("", "", string.punctuation)) for words in text]
    return text


def remove_stops(text):
    """
    Removes stopwords from a list of stopwords.

    :param list text: Tokens of document
    :return: Stopword-free list of tokens
    """
    sentence = []
    for word in text:
        if word not in stops:
            word = re.sub(r"\d", "", word)
            sentence.append(word)
    return sentence


def process_data(data):
    """
    Tokenizes, removes stopwords, and removes punctuations of given documents.

    :param list data: Documents to be pre-processed
    :return: Pre-processed documents
    """
    processed_data = []
    for text in data:
        txt = tokenizer(text)
        txt = remove_stops(txt)
        txt = remove_punctuation(txt)
        processed_data.append([" ".join(txt)])
    return processed_data
