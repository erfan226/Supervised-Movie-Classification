from sklearn.feature_extraction.text import TfidfVectorizer
from utils import pickle_it, unpickle_it


class LanguageProcessor:
    def __init__(self):
        self.feature_names = None
        self.df_counts = None
        self.tfidf_vectors = None

    def word_to_vec(self, data, is_test=True):
        """
        Converts documents to tf-idf vectors.

        :param list data: Text documents to be converted
        :param bool is_test: If true then uses trained tf-idf model. Else, trains and saved tf-idf model
        :return:
        """
        if is_test:
            vectorizer = unpickle_it("tfidf_model")
            self.tfidf_vectors = vectorizer.transform(data)
        else:
            vectorizer = TfidfVectorizer()
            vectorizer.fit(data)
            self.tfidf_vectors = vectorizer.transform(data)
            pickle_it(vectorizer, "tfidf_model")
            self.feature_names = vectorizer.get_feature_names()
