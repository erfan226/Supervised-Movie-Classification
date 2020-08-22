from utils import pickle_it, unpickle_it


class NaiveBayes:
    """
    Naive Bayes Algorithm.

    Attributes:
        class_probabilities (list): Probability of each class (p(y))
        features_prob (list): Probability of each feature (p(x))
        estimated_classes (list): A list of integer(s) which shows the predicted label for document(s)
    """
    def __init__(self):
        self.class_probabilities = []
        self.features_prob = []
        self.estimated_classes = []

    def reset(self):
        """
        Reset predicted data list to avoid confusions in next iterations.

        :return:
        """
        self.estimated_classes = []

    def estimate_parameters(self, docs, tokenized_docs, feature_names):
        """
        Estimates probability of each class, each feature in each class and saves the trained parameters.

        :param list docs: Documents with defined classes
        :param tokenized_docs: Tokenized documents
        :param feature_names: Feature vector which is computed with scikit-learn TfidfVectorizer module
        :return:
        """
        self.class_probability(docs)
        self.feature_probability(feature_names, tokenized_docs)
        pickle_it(self, "naive_bayes_model")
        print("Parameter estimation complete. Predicting trained documents...")

    def class_probability(self, class_docs):
        """
        Takes the whole document and assigns a probability to each class.

        :param list class_docs: All classes with their documents
        :return
        """
        total_docs = sum([len(docs) for docs in class_docs])
        for docs in class_docs:
            self.class_probabilities.append(len(docs) / total_docs)

    def feature_probability(self, features, docs):
        """
        Takes all of the features for each class, then assigns a probability to it.
        These probabilities are accessible through features_prob attribute.

        :param list features: Extracted features
        :param list docs: List of tokenized documents for each class
        """
        for cls in docs:
            temp_prob = []
            cls_total_words = sum([len(doc) for doc in cls])  # Total words from all docs in each class
            for feature in features:
                feature_count_by_cls = sum([doc.count(feature) for doc in cls])
                word_prob = (feature_count_by_cls + 1) / (cls_total_words + 1)  # Add-one smoothing is used to avoid
                # zero-probability when the feature is not observed in one class but in the other
                temp_prob.append(word_prob)
            self.features_prob.append(temp_prob)

    def probability_calculation(self, vectors, classes_prob, features_prob):
        """
        Takes a test vector and based on the probability of each feature for each class,
        multiplies them with the probability of their class, then calculates the probability of the given vector.

        :param list vectors: Vector to be tested
        :param list classes_prob: Probability of all classes
        :param list features_prob: Probability of all features
        :return int: Value of the class with the biggest probability
        """
        cls_prob = []
        predicted_class = 0
        for i, class_prob in enumerate(classes_prob):
            value = 1
            for j, token in enumerate(vectors):
                if token != 0:
                    value = features_prob[i][j] * value
            cls_prob.append(value * class_prob)
            predicted_class = cls_prob.index(max(cls_prob))
            # Making sure of unseen data not causing problems
            # if 1 not in vector:
            #     value = 0
        return predicted_class

    # May result in floating-point underflow(?)
    def predict_class(self, test_vectors):
        """
        Takes vectored documents, loads trained model, and for each vector, predicts the class with highest probability.

        :param numpy.ndarray test_vectors: List of vector(s)
        :return list: Predicted class for each test vector
        """
        self.reset()  # Reset attribute so outputs of different calls (test/train) do not mix up!
        trained_model = unpickle_it("naive_bayes_model")
        for vector in test_vectors:
            self.estimated_classes.append(self.probability_calculation(vector, trained_model.class_probabilities, trained_model.features_prob))
        return self.estimated_classes
