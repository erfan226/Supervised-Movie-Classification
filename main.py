import pandas as pd
from utils import convert_to_labels, prep_docs, merge_documents, prep_dataset, save_test_dataset, display_info, cli_exc, shuffle_test_data, display_result, read_test_labels, plot_test_accuracy
from languageprocessor import LanguageProcessor
from naive_bayes import NaiveBayes
import numpy as np
from constants import PROCESSED_DATA, TEST_DATA

# Configurations for CLI
args = cli_exc()

pre_process = args.prep
doc_limit = args.doclimit  # To avoid bias, equalize number of docs for each class/genre
word_limit = args.wordlimit
mode = args.mode
n = args.iter

# Creating necessary instances
lp = LanguageProcessor()
nb = NaiveBayes()

# Pre-process data if "--prep" is passed from CLI. Preprocessed data should already be in Data directory
if pre_process:
    prep_dataset("wiki_movie_plots_deduped.csv", PROCESSED_DATA)

movie_data = pd.read_csv("Data/movie_plots_processed.csv")  # Read preprocessed data-set
grouped_docs = movie_data.groupby('Genre')["Plot"].apply(list)  # Group docs by genre (label)

if mode == "train":
    true_train_labels = convert_to_labels(grouped_docs, end=doc_limit)  # Convert docs in each class with index of its genre
    save_test_dataset(grouped_docs, doc_limit, TEST_DATA)  # Save the rest of data-set for test
    docs, tokenized_docs = prep_docs(grouped_docs, doc_limit, word_limit)  # Each class is separated for parameter est.
    merged_docs = merge_documents(docs)  # Convert docs to a 1-D array to be ready for tf-idf conversion
    display_info(movie_data, merged_docs, word_limit)
    lp.word_to_vec(merged_docs, is_test=False)  # Create feature vectors & convert to tf-idf vectors
    nb.estimate_parameters(docs, tokenized_docs, lp.feature_names)  # Est. parameters for naive bayes algorithm
    vectors = np.asarray(lp.tfidf_vectors.todense())
    pred_labels = nb.predict_class(vectors)  # Predict
    display_result(true_train_labels, pred_labels, movie_data)

elif mode == "test":
    acc_history = [0]
    for i in range(n):
        true_test_labels = read_test_labels(grouped_docs)
        print("Testing with {0} documents...".format(doc_limit))
        test_data = pd.read_csv("Data/Test_data.csv")
        test_data = test_data["Plot"].values.tolist()
        t_label, test_data = shuffle_test_data(true_test_labels, test_data)  # Shuffle sorted data
        lp.word_to_vec(test_data[0:doc_limit], is_test=True)  # Convert to tf-idf vectors
        test_vectors = np.asarray(lp.tfidf_vectors.todense())
        pred_test_labels = nb.predict_class(test_vectors)  # Predict
        acc_history.append(display_result(t_label[0:doc_limit], pred_test_labels, movie_data))
    if n > 1:
        plot_test_accuracy(acc_history, n + 1)
else:
    print("Possible options are train & test. Run again like example:\nExample: python main.py test")
