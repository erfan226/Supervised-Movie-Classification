import argparse
from constants import DATA_DIR, TEST_SPLIT_IDX
import csv
import pickle
from preprocessor import tokenizer, process_data
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

def prep_dataset(fn, csv_fn):
    """
    Reads data-set, removes empty/useless rows and finally pre-process it.

    :param str fn: Name of data-set file to read
    :param csv_fn: Name of file which pre-processed data would be written into
    :return:
    """
    movie_data = pd.read_csv(''.join([DATA_DIR, fn]))
    counts = movie_data['Genre'].value_counts()
    rows_to_remove = counts[counts <= 1000].index
    movie_data['Genre'].replace(rows_to_remove, "Null", inplace=True)  # To remove Genres with low frequency (<1000)
    movie_data = movie_data[movie_data.Genre != "Null"]
    movie_data = movie_data[movie_data.Genre != "unknown"]
    movie_data = movie_data[["Genre", "Plot"]]
    processed_data = process_data(movie_data["Plot"])
    save_csv(processed_data, csv_fn, ["Plot", "Genre"], movie_data["Genre"].tolist())


def save_file(data, fn, mode="w"):
    """
    Saves the given data to the specified file.

    :param list data: Data to be saved
    :param str fn: Name of the file & extension to be saved (Ex: File.txt)
    :param str mode: Mode of file writing (Ex: w, a)
    :return:
    """
    file = open(''.join([DATA_DIR, fn, ".txt"]), mode)
    file.writelines(data)
    file.close()


def save_csv(data, fn, cols, labels=None):
    """
    Saves the given data to the specified file.

    :param list data: Data to be saved
    :param str fn: Name of the file & extension to be saved (Ex: File.txt)
    :param list cols: Column names (Ex: ["col1", "col2"]
    :param list labels: Corresponding labels of the given data
    :return:
    """
    with open(''.join([DATA_DIR, fn, ".csv"]), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(cols)
        if labels:
            for i, item in enumerate(data):
                writer.writerow([item[0], labels[i]])
        else:
            for i, item in enumerate(data):
                writer.writerow([item])


def read_file(fn, limit=None):
    """
    Read the data file and converts to a list.

    :param str fn: Name of file with data in it
    :param int limit: Limit each document to a number of words
    :return: List of documents
    """
    data = []
    try:
        with open(''.join([DATA_DIR, fn, ".txt"])) as file:
            if limit:
                for item in file:
                    item = item.strip().split()[:limit]
                    item = " ".join(item)
                    data.append(item)
            else:
                for item in file:
                    data.append(item.strip())
        return data
    except FileNotFoundError:
        print("File not found at", fn)
        exit(1)


def pickle_it(data, fn):
    """
    Saves given data as a binary file using Pickle library.

    :param data: Data to be saved
    :param str fn: Name of file to save
    :return:
    """
    with open(''.join([DATA_DIR, fn]), 'wb') as f:
        pickle.dump(data, f)


def unpickle_it(fn):
    """
    Loads a binary file using Pickle library and converts it to corresponding format.

    :param str fn: Name of file to load
    :return: Pickled data in the given format at the time of save
    """
    with open(''.join([DATA_DIR, fn]), 'rb') as f:
        data = pickle.load(f)
    return data


def convert_to_labels(data, start=0, end=None):
    """
    Converts each data value to the label of its class, given class's index.

    :param list data: List of classes with documents
    :param int start: Starting index to read & convert data from that position
    :param int end: Limiting index to stop converting data
    :return: Labels of documents as a list
    """
    labels = []
    for i, data in enumerate(data):
        for item in data[start:end]:
            labels.append(i)
    return labels


def prep_docs(classes, doc_limit=1000, word_limit=150):
    """
    Prepares data by limiting and converting to the right format for algorithms to work.

    :param list classes: Documents in each class
    :param int doc_limit: Limit documents of each class to a certain number. Can not be more than available data-set
    :param int word_limit: Limit word tokens in each document to a certain number
    :return: List of documents & tokenized documents at an equal number
    """
    tokenized_docs = []
    docs_equalized = []
    for cls in classes:
        temp_tokenized_docs = []
        temp_class_docs = []
        for doc in cls[0:doc_limit]:
            temp_tokenized_docs.append(tokenizer(doc))  # Tokenize docs in each class
            temp_class_docs.append(" ".join(tokenizer(doc)[0:word_limit]))  # Limit each doc to a number of words
        tokenized_docs.append(temp_tokenized_docs)
        docs_equalized.append(temp_class_docs)
    return docs_equalized, tokenized_docs


def merge_documents(classes, start=0):
    """
    Converts a 2-D array to a 1-D array.

    :param list classes: List of classes with their documents
    :param int start: Starting index to convert from that position
    :return: A 1-D array consisting all given documents
    """
    merged_docs = []
    for docs in classes:
        for doc in docs[start:]:
            merged_docs.append(doc)
    return merged_docs


def save_test_dataset(data, idx, fn):
    """
    Saves unused data-set in training phase for testing phase.

    :param list data: List of classes with their documents
    :param int idx: Start index to convert from that position (An index position where training split occurred)
    :param str fn: Name of file to save test data
    :return:
    """
    data = merge_documents(data, idx)
    save_csv(data, fn, ["Plot"])
    save_file([str(idx)], TEST_SPLIT_IDX)  # Save start index position for test data-set


def read_test_labels(data):
    """
    Reads documents and converts each data value to the label of its class, given class's index.

    :param list data: Grouped documents in each class
    :return: List of labels of documents
    """
    idx = int(read_file(TEST_SPLIT_IDX)[0])
    labels = convert_to_labels(data, start=idx)
    return labels


def display_info(data, train_data, w_limit):
    """
    Displays some info regarding configured options in running code.

    :param list data: Total data to compute its size
    :param list train_data: Used portion of data in training
    :param int w_limit: Limit of word trim option
    :return:
    """
    train_num = len(train_data)
    test_num = len(data) - train_num
    print(
        "Using {0} documents for training and saving {1} documents for test. Each document is trimmed to {2} words for "
        "training.".format(train_num, test_num, w_limit))


def display_result(t_labels, p_labels, data, n=None):
    """
    Displays result of algorithm either in training or test.

    :param list t_labels: True labels
    :param list p_labels: Predicted labels
    :param pandas.core.frame.DataFrame data: The data-set
    :param int n: Number of iterations in test phase
    :return:
    """
    accuracy_percent = round(accuracy_score(t_labels, p_labels) * 100, 2)
    stats = precision_recall_fscore_support(t_labels, p_labels, average='weighted')

    print("\nResults:\nPredicted labels: {0}".format(p_labels))
    print("True labels: {0}".format(t_labels))
    print("Accuracy: {0}%".format(accuracy_percent))
    print("Precision: {0}%\nRecall: {1}%\nFScore: {2}%".format(round(stats[0]*100), round(stats[1]*100), round(stats[2]*100)))
    print("\nLabels guide:")

    labels_group = data.groupby('Genre')
    for i, label in enumerate(labels_group.groups):
        print("{0} => {1}".format(i, label))

    print("Confusion Matrix:\n", confusion_matrix(t_labels, p_labels))
    return accuracy_percent


def cli_exc():
    """
    Configurations fo CLI to run program with. Will be set to default values if run in other environments.

    :return:
    """
    parser = argparse.ArgumentParser(
        description="Manual to use this script:", usage="python main.py mode [--doclimit] [--wordlimit]")
    parser.add_argument('mode', type=str, nargs='?', default="test", help='Estimate the parameters for algorithm to work with or predict data')
    parser.add_argument('-l', '--doclimit', type=int, default=10, help='Limits number of docs for each class to process (default=10)')
    parser.add_argument('-w', '--wordlimit', type=int, default=150,
                        help='Limits number of words in each doc to process (default=150)')
    parser.add_argument('-n', '--iter', type=int, default=1, help='Number of times to test shuffled test data. Will plot prediction accuracy if > 1')

    parser.add_argument('--prep', type=bool, default=False, help='Pre-process data-set (default=False)')
    args = parser.parse_args()
    return args


def shuffle_test_data(t_labels, test_data):
    """
    Shuffles data and corresponding labels.

    :param list t_labels: Labels of documents
    :param list test_data: Documents
    :return: Shuffled version of labels and documents
    """
    map_idx_position = list(zip(t_labels, test_data))
    random.shuffle(map_idx_position)
    shuffled_label, shuffled_data = zip(*map_idx_position)
    return shuffled_label, shuffled_data


def plot_test_accuracy(accuracy, iters):
    """
    Plots accuracy of each iteration.

    :param list accuracy: Accuracy of each iteration
    :param int iters: Number of iterations
    :return:
    """
    iters = list(range(0, iters))
    plt.figure()
    plt.plot(iters, accuracy)
    plt.suptitle('Accuracy of {0} tests'.format(len(iters)-1))
    plt.show()
