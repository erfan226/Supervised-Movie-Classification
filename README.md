# Movie Classifier

A simple class project to classify the genre of movies by their plots. 4000 documents were used to train the current model (in Data directory.)


#### Path to training files:
All data are located in the data directory. For example the original data-set that is used is at: `Data/wiki_movie_plots_deduped.csv`

## Instructions:
In Terminal/CLI `cd` to the project folder, then do a `pip install -r requirements.txt` to install the required packages. After that, run program with this command: `python main.py`.

#### CLI Guide:
usage: `python main.py mode [--doclimit] [--wordlimit]`

##### positional arguments:
  mode:                  Estimate the parameters for algorithm to work with and/or
                        predict data. Options are `test` and `train`

##### optional arguments:
  -h, --help:            show help message and exit<br>
  -l, --doclimit:        Limits number of docs for each class to process
                        (default=10)<br>
  -w, --wordlimit:
                        Limits number of words in each doc to process
                        (default=150)<br>
  -n, --iter:  Number of times to test shuffled test data. Will plot prediction accuracy if > 1<br>
  --prep:           Pre-process original data-set (default=False)<br>

- To train model on documents, run `python train --l i --w j`. i is the number of documents for each class that would be use in training, so 200 would result in 800 as there are 4 class. j specifies how many words should documents be trimmed to.
- To test the trained model, run `python test --l i -n j`. i is the total number of documents to be tested. j is the number of iterations to run tests with shuffled data and finally to plot the accuracies.
