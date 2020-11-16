import argparse
import pickle
import os
import csv

import logging
from the_pile.logger import setup_logger_tqdm
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='n-gram statistics')
parser.add_argument("-dir", "--working_directory", default="")

def trim_ngram_dict(n_grams):
    logger.info("Split finished, trimming dict.")
    sorted_dict = {key: val for key, val in sorted(n_grams.items(), key = lambda ele: ele[1], reverse = True)}     
    trimmed_dict = {}
    for i, (n_gram, count) in enumerate(sorted_dict.items()):
        if i == 1000:
            break
        trimmed_dict[n_gram] = count

    return trimmed_dict

def main(working_directory):
    ngrams_pickle_path = os.path.join(working_directory, "ngrams.pkl")

    ngrams = pickle.load(open(ngrams_pickle_path, "rb"))

    # for i, (ngram, count) in enumerate(ngrams.items()):
    #     if i == 10:
    #         break

    #     logger.info(f"{ngram} : {count}")

    trimmed_ngrams = trim_ngram_dict(ngrams)

    csv_path = os.path.join(working_directory, "ngrams.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['ngram', 'count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for ngram, count in trimmed_ngrams.items():
            writer.writerow({'ngram': ngram, 'count': count})

if __name__ == '__main__':
    setup_logger_tqdm()

    args = parser.parse_args()
    main(args.working_directory)