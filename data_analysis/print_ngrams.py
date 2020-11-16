import argparse
import pickle
import os

import logging
from the_pile.logger import setup_logger_tqdm
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='n-gram statistics')
parser.add_argument("-dir", "--working_directory", default="")

def main(working_directory):
    ngrams_pickle_path = os.path.join(working_directory, "ngrams.pkl")

    ngrams = pickle.load(open(ngrams_pickle_path, "rb"))

    for i, (ngram, count) in enumerate(ngrams.items()):
        if i == 10:
            break

        logger.info(f"{ngram} : {count}")

if __name__ == '__main__':
    setup_logger_tqdm()

    args = parser.parse_args()
    main(args.working_directory)