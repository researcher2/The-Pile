import os
import argparse
import pickle

import tqdm
import nltk
from nltk.util import ngrams
from nltk.probability import FreqDist
from tqdm_multiprocess import TqdmMultiProcessPool

from the_pile.datasets import CommonCrawlDataset

import logging
from the_pile.logger import setup_logger_tqdm
logger = logging.getLogger(__name__)

# Multiprocessing
def extract_ngrams(data, num, tqdm_func, global_tqdm):
    n_grams = ngrams(nltk.word_tokenize(data), num)
    return [ ' '.join(grams) for grams in n_grams]

def process_batch(pool, batch, n_value):
    tasks = []
    for document in batch:
        task = (extract_ngrams, (document, n_value))
        tasks.append(task)

    on_done = lambda _ : None
    on_error = lambda _ : None
    documents = pool.map(None, tasks, on_error, on_done)

    ngrams = {}
    for document_ngrams in documents:
        for n_gram in document_ngrams:
            if n_gram in ngrams:
                ngrams[n_gram] += 1
            else:
                ngrams[n_gram] = 0

    res = {key: val for key, val in sorted(ngrams.items(), key = lambda ele: ele[1], reverse = True)} 
    for n_gram, count in res.items():
        logger.info(f"{count}: {n_gram}")


def main(working_directory, process_count, n_value):
    nltk.download('punkt')

    cc_dataset = CommonCrawlDataset()

    batch_size = 1000
    batch = []
    pool = TqdmMultiProcessPool(process_count)

    with tqdm.tqdm(total=cc_dataset.num_docs(), dynamic_ncols=True, unit="docs") as progress:
        for document, meta in cc_dataset.documents():

            batch.append(document)

            if len(batch) == batch_size:
                process_batch(pool, batch, n_value)
                batch = []
                progress.update(batch_size)
                break

        if len(batch) != 0:
            process_batch(pool, batch, n_value)
            progress.update(len(batch))

parser = argparse.ArgumentParser(description='n-gram statistics')
parser.add_argument("-dir", "--working_directory", default="")
parser.add_argument("-procs", "--process_count", type=int, default=4)
parser.add_argument("-n", "--n_value", type=int, default=13)

if __name__ == '__main__':
    logfile_path = "ngrams.log"
    setup_logger_tqdm(logfile_path)

    args = parser.parse_args()
    main(args.working_directory, args.process_count, args.n_value)