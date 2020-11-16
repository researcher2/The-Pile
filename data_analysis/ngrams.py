import os
import argparse
import pickle
import math
import sys

import tqdm
import nltk
from nltk.util import ngrams
from nltk.probability import FreqDist
from tqdm_multiprocess import TqdmMultiProcessPool

from the_pile.datasets import CommonCrawlDataset, OpenWebText2Dataset

import logging
from the_pile.logger import setup_logger_tqdm
logger = logging.getLogger(__name__)

# Multiprocessing
def extract_ngrams(data, num, tqdm_func, global_tqdm):
    n_grams = ngrams(nltk.word_tokenize(data), num)
    return [ ' '.join(grams) for grams in n_grams]

def process_batch(pool, batch, n_value, n_grams):
    tasks = []
    for document in batch:
        task = (extract_ngrams, (document, n_value))
        tasks.append(task)

    on_done = lambda _ : None
    on_error = lambda _ : None
    documents = pool.map(None, tasks, on_error, on_done)

    for document_ngrams in documents:
        for n_gram in document_ngrams:
            if n_gram in n_grams:
                n_grams[n_gram] += 1
            else:
                n_grams[n_gram] = 0


gigabyte = 1000 * 1000 * 1000

def trim_ngram_dict(n_grams):
    logger.info("Trimming dict.")
    trimmed_dict = {}
    for i, (n_gram, count) in sorted(n_grams.items(), key = lambda ele: ele[1], reverse = True):
        if i == 100000:
            break
        trimmed_dict[n_gram] = count

    return trimmed_dict

def dump_ngram_dict(working_directory, n_grams, dump_batch_number):
    ngrams_pickle_file = os.path.join(working_directory, f"ngrams_{dump_batch_number}.pkl")
    pickle.dump(n_grams, open(ngrams_pickle_file, "wb"))


def main(working_directory, process_count, n_value, approx_ram_gb, dataset):
    nltk.download('punkt')

    maximum_memory = approx_ram_gb * gigabyte
    total_size = dataset.size()
    total_ngrams_size_worst = total_size * n_value
    memory_usage = total_ngrams_size_worst * 4 * 2 # x4 for dict x2 for sorted
    split_count = math.ceil(memory_usage / maximum_memory)
    documents_per_batch = dataset.num_docs() / split_count

    logger.info(f"Allocated RAM: {maximum_memory:,} bytes")
    logger.info(f"Total CC Size: {total_size:,} bytes")
    logger.info(f"Worst Case Ngrams Size: {total_ngrams_size_worst:,} bytes")
    logger.info(f"Approxmiate Max Memory Usage: {memory_usage}")
    logger.info(f"Split Count: {split_count}")
    logger.info(f"Documents per batch: {documents_per_batch}")

    batch_size = 1000
    batch = []
    pool = TqdmMultiProcessPool(process_count)
    count = 0
    n_grams = {}
    dump_batch_number = 0
    with tqdm.tqdm(total=dataset.num_docs(), dynamic_ncols=True, unit="docs") as progress:
        for document, meta in dataset.documents():

            batch.append(document)

            if len(batch) == batch_size:
                process_batch(pool, batch, n_value, n_grams)
                batch = []
                progress.update(batch_size)
                count += batch_size

                if count >= documents_per_batch:
                    n_grams = trim_ngram_dict(n_grams)
                    # dump_ngram_dict(working_directory, n_grams, dump_batch_number)
                    # n_grams = {}
                    count = 0
                    dump_batch_number += 1


        if len(batch) != 0:
            process_batch(pool, batch, n_value, n_grams)
            n_grams = trim_ngram_dict(n_grams)
            # dump_ngram_dict(working_directory, n_grams, dump_batch_number)
            progress.update(len(batch))

    pickle_file = os.path.join(working_directory, "ngrams.pkl")
    pickle.dump(n_grams, open(pickle_file, "wb"))

parser = argparse.ArgumentParser(description='n-gram statistics')
parser.add_argument("-dir", "--working_directory", default="")
parser.add_argument("-dataset", "--dataset", default="cc")
parser.add_argument("-procs", "--process_count", type=int, default=4)
parser.add_argument("-n", "--n_value", type=int, default=13)
parser.add_argument("-ram", "--approx_ram_gb", type=int, default=80)

if __name__ == '__main__':
    logfile_path = "ngrams.log"
    setup_logger_tqdm(logfile_path)

    args = parser.parse_args()

    if args.dataset == "cc":
        logger.info("Dataset: Common Crawl")
        dataset = CommonCrawlDataset()
    elif args.dataset == "owt2":
        dataset = OpenWebText2Dataset()
    else:
        logger.info("Invalid dataset")
        sys.exit(-1)

    main(args.working_directory, args.process_count, args.n_value, args.approx_ram_gb, dataset)