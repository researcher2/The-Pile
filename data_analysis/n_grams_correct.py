import os
import argparse
import pickle
import math
import sys
import csv
from pathlib import Path
import sys

import nltk
from nltk.util import ngrams as get_ngrams
from nltk.probability import FreqDist
from tqdm_multiprocess import TqdmMultiProcessPool

from the_pile.datasets import *

import logging
from the_pile.logger import setup_logger_tqdm
logger = logging.getLogger(__name__)

# def broken_map():
#     # print(list(ngrams))
#     stuffs = list(ngrams)
#     print("bong")
#     print(stuffs[0])
#     print(type(hash(stuffs[0])))    
#     print(hash(stuffs[0]))

#     print("bing")
#     hashes = map(lambda x: hash(x), ngrams)
#     hashes_list = list(hashes)
#     print(hashes_list)
#     print("dong")
#     print(type(hashes_list[0]))
#     print(hashes_list[0])
#     # print(list(hashes))
#     print("nong")

# Multiprocessing
def extract_ngrams(data, num, tqdm_func, global_tqdm):
    ngram_lists = get_ngrams(nltk.word_tokenize(data), num)
    ngrams = list(map(lambda x: " ".join(x), ngram_lists))
    ngrams_with_hash = list(map(lambda x: (x, hash(x)), ngrams))

    return ngrams_with_hash

def process_batch(working_directory, dataset_name, pool, batch, n_value, num_buckets):
    tasks = []
    for document in batch:
        task = (extract_ngrams, (document, n_value))
        tasks.append(task)

    on_done = lambda _ : None
    on_error = lambda _ : None
    results = pool.map(None, tasks, on_error, on_done)

    bucket_files = [None] * num_buckets
    for i in range(num_buckets):
        bucket_file_path = os.path.join(working_directory, f"ngrams_{dataset_name}_{i}.bkt")
        bucket_files[i] = open(bucket_file_path, "a")

    for result in results:
        for (ngram, ngram_hash) in result:
            bucket = ngram_hash % num_buckets
            bucket_files[bucket].write(f"{ngram}\n")

    for bucket_file in bucket_files:
        bucket_file.close()


    # for document_ngrams in documents:
    #     for n_gram in document_ngrams:
    #         if n_gram in n_grams:
    #             n_grams[n_gram] += 1
    #         else:
    #             n_grams[n_gram] = 1


gigabyte = 1000 * 1000 * 1000

# def merge_ngrams(n_grams_master, n_grams):
#     logger.info("Merging top 100,000 ngrams in batch.")
#     for i, (n_gram, count) in enumerate(sorted(n_grams.items(), key = lambda ele: ele[1], reverse = True)):
#         if i == 100000:
#             break

#         if n_gram in n_grams_master:
#             n_grams_master[n_gram] += count
#         else:
#             n_grams_master[n_gram] = 1

# def dump_ngram_dict(working_directory, n_grams, dump_batch_number):
#     ngrams_pickle_file = os.path.join(working_directory, f"ngrams_{dump_batch_number}.pkl")
#     pickle.dump(n_grams, open(ngrams_pickle_file, "wb"))

# def dump_ngram_csv(working_directory, n_grams, dataset_name):
#     csv_path = os.path.join(working_directory, f"ngrams_{dataset_name}.csv")
#     with open(csv_path, 'w', newline='') as csvfile:
#         fieldnames = ['ngram', 'count']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#         writer.writeheader()
#         for i, (ngram, count) in enumerate(sorted(n_grams.items(), key = lambda ele: ele[1], reverse = True)):
#             if i == 1000:
#                 break
#             writer.writerow({'ngram': ngram, 'count': count})

def do_ngrams_in_buckets(working_directory, process_count, n_value, allocated_ram, dataset, split_count):
    logger.info("Generating ngrams and bucketing for later")

    done_file = os.path.join(working_directory, "ngram_buckets.done")
    if os.path.exists(done_file):
        logger.info("ngrams already generated and bucketed, skipping")
        return

    lock_file = os.path.join(working_directory, "ngram_buckets.lock")
    if os.path.exists(lock_file):
        logger.info("Looks like you stopped and need to start again, clear the data directory first...")
        sys.exit(0)

    Path(lock_file).touch()

    dataset_name = dataset.name().lower()

    pickle_file = os.path.join(working_directory, f"ngrams_{dataset_name}.pkl")
    if os.path.exists(pickle_file):
        logger.info("Dataset pickle file already found, skipping")
        return

    batch_size = 10000
    batch = []
    pool = TqdmMultiProcessPool(process_count)
    with tqdm(total=dataset.num_docs(), dynamic_ncols=True, unit="docs") as progress:
        for document, meta in dataset.documents():

            batch.append(document)

            if len(batch) == batch_size:
                process_batch(working_directory, dataset_name, pool, batch, n_value, split_count)
                batch = []
                progress.update(batch_size)

        if len(batch) != 0:
            process_batch(working_directory, dataset_name, pool, batch, n_value, split_count)
            progress.update(len(batch))

    os.path.remove(lock_file)
    Path(done_file).touch()

def main(working_directory, process_count, n_value, allocated_ram, dataset):
    nltk.download('punkt')

    dataset_name = dataset.name().lower()
    logger.info(f"Dataset: {dataset_name}")

    pickle_file = os.path.join(working_directory, f"ngrams_{dataset_name}.pkl")
    if os.path.exists(pickle_file):
        logger.info("Dataset pickle file already found, skipping")
        return

    # Basically doing map/reduce - ngrams split by hash into different buckets for later counting
    # Do some basic worst case calculations on memory usage to avoid blowing out memory
    maximum_memory = allocated_ram * gigabyte
    total_size = dataset.size()
    total_ngrams_size_worst = total_size * n_value
    memory_usage = total_ngrams_size_worst * 4 * 2 # x4 for dict x2 for sorted
    split_count = math.ceil(memory_usage / maximum_memory)

    logger.info(f"Allocated RAM: {maximum_memory:,} bytes")
    logger.info(f"Total Dataset Size: {total_size:,} bytes")
    logger.info(f"Worst Case Ngrams Size: {total_ngrams_size_worst:,} bytes")
    logger.info(f"Approxmiate Max Memory Usage: {memory_usage:,} bytes")
    logger.info(f"Split Count: {split_count}")

    do_ngrams_in_buckets(working_directory, process_count, n_value, allocated_ram, dataset, split_count)



    # pickle.dump(n_grams_master, open(pickle_file, "wb"))
    # dump_ngram_csv(working_directory, n_grams_master, dataset_name)

parser = argparse.ArgumentParser(description='n-gram statistics')
parser.add_argument("-dir", "--working_directory", default="")
parser.add_argument("-dataset", "--dataset_name", default="CommonCrawlDataset")
parser.add_argument("-procs", "--process_count", type=int, default=4)
parser.add_argument("-n", "--n_value", type=int, default=13)
parser.add_argument("-ram", "--allocated_ram", type=int, default=80)

datasets = [
    EnronEmailsDataset(),
    PubMedCentralDataset(),
    ArXivDataset(),
    FreeLawDataset(),
    USPTODataset(),
    PubMedDataset(),
    PhilPapersDataset(),
    ExPorterDataset(),
    CommonCrawlDataset(),
    OpenWebText2Dataset(),
    StackExchangeDataset(),
    WikipediaDataset(),
    BibliotikDataset(),
    GutenbergDataset(),
    LiteroticaDataset(),
    BookCorpusDataset(),
    GithubDataset(),
    UbuntuIRCDataset(),
    HackerNewsDataset(),
    EuroParlDataset(),
    YTSubtitlesDataset(),
    OpensubtitlesDataset(),
    DMMathDataset() 
]

dataset_lookup = {x.name() : x for x in datasets}

if __name__ == '__main__':
    logfile_path = "ngrams.log"
    setup_logger_tqdm(logfile_path)

    args = parser.parse_args()

    if args.dataset_name == "all":
        for dataset_name, dataset in dataset_lookup.items():
            main(args.working_directory, args.process_count, args.n_value, args.allocated_ram, dataset)
            # dataset.clean()
    else:
        if args.dataset_name not in dataset_lookup:
            logger.info("Dataset not found in datsets, valid datasets:")

        dataset = dataset_lookup[args.dataset_name]
        main(args.working_directory, args.process_count, args.n_value, args.allocated_ram, dataset)
        # dataset.clean()

