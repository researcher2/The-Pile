import os
import argparse
import pickle
import math
import sys
import csv
from pathlib import Path
import sys
import jsonlines

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


gigabyte = 1000 * 1000 * 1000


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
        bucket_file_path = os.path.join(working_directory, f"ngrams_{dataset_name}_{i}.bkt.jsonl")
        bucket_files[i] = jsonlines.open(bucket_file_path, mode='a')

    for result in results:
        for (ngram, ngram_hash) in result:
            bucket = ngram_hash % num_buckets
            bucket_files[bucket].write(ngram)

    for bucket_file in bucket_files:
        bucket_file.close()

def do_ngrams_in_buckets(working_directory, process_count, n_value, dataset, split_count):
    logger.info("Generating ngrams and bucketing for later")

    dataset_name = dataset.name().lower()

    done_file = os.path.join(working_directory, f"ngram_buckets_{dataset_name}.done")
    if os.path.exists(done_file):
        logger.info("ngrams already generated and bucketed, skipping")
        return

    lock_file = os.path.join(working_directory, f"ngram_buckets_{dataset_name}.lock")
    if os.path.exists(lock_file):
        logger.info("Looks like you stopped and need to start again, clear the data directory first...")
        sys.exit(0)

    Path(lock_file).touch()    

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

    os.remove(lock_file)
    Path(done_file).touch()

# Multiprocessed
def count_ngrams_bucket(bucket_file_path, tqdm_func, global_tqdm):
    bucket_pickle_file = bucket_file_path.replace(".bkt.jsonl", ".pkl")
    if os.path.exists(bucket_pickle_file):
        logger.info("Bucket pickle file already exists, skipping.")
        global_tqdm.update()
        return

    ngrams = {}
    with jsonlines.open(bucket_file_path) as reader:
        for ngram in reader:
            if ngram in ngrams:
                ngrams[ngram] += 1
            else:
                ngrams[ngram] = 1

    ngrams_sorted = list(sorted(ngrams.items(), key = lambda ele: ele[1], reverse = True))
    pickle.dump(ngrams_sorted, open(bucket_pickle_file, "wb"))

    global_tqdm.update()

def count_ngrams_in_buckets(working_directory, process_count, dataset_name):
    count = 0
    bucket_file_paths = []
    while True:
        bucket_file_path = os.path.join(working_directory, f"ngrams_{dataset_name}_{count}.bkt.jsonl")
        if not os.path.exists(bucket_file_path):
            break

        bucket_file_paths.append(bucket_file_path)
        count += 1

    pool = TqdmMultiProcessPool(1) # Oops this is memory limited
    tasks = []
    for bucket_file_path in bucket_file_paths:
        task = (count_ngrams_bucket, (bucket_file_path,))
        tasks.append(task)

    with tqdm(total=len(tasks), dynamic_ncols=True, unit="buckets") as progress:
        on_done = lambda _ : None
        on_error = lambda _ : None
        pool.map(progress, tasks, on_error, on_done)


def dump_ngram_csv(working_directory, ngrams, dataset_name, limit):
    csv_path = os.path.join(working_directory, f"ngrams_{dataset_name}_limit{limit}.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['ngram', 'count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for ngram, count in ngrams:
            writer.writerow({'ngram': ngram, 'count': count})

def get_top_ngrams(working_directory, dataset_name, limit):

    logger.info("Getting top {limit} ngrams.")

    overall_pickle_file = os.path.join(working_directory, f"ngrams_{dataset_name}_limit{limit}.pkl")
    if os.path.exists(overall_pickle_file):
        logger.info("Overall pickle file already exists, skipping")

    count = 0
    bucket_pickle_paths = []
    while True:
        bucket_pickle_file = os.path.join(working_directory, f"ngrams_{dataset_name}_{count}.pkl")
        if not os.path.exists(bucket_pickle_file):
            break

        bucket_pickle_paths.append(bucket_pickle_file)
        count += 1

    ngrams_limited = {}
    for bucket_pickle_file in tqdm(bucket_pickle_paths):

        bucket_ngrams = pickle.load(open(bucket_pickle_file, "rb")) # Presorted above
        for i, (ngram, count) in enumerate(bucket_ngrams[0:limit]):
            if i == limit:
                break
            ngrams_limited[ngram] = count

    overall_ngrams_sorted = {}    
    for i, (ngram, count) in enumerate(sorted(ngrams_limited.items(), key = lambda ele: ele[1], reverse = True)):
        if i == limit:
            break

        overall_ngrams_sorted[ngram] = count

    pickle.dump(overall_ngrams_sorted, open(overall_pickle_file, "wb"))

    logger.info("Saving to CSV.")
    dump_ngram_csv(working_directory, overall_ngrams_sorted, dataset_name, limit)

def main(working_directory, process_count, n_value, allocated_ram, dataset, top_limit):
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

    do_ngrams_in_buckets(working_directory, process_count, n_value, dataset, split_count)
    count_ngrams_in_buckets(working_directory, process_count, dataset_name) 
    get_top_ngrams(working_directory, dataset_name, top_limit)

parser = argparse.ArgumentParser(description='n-gram statistics')
parser.add_argument("-dir", "--working_directory", default="")
parser.add_argument("-dataset", "--dataset_name", default="CommonCrawlDataset")
parser.add_argument("-procs", "--process_count", type=int, default=4)
parser.add_argument("-n", "--n_value", type=int, default=13)
parser.add_argument("-ram", "--allocated_ram", type=int, default=80)
parser.add_argument("-limit", "--top_limit", type=int, default=1000)

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
            main(args.working_directory, args.process_count, args.n_value, args.allocated_ram, dataset, args.top_limit)
            # dataset.clean()
    else:
        if args.dataset_name not in dataset_lookup:
            logger.info("Dataset not found in datsets, valid datasets:")

        dataset = dataset_lookup[args.dataset_name]
        main(args.working_directory, args.process_count, args.n_value, args.allocated_ram, dataset, args.top_limit)
        # dataset.clean()

