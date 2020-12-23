import pickle
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

import logging
from the_pile.logger import setup_logger_tqdm
logger = logging.getLogger(__name__)

def main(working_directory, dataset_name, limit):

    output_directory = os.path.join(working_directory, dataset_name)

    statistics_pickle_file = os.path.join(output_directory, f"statistics_{dataset_name}_limit{limit}.pkl")
    if not os.path.exists(statistics_pickle_file):
        logger.info(f"Statistics file not found: {statistics_pickle_file}")
        return

    overall_count, overall_unique_count, overall_ngrams_sorted, frequencies = tuple(pickle.load(open(statistics_pickle_file, "rb")))

    logger.info(f"Overall Count: {overall_count}")
    logger.info(f"Unique Count: {overall_unique_count}")

    # cc max
    cc_histogram_bucket_max = 11034000
    histogram_bucket_count = 1000
    histogram_bucket_size = cc_histogram_bucket_max / histogram_bucket_count

    # x = [21,22,23,4,5,6,77,8,9,10,31,32,33,34,35,36,37,18,49,50,100]
    x = list(range(0, cc_histogram_bucket_max, int(histogram_bucket_size)))
    # print(frequencies)
    n, bins, patches = plt.hist(x, log=True, weights=frequencies, bins=histogram_bucket_count, facecolor='blue', alpha=0.5)
    plt.title(f"Histogram of 13-gram count for {dataset_name}")
    plt.xlabel("13-gram count")
    plt.ylabel("Bin Frequency")
    plt.show()
    plt.savefig(os.path.join(output_directory,'frequency_hist.png'))

def main100bins(working_directory, dataset_name, limit):    
    output_directory = os.path.join(working_directory, dataset_name)

    statistics_pickle_file = os.path.join(output_directory, f"statistics_{dataset_name}_limit{limit}.pkl")
    if not os.path.exists(statistics_pickle_file):
        logger.info(f"Statistics file not found: {statistics_pickle_file}")
        return

    overall_count, overall_unique_count, overall_ngrams_sorted, frequencies = tuple(pickle.load(open(statistics_pickle_file, "rb")))

    logger.info(f"Overall Count: {overall_count}")
    logger.info(f"Unique Count: {overall_unique_count}")

    # cc max
    cc_histogram_bucket_max = 11034000
    histogram_bucket_count = 100
    histogram_bucket_size = cc_histogram_bucket_max / histogram_bucket_count
    
    frequencies_new = []
    count = 0
    current_bin_frequency = 0
    for bin_frequency in frequencies:
        current_bin_frequency += bin_frequency
        if count == 9:
            frequencies_new.append(current_bin_frequency)
            count = 0
            current_bin_frequency = 0
        else:
            count += 1

    if count != 0:
        frequencies_new.append(current_bin_frequency)

    logger.info(f"Bin Count: {len(frequencies_new)}")
    x = list(range(0, cc_histogram_bucket_max, int(histogram_bucket_size)))

    n, bins, patches = plt.hist(x, log=True, weights=frequencies_new, bins=histogram_bucket_count, facecolor='blue', alpha=0.5)
    plt.title(f"Histogram of 13-gram count for {dataset_name}")
    plt.xlabel("13-gram count")
    plt.ylabel("Bin 13-gram aggregate count")
    plt.show()
    plt.savefig(os.path.join(output_directory,'frequency_hist.png'))

def test_hist():
    # x = [21,22,23,4,5,6,77,8,9,10,31,32,33,34,35,36,37,18,49,50,100]

    cc_histogram_bucket_max = 11034000
    histogram_bucket_count = 1000
    histogram_bucket_size = cc_histogram_bucket_max / histogram_bucket_count

    x = list(range(0, cc_histogram_bucket_max, int(histogram_bucket_size)))
    frequencies = []
    count = 0
    for i in range(len(x)):
        frequencies.append(count)
        count += 5

    n, bins, patches = plt.hist(x, weights=frequencies, bins=histogram_bucket_count, facecolor='blue', alpha=0.5)
    plt.show()

parser = argparse.ArgumentParser(description='n-gram statistics')
parser.add_argument("-dir", "--working_directory", default="")
parser.add_argument("-dataset", "--dataset_name", default="CommonCrawl")
parser.add_argument("-limit", "--top_limit", type=int, default=1000)

if __name__ == '__main__':
    logfile_path = "process_statistics.log"
    setup_logger_tqdm(logfile_path)

    args = parser.parse_args()    

    # test_hist()
    main100bins(args.working_directory, args.dataset_name.lower(), args.top_limit)
