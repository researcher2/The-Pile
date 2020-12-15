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

    # cc max
    cc_histogram_bucket_max = 11034000
    histogram_bucket_count = 1000
    histogram_bucket_size = cc_histogram_bucket_max / 1000
    frequencies = [0] * histogram_bucket_count

    # x = [21,22,23,4,5,6,77,8,9,10,31,32,33,34,35,36,37,18,49,50,100]
    x = list(range(0, cc_histogram_bucket_max, int(histogram_bucket_size)))
    n, bins, patches = plt.hist(x, weights=frequencies, bins=histogram_bucket_count, facecolor='blue', alpha=0.5)
    # plt.show()
    plt.savefig('plot.png')

parser = argparse.ArgumentParser(description='n-gram statistics')
parser.add_argument("-dir", "--working_directory", default="")
parser.add_argument("-dataset", "--dataset_name", default="CommonCrawl")
parser.add_argument("-limit", "--top_limit", type=int, default=1000)

if __name__ == '__main__':
    logfile_path = "process_statistics.log"
    setup_logger_tqdm(logfile_path)

    args = parser.parse_args()    

    main(args.working_directory, args.dataset_name.lower(), args.top_limit)
