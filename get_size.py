from the_pile.datasets import WikipediaDataset, GutenbergDataset, PubMedCentralDataset, EnronEmailsDataset

def main():
    # wiki = WikipediaDataset()
    # print("Wiki Size: ", wiki.size())

    # gutenberg = GutenbergDataset()
    # print("Gutenberg Size: ", gutenberg.size())

    # pubmed = PubMedCentralDataset()
    # pubmed._download()

    enron = EnronEmailsDataset()
    enron._download()

if __name__ == '__main__':
    main()
