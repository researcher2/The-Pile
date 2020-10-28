from the_pile.datasets import WikipediaDataset, GutenbergDataset

def main():
    wiki = WikipediaDataset()
    print("Wiki Size: ", wiki.size())

    gutenberg = GutenbergDataset()
    print("Gutenberg Size: ", gutenberg.size())

if __name__ == '__main__':
    main()
