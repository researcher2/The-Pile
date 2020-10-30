import os

class ExitCodeError(Exception): pass

def sh(x):
    if os.system(x): raise ExitCodeError()

def main(self):
    download_directory = "components/gutenberg/pg19_train"
    done_file = os.path.join(download_directory, "download.done")
    if not os.path.exists(done_file):
        os.makedirs(download_directory, exist_ok=True)
        sh(f"gsutil -m rsync gs://deepmind-gutenberg/train {download_directory}")

        with open(done_file, "w") as fh:
            fh.write("done!")

if __name__ == '__main__':
    main()