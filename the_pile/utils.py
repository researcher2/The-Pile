import re
import os
import hashlib
from concurrent_iterator.thread import Producer
from functools import reduce
import operator
import collections
import urllib.request
from pathlib import Path
import gdown
import tarfile
import requests
import shutil
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import pickle

def touch(x):
    Path(x).touch()

def get_url_content_length(url):
    response = requests.head(url)
    response.raise_for_status()

    if "Content-Length" in response.headers:
        return int(response.headers['Content-length'])
    else:
        return None

# Support 3 retries and backoff
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    method_whitelist=["HEAD", "GET", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)
session.mount("http://", adapter)

# def download_file(url, to):
#     # modified from https://stackoverflow.com/a/37573701
#     print('Downloading {}'.format(url))

#     response = session.get(url, stream=True)
#     size = int(response.headers.get('content-length', 0))
#     block_size = 1024*1024
#     pbar = tqdm(total=size, unit='iB', unit_scale=True)
#     with open(to, 'wb') as fout:
#         for data in response.iter_content(block_size):
#             pbar.update(len(data))
#             fout.write(data)
#     pbar.close()
#     assert not (size != 0 and pbar.n != size)

Source = collections.namedtuple('Source', ['type', 'url'])

h = hashlib.sha256()
b = bytearray(128*1024)
mv = memoryview(b)
progress = tqdm(total=os.path.getsize(filename), unit="byte", unit_scale=1)
tqdm.write(f"Verifying checksum for {filename}")
with open(filename, 'rb', buffering=0) as f:
    for n in iter(lambda : f.readinto(mv), 0):
        h.update(mv[:n])
        progress.update(n)
progress.close()

def download_file(url, to, checksum):
    print('Downloading {}'.format(url))
    expected_size = get_url_content_length(url)

    max_retries = 3
    fail_count = 0
    download_checkpoint = to + "ckpnt"
    while True:
        resume_point = 0
        temp_checksum = hashlib.sha256()
        if os.path.exists(to):
            if expected_size and os.path.getsize(to) != expected_size:
                # Will resume below
                fail_count += 1
                if os.path.exists(download_checkpoint):
                    resume_point, temp_checksum = pickle.load(open(download_checkpoint, "rb"))
                else:
                    resume_point = os.path.getsize(to)
                    temp_checksum = hashlib.sha256()
                    with open(to, "rb") as f:
                        for byte_block in iter(lambda: f.read(4096),b""):
                            temp_checksum.update(byte_block)
            else:
                # Full size (just missing .done file, edge case)
                print("Verifying sha256sum...")
                try:
                    sha256sum(to, expected=checksum)
                    return
                except:
                    fail_count += 1

        chunk_size = 1024*1024
        with tqdm(total=expected_size, unit="byte", unit_scale=1) as progress:
            try:
                # Support resuming
                if os.path.exists(to):
                    tqdm.write("File already exists, resuming download.")
                    headers = {}
                    headers["Range"] = f"bytes={resume_point}-"
                    progress.update(resume_point)
                else:
                    headers=None

                with session.get(url, headers=headers, stream=True) as r, \
                     open(to, 'ab') as f:
                    r.raise_for_status()
                    for chunk in r.iter_content(chunk_size):
                        f.write(chunk)

                        chunk_length = len(chunk)                        
                        resume_point += chunk_length
                        temp_checksum.update(chunk)                        
                        pickle.dump((resume_point, temp_checksum), open(download_checkpoint,"wb"))

                        progress.update(chunk_length)

            except Exception as ex:
                tqdm.write(f"Download error: {ex}")
                fail_count += 1
            
        if fail_count == max_retries:
            raise Exception("Download failed")

def download(fname, checksum, sources, extract=False):
    if os.path.exists(fname + '.done'): return
            
    print('Finding source for', fname)

    parentdir = Path(fname).parent
    os.makedirs(parentdir, exist_ok=True)

    for source in sources:
        try:
            # todo: implement torrent handling
            if source.type == 'direct':
                download_file(source.url, fname, checksum)
            elif source.type == 'gdrive':
                if os.path.exists(fname):
                    try:
                        print(fname, 'already exists.')
                        sha256sum(fname, expected=checksum)
                        touch(fname + '.done')
                        return
                    except AssertionError:
                        print('{} exists but doesn\'t match checksum!'.format(fname))
                        rm_if_exists(fname)

                gdown.download(source.url, fname, quiet=False)
                sha256sum(fname, expected=checksum)
            elif source.type == 'gcloud':
                raise NotImplementedError('gcloud download not implemented!')   

            if extract:
                tar_xf(fname)
                rm_if_exists(fname)
            touch(fname + '.done')
            return
        except KeyboardInterrupt:
            raise
        except:
            import traceback
            traceback.print_exc()
            print('Download method [{}] {} failed, trying next option'.format(source.type, source.url))
            # rm_if_exists(fname)
            continue

        break

    raise Exception('Failed to download {} from any source'.format(fname))


def tar_xf(x):
    parentdir = Path(x).parent
    tf = tarfile.open(x)
    tf.extractall(parentdir)

class ExitCodeError(Exception): pass


def stableorder(x):
    arr = [(elem, sha256str(elem.encode('utf-8'))) for elem in x]
    arr.sort(key=lambda x: x[1])
    return [elem for elem,_ in arr]

def id(x):
    return x

def utf8len(s):
    return len(s.encode('utf-8'))

def sh(x):
    if os.system(x): raise ExitCodeError()

def fwrite(fname, content):
    with open(fname, 'w') as fh:
        fh.write(content)

def fread(fname):
    with open(fname) as fh:
        return fh.read()

def ls(x):
    return [x + '/' + fn for fn in stableorder(os.listdir(x))]


def cycle_documents(dataset):
    while True:
        yield from filter(id, dataset.documents())

def concat(xs):
    for x in xs:
        yield from x


def flatMap(f, x):
    return reduce(operator.add, map(f, x), [])


def sha256str(s):
    h = hashlib.sha256()
    h.update(s)
    return h.hexdigest()

def sha256sum(filename, expected=None):
    h  = hashlib.sha256()
    b  = bytearray(128*1024)
    mv = memoryview(b)
    progress = tqdm(total=os.path.getsize(filename), unit="byte", unit_scale=1)
    tqdm.write(f"Verifying checksum for {filename}")
    with open(filename, 'rb', buffering=0) as f:
        for n in iter(lambda : f.readinto(mv), 0):
            h.update(mv[:n])
            progress.update(n)
    progress.close()
    
    if expected:
        assert h.hexdigest() == expected
        print('CHECKSUM OK', filename)
    else:
        print(filename, h.hexdigest())


def rm_if_exists(path):
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
    except NotADirectoryError:
        os.remove(path)


# https://stackoverflow.com/questions/12523586/python-format-size-application-converting-b-to-kb-mb-gb-tb/37423778
def humanbytes(B):
   'Return the given bytes as a human friendly KB, MB, GB, or TB string'
   B = float(B)
   KB = float(1024)
   MB = float(KB ** 2) # 1,048,576
   GB = float(KB ** 3) # 1,073,741,824
   TB = float(KB ** 4) # 1,099,511,627,776

   if B < KB:
      return '{0} {1}'.format(B,'Bytes' if 0 == B > 1 else 'Byte')
   elif KB <= B < MB:
      return '{0:.2f} KiB'.format(B/KB)
   elif MB <= B < GB:
      return '{0:.2f} MiB'.format(B/MB)
   elif GB <= B < TB:
      return '{0:.2f} GiB'.format(B/GB)
   elif TB <= B:
      return '{0:.2f} TiB'.format(B/TB)


def strip_markdown_colons(x):
    return re.sub(r'^:::.*?\n', '', x, flags=re.MULTILINE)

def remove_advertisement(x):
    return re.sub(r'^Advertisement\n', '', x, flags=re.MULTILINE)


def compose(*fs):
    def _f(x):
        for f in reversed(fs):
            x = f(x)
        return x

    return _f


def parse_size(sizestr):
    unit = sizestr[-1]
    size = float(sizestr[:-1])

    if unit.upper() == 'B':
        return size
    if unit.upper() == 'K':
        return size * 1024
    if unit.upper() == 'M':
        return size * 1024 * 1024
    if unit.upper() == 'G':
        return size * 1024 * 1024 * 1024
    if unit.upper() == 'T':
        return size * 1024 * 1024 * 1024 * 1024