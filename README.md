# Intro
**CD4Py** is a code de-duplication tool for Python programming language. 
It detects **near** and **exact** duplicate source code files. To train a machine learning model on source code files, 
it is essential to identify and remove duplicate source code files from the dataset. Otherwise, code duplication 
significantly affects the practicality of machine learning-based tools, especially on unseen data.

- [Quick Installation](#quick-installation)
- [Usage](#usage)
  - [Examples](#examples)
- [Approach](#approach)

# Quick Installation
```
$ git clone https://github.com/saltudelft/CD4Py.git & cd CD4Py
$ pip install .
```

# Usage
```
$ cd4py --help
usage: cd4py [-h] --p P --od OD --ot OT [--d D] [--th TH] [--k K] [--tr TR]

Code De-Duplication for Python

optional arguments:
  -h, --help  show this help message and exit
  --p P       Path to Python projects
  --od OD     Output folder to store detected duplicate files.
  --ot OT     Output folder to store tokenized files.
  --d D       Dimension of TF-IDF vectors [default: 2048].
  --th TH     Threshold to identify duplicate files [default: 0.95].
  --k K       Number of nearest neighbor [default: 10].
  --tr TR     Number trees to build the index. More trees gives higher
              precision but slower [default: 20].
```

## Examples
- Run `CD4Py` to identify duplicate files for a Python dataset
```
$ cd4py --p $PYHON_DATASET --ot $TOKENS --od py_dataset_duplicates.jsonl.gz --d 1024
```
Replace `$PYHON_DATASET` with the path to the Python project folders and `$TOKENS` with the path to store 
tokenized project files. Also, note that detected duplicate files will be stored in 
the file `py_dataset_duplicates.jsonl.gz`.

- The following code example shows the removal of duplicate files using the example file `py_dataset_duplicates.jsonl.gz`:
```python
from dpu_utils.utils.dataloading import load_jsonl_gz
import random
# Selects randomly a file from each cluster of duplicate files
clusters_rand_files = [l.pop(random.randrange(len(l))) for l in load_jsonl_gz('py_dataset_duplicates.jsonl.gz')]
duplicate_files = [f for l in load_jsonl_gz('py_dataset_duplicates.jsonl.gz') for f in l]
duplicate_files = set(duplicate_files).difference(set(clusters_rand_files))
```

# Approach
The `CD4Py` code de-duplication tool uses the following procedure to identify duplicate files in a Python code corpus:

1. Tokenize all the source code files in the code corpus using `tokenize` module of Python standard library.
2. Preprocess tokenized source files by only selecting identifier tokens and removing language keywords.
3. Convert pre-processed tokenized files to a vector representation using the TF-IDF method.
4. Perform `k`-nearest neighbor search to find `k` candidate duplicate files for each source code file. 
Next, filter out candidate duplicate files by considering the threshold `t`.
5. Find clusters of duplicate source code files while assuming that similarity is transitive.
