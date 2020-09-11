# License: GNU General Public License v3.0

from cd4py import log_step, dummy_preprocessor
from cd4py.pytokenizer import tokenize_all_project_folders
from typing import List, Tuple
from dpu_utils.utils.dataloading import load_jsonl_gz
from dpu_utils.codeutils import get_language_keywords, split_identifier_into_parts
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import os
import re
import pandas as pd
import numpy as np

# Files with fewer than the specified number of identifier tokens will be excluded
NO_IDENTIFIER_TOKENS = 20
IDENTIFIER_REGEX = re.compile('[_a-zA-Z][_a-zA-Z0-9]*')


def get_tokenized_py_files(tokenized_files_path: str) -> List[Tuple[str, List[str]]]:
    """
    Gathers all the tokenized Python source code files from a directory
    """

    tokenized_files = os.listdir(tokenized_files_path)
    all_tokenized_py_files: List[Tuple[str, List[str]]] = []

    for f in tqdm(tokenized_files, total=len(tokenized_files)):
        if f.endswith(".jsonl.gz"):
            for d in load_jsonl_gz(os.path.join(tokenized_files_path, f)):
                if len(d['tokens']) != 0:
                    all_tokenized_py_files.append((d['filename'], d['tokens']))
                # else:
                # print("Found a file without tokens: ", d['filename'])
    return all_tokenized_py_files


def preprocess_tokenized_files(tokenized_py_files: List[Tuple[str, List[str]]]) -> List[List[str]]:
    """
    Applies the following preprocess steps to tokenized source code files:
    1- Excludes source code files that have fewer identifiers tokens than the specified number
    2- Selects only identifier tokens
    3- Removes language keywords
    """

    df_tokenized_files = pd.DataFrame(tokenized_py_files, columns=['filename', 'tokens'])
    df_tokenized_files = df_tokenized_files[df_tokenized_files['tokens'].map(lambda x: True if len(x) > \
                                                                                               NO_IDENTIFIER_TOKENS else False)].copy()
    print(f"Number of source code files: {df_tokenized_files.shape[0]:,}")
    print(f"Total number of tokens: {sum(df_tokenized_files['tokens'].apply(lambda x: len(x))):,}")

    all_tokens = df_tokenized_files['tokens'].tolist()
    # Select only identifiers and remove language keywords
    all_tokens = [[t for t in f_tks if IDENTIFIER_REGEX.match(t) and t not in \
                   get_language_keywords('python')] for f_tks in tqdm(all_tokens, total=len(all_tokens))]
    all_tokens = [[i for t in src for i in split_identifier_into_parts(t)] for src in tqdm(all_tokens,
                                                                                           total=len(all_tokens))]
    return all_tokens


def vectorize_tokenized_files(preprocessed_file_tokens: List[List[str]], dim_tfidf_vec: int) -> np.array:
    """
    Vectorize all the preprocessed source code files using TF-IDF approach
    """

    tfidf = TfidfVectorizer(analyzer='word', tokenizer=dummy_preprocessor, preprocessor=dummy_preprocessor,
                            token_pattern=None, max_features=dim_tfidf_vec)
    tfidf.fit(preprocessed_file_tokens)
    all_vectorized_py_files = np.float32(np.vstack([tfidf.transform([d]).toarray() for d in \
                                                    tqdm(preprocessed_file_tokens, total=len(preprocessed_file_tokens))]))

    return all_vectorized_py_files


def deduplicate_py_data(py_projects_path: str, tokenized_files_path: str, dim_tfidf_vec: int):
    """
    Identifies near or exact duplicate files in a Python corpus
    """

    log_step("Tokenzing Python source code files")
    #tokenize_all_project_folders(py_projects_path, tokenized_files_path)

    log_step("Loading all the tokenized Python source code files")
    all_tokenized_py_files = get_tokenized_py_files(tokenized_files_path)

    log_step("Preprocessing tokenized files")
    all_preprocessed_tokens = preprocess_tokenized_files(all_tokenized_py_files)

    log_step("Vectorize pre-processed source code files using TF-IDF")
    vectorize_tokenized_files(all_preprocessed_tokens, dim_tfidf_vec)
