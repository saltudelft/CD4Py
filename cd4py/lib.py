# License: GNU General Public License v3.0

from cd4py import log_step
from cd4py.pytokenizer import tokenize_all_project_folders
from typing import List, Tuple
from dpu_utils.utils.dataloading import load_jsonl_gz
from dpu_utils.codeutils import get_language_keywords, split_identifier_into_parts
from tqdm import tqdm
import os
import re
import pandas as pd

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


def deduplicate_py_data(py_projects_path: str , tokenized_files_path: str):
    """
    Identifies near or exact duplicate files in a Python corpus
    """

    log_step("Tokenzing Python source code files")
    #tokenize_all_project_folders(py_projects_path, tokenized_files_path)

    log_step("Loading all the tokenized Python source code files")
    all_tokenized_py_files = get_tokenized_py_files(tokenized_files_path)

    log_step("Preprocessing tokenized files")
    df_tokenized_files = pd.DataFrame(all_tokenized_py_files, columns=['filename', 'tokens'])
    df_tokenized_files = df_tokenized_files[df_tokenized_files['tokens'].map(lambda x: True if len(x) > \
                                                                         NO_IDENTIFIER_TOKENS else False)].copy()
    print(f"Number of source code files: {df_tokenized_files.shape[0]:,}")
    print(f"Total number of tokens: {sum(df_tokenized_files['tokens'].apply(lambda x: len(x))):,}")

    all_tokens = df_tokenized_files['tokens'].tolist()
    # Select only identifiers and remove language keywords
    all_tokens = [[t for t in f_tks if IDENTIFIER_REGEX.match(t) and t not in \
                   get_language_keywords('python')] for f_tks in all_tokens]
    all_tokens = [[i for t in src for i in split_identifier_into_parts(t)] for src in all_tokens]
