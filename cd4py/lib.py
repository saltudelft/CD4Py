# License: GNU General Public License v3.0

from cd4py import log_step, dummy_preprocessor
from cd4py.pytokenizer import tokenize_all_project_folders
from typing import List, Tuple, Dict, Set
from dpu_utils.utils.dataloading import load_jsonl_gz, save_jsonl_gz
from dpu_utils.codeutils import get_language_keywords, split_identifier_into_parts
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import os
import re
import time
import pandas as pd
import numpy as np
import annoy

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


def preprocess_tokenized_files(tokenized_py_files: List[Tuple[str, List[str]]]) -> Tuple[pd.DataFrame,
                                                                                         List[List[str]]]:
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
    return df_tokenized_files, all_tokens


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


def build_knn_index(vectorized_files: np.array, vec_dim: int, knn_tree_size: int=20) -> annoy.AnnoyIndex:
    """
    Builds an index trees for finding nearest neighbors
    """

    annoy_idx = annoy.AnnoyIndex(vec_dim, 'dot')
    for i, v in enumerate(tqdm(vectorized_files, total=vectorized_files.shape[0])):
        annoy_idx.add_item(i, v)

    annoy_idx.build(knn_tree_size)
    return annoy_idx


def find_knn(vectorized_files, knn_index: annoy.AnnoyIndex, k: int) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Find k-nearest neighbors of vectorized source code files for identifying duplicate files
    """

    knn_idx, knn_dist = [], []
    for v in tqdm(vectorized_files, total=vectorized_files.shape[0]):
        idx, dist = knn_index.get_nns_by_vector(v, k, include_distances=True)
        knn_idx.append(idx)
        knn_dist.append(dist)

    return knn_idx, knn_dist


def find_duplicate_sets(df_tokenized_files: pd.DataFrame, t: int, k: int, files_knn_idx: List[List[int]],
                        files_knn_dist: List[List[float]]) -> Dict[str, List[Tuple[str, float]]]:
    """
    Finds duplicate sets of files for each source code file
    """

    clone_sets = {}
    for i in tqdm(range(df_tokenized_files.shape[0])):
        clone_sets[df_tokenized_files['filename'].iloc[i]] = []
        for j in range(1, k):
            # print(i,j)
            if files_knn_dist[i][j] > t:
                if df_tokenized_files['filename'].iloc[i] != df_tokenized_files['filename'].iloc[files_knn_idx[i][j]]:
                    clone_sets[df_tokenized_files['filename'].iloc[i]].append(
                        (df_tokenized_files['filename'].iloc[files_knn_idx[i][j]],
                         files_knn_dist[i][j]))
                    # print(f"Found a duplicate {df_tokenized_files['filename'].iloc[i]} -> ({df_tokenized_files['filename'].iloc[I[i][j]]}, {D[i][j]})")
            else:
                break

    return clone_sets


def find_transitive_duplicate_sets(duplicate_files_set: Dict[str, List[Tuple[str, float]]]) -> Tuple[List[Set[str]],
                                                                                                     List[str]]:
    """
    Finds clusters of near or exact duplicate files based on the assumption that similarity is transitive.
    """

    duplicate_files_closure = []
    files_clone_idx = {}
    # Compute the transitive closure of this relationship
    documents_to_visit = set(duplicate_files_set.keys())
    while len(documents_to_visit) > 0:
        current_idx = documents_to_visit.pop()
        current_idx_closure = {current_idx}
        visit_queue = []
        for f in duplicate_files_set[current_idx].copy():
            if f[0] in files_clone_idx:
                duplicate_files_closure[files_clone_idx[f[0]]].add(current_idx)
                files_clone_idx[current_idx] = files_clone_idx[f[0]]
                visit_queue = []
                break
            else:
                visit_queue.append(f[0])

        if len(visit_queue) != 0:
            while len(visit_queue) > 0:
                other_idx = visit_queue.pop()
                current_idx_closure.add(other_idx)
                documents_to_visit.discard(other_idx)

                # Add to queue
                visit_queue.extend(next_idx[0] for next_idx in duplicate_files_set[other_idx]
                                   if next_idx[0] in documents_to_visit)

            duplicate_files_closure.append(set(f for f in current_idx_closure))
            for f in current_idx_closure:
                files_clone_idx[f] = len(duplicate_files_closure) - 1

        else:
            continue

    return duplicate_files_closure, [f for c in duplicate_files_closure for f in c]


def report_duplicate_files_stats(no_src_files: int, no_duplicate_files: int,
                                 duplicate_files_set_closure: List[Set[str]]):

    print(f"Number of duplicated files: {no_duplicate_files:,} ({no_duplicate_files / no_src_files * 100.0:.2f}%)")
    print(f"Number of detected clusters: {len(duplicate_files_set_closure):,}")
    print("Avg. number of files per clones: %.2f" % np.mean([len(c) for c in duplicate_files_set_closure]))
    print("Median number of files per clones: %.2f" % np.median([len(c) for c in duplicate_files_set_closure]))
    print("Duplication ratio: %.2f%%" % (
                (no_duplicate_files - len(duplicate_files_set_closure)) / no_src_files * 100.0))


def deduplicate_py_data(py_projects_path: str, tokenized_files_path: str, detected_duplicate_f_path: str,
                        dim_tfidf_vec: int, t: int, no_knn: int, knn_tree_size: int):
    """
    Identifies near or exact duplicate files in a Python corpus.

    :param py_projects_path: Path to the Python project files
    :param tokenized_files_path: Path to store tokenized files
    :param detected_duplicate_f_path: Path to store detected duplicate files
    :param dim_tfidf_vec: Dimension of vectorized files
    :param t: Threshold to identify a file as duplicate
    :param no_knn: Number of nearest neighbors to find when performing KNN search
    :param knn_tree_size: Size of trees when building KNN index. Higher number gives more precision but slower.
    """

    start_t = time.time()

    log_step("Tokenizing Python source code files")
    tokenize_all_project_folders(py_projects_path, tokenized_files_path)

    log_step("Loading all the tokenized Python source code files")
    all_tokenized_py_files = get_tokenized_py_files(tokenized_files_path)

    log_step("Preprocessing tokenized files")
    df_tokenized_files, all_preprocessed_tokens = preprocess_tokenized_files(all_tokenized_py_files)

    log_step("Vectorize pre-processed source code files using TF-IDF")
    vectorized_files = vectorize_tokenized_files(all_preprocessed_tokens, dim_tfidf_vec)

    log_step("Building KNN index and finding nearest neighbors")
    knn_index = build_knn_index(vectorized_files, dim_tfidf_vec, knn_tree_size)
    files_knn_idx, files_knn_dist = find_knn(vectorized_files, knn_index, no_knn)

    log_step("Finding exact and near duplicate files")
    duplicate_files_set = find_duplicate_sets(df_tokenized_files, t, no_knn, files_knn_idx, files_knn_dist)
    duplicate_files_set_closure, duplicate_files = find_transitive_duplicate_sets(duplicate_files_set)
    # A sanity check to make sure that there is no intersection between the clusters of duplicate files
    assert len(duplicate_files) == sum(len(c) for c in duplicate_files_set_closure)

    log_step("Report duplication stats & saving detected duplicate files")
    report_duplicate_files_stats(df_tokenized_files.shape[0], len(duplicate_files), duplicate_files_set_closure)
    save_jsonl_gz([list(c) for c in duplicate_files_set_closure], detected_duplicate_f_path)

    print("Finished duplicate files detection in %.2f minutes." % ((time.time() - start_t) / 60.0))
