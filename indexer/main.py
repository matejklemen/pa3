import sqlite3
import nltk
from bs4 import BeautifulSoup
from os import listdir, sep
from os.path import join, isfile, exists
from time import time
import numpy as np


# TODO: there might be some more stopwords in the example in instructions (didn't check yet)
with open("../data/slovenian-stopwords.txt", "r") as f_stopwords:
    stopwords = set([line.strip() for line in f_stopwords])

with open("../data/vocabulary_ccgigafida.txt", "r") as f_stopwords:
    vocabulary = set([line.strip() for line in f_stopwords])


def preprocess_data(content):
    """ Turns content to lowercase letters, tokenizes it (Slovene-specific) and removes
    stopwords (Slovene-specific).

    Parameters
    ----------
    content: str
        Website content or query content

    Returns
    -------
    tokens_no_stopwords: list of str
        Tokens for the preprocessed website or query

    token_indices_no_stopwords: list of int
        Locations (offsets) for the resulting tokens in ORIGINAL (unprocessed) text
    """
    # Lowercase conversion, tokenization (Slovene), stopwords filtering (Slovene)
    normalized_text = content.lower()
    tokens = nltk.word_tokenize(normalized_text, language="slovene")
    mask = list(map(lambda word: word not in stopwords, tokens))

    # TODO: figure out these offsets
    # get offset for words in original text before removing stop words
    token_indices_no_stopwords = list(filter(lambda i: mask[i], range(len(tokens))))
    tokens_no_stopwords = [tokens[i] for i in token_indices_no_stopwords]

    return tokens_no_stopwords, token_indices_no_stopwords


def cleanup_tables(conn):
    cursor = conn.cursor()
    cursor.execute("DELETE FROM IndexWord")
    cursor.execute("DELETE FROM Posting")
    conn.commit()


def process_document(document_path, conn):
    c = conn.cursor()

    with open(document_path, "r", encoding="iso 8859-1") as f_page:
        soup = BeautifulSoup(f_page, "lxml")

    # Remove style and script tags - they mostly don't help us at this stage
    for s in soup(["script", "style"]):
        s.extract()

    preprocessed_text, offsets = preprocess_data(soup.text)

    # Get counts and offsets of words in preprocessed document
    doc_stats = {}
    for i, curr_word in enumerate(preprocessed_text):
        # No point in keeping track of words that we will not index
        if curr_word not in vocabulary:
            continue

        curr_word_stats = doc_stats.get(curr_word, None)
        if curr_word_stats is None:
            # Note: saving str(offset) because `join(...)` method expects list of strings
            curr_word_stats = {"count": 1, "offsets": [str(offsets[i])]}
        else:
            curr_word_stats["count"] += 1
            curr_word_stats["offsets"].append(str(offsets[i]))

        doc_stats[curr_word] = curr_word_stats

    # Prepare and insert data into db
    for curr_word, curr_word_stats in doc_stats.items():
        curr_count = curr_word_stats["count"]
        curr_offsets = ",".join(curr_word_stats["offsets"])

        c.execute("INSERT INTO Posting VALUES (?, ?, ?, ?)", (curr_word,
                                                              document_path,
                                                              curr_count,
                                                              curr_offsets))

    conn.commit()


def build_index(conn):
    # Check if index is already built
    if exists(join("..", "data", ".index_exists.tmp")):
        print("Inverted index exists, therefore not creating it again...")
        return

    cleanup_tables(conn)
    c = conn.cursor()
    # Create word index according to predefined vocabulary of tokens
    c.executemany("INSERT INTO IndexWord VALUES (?)", [(word,) for word in vocabulary])
    conn.commit()

    for curr_website in ["e-prostor.gov.si", "e-uprava.gov.si", "evem.gov.si", "podatki.gov.si"]:
        curr_website_dir = join("..", "data", curr_website)
        curr_website_files = [f for f in listdir(curr_website_dir)
                              if isfile(join(curr_website_dir, f))]

        for file in curr_website_files:
            doc_path = join(curr_website_dir, file)
            print("Current file: '{}'...".format(doc_path))
            process_document(doc_path, conn)

    # Create cache file to signal that index should not be rebuilt on next run of this program
    with open(join("..", "data", ".index_exists.tmp"), "w") as f:
        pass


def display_results(res):
    for file_path, stats in res:
        doc_name = file_path.split(sep)[-1]
        with open(file_path, "r") as curr_f:
            curr_soup = BeautifulSoup(curr_f, "lxml")

        for s in curr_soup(["script", "style"]):
            s.extract()

        # TODO: figure what and how to display summary text in search results (this aint it, chief)
        # TODO: (might need to also change the way offsets are calculated in preprocess function)
        curr_text = curr_soup.text
        curr_freq = stats["freq"]
        curr_offsets = stats["offsets"]

        print("Document: '{}', frequency: {}, text: ".format(doc_name, curr_freq), end="")

        curr_offsets_w_neigh = []
        for offset in curr_offsets:
            for k in range(-3, 3 + 1):
                curr_offsets_w_neigh.append(max(0, offset + k))

        # uniq_offsets = np.unique(curr_offsets_w_neigh)
        # print(curr_text[uniq_offsets[0]], end=" ")
        # for i in range(1, len(uniq_offsets)):
        #     if uniq_offsets[i] - uniq_offsets[i - 1] > 1:
        #         print("...", end=" ")
        #
        #     print(curr_text[uniq_offsets[i]], end=" ")
        print("TBD")


def search_index(query, conn):
    c = conn.cursor()
    norm_query, norm_query_offsets = preprocess_data(query)

    search_results = {}
    # Sum up frequencies of query word occurences in each of the documents they occur in
    for curr_word in norm_query:
        curr_offset, res_limit = 0, 10000
        # Assign some garbage to results so that it's non-empty and first
        # iteration of while-loop always executes
        res = [None]
        while len(res) > 0:
            c.execute("SELECT * FROM Posting WHERE word = ? LIMIT ? OFFSET ?",
                      (curr_word, res_limit, curr_offset))
            res = c.fetchall()
            curr_offset += res_limit

            for _, doc_name, curr_freq, curr_offsets in res:
                curr_doc_stats = search_results.get(doc_name, None)
                if curr_doc_stats is None:
                    curr_doc_stats = dict()
                    curr_doc_stats["freq"] = curr_freq
                    curr_doc_stats["offsets"] = list(map(int, curr_offsets.split(",")))
                else:
                    curr_doc_stats["freq"] += curr_freq
                    curr_doc_stats["offsets"].extend(map(int, curr_offsets.split(",")))
                search_results[doc_name] = curr_doc_stats

    # Sort descending by sum of query word frequencies
    sorted_res = sorted(search_results.items(),
                        key=lambda pair: pair[1]["freq"],
                        reverse=True)

    # TODO: fix this function
    display_results(sorted_res)


def search_naive(query):
    norm_query, norm_query_offsets = preprocess_data(query)

    search_results = {}
    for curr_website in ["e-prostor.gov.si", "e-uprava.gov.si", "evem.gov.si", "podatki.gov.si"]:
        curr_website_dir = join("..", "data", curr_website)
        curr_website_files = [f for f in listdir(curr_website_dir)
                              if isfile(join(curr_website_dir, f))]

        for file in curr_website_files:
            doc_path = join(curr_website_dir, file)

            with open(doc_path, "r", encoding="iso 8859-1") as f_page:
                soup = BeautifulSoup(f_page, "lxml")

            for s in soup(["script", "style"]):
                s.extract()

            preprocessed_text, offsets = preprocess_data(soup.text)
            curr_doc_stats = dict()
            # yikes, how many nested for loops do we need
            for i, token in enumerate(preprocessed_text):
                for q_word in norm_query:
                    if q_word == token:
                        curr_doc_stats["freq"] = curr_doc_stats.get("freq", 0) + 1
                        curr_offsets = curr_doc_stats.get("offsets", None)
                        if curr_offsets is None:
                            curr_offsets = [offsets[i]]
                        else:
                            curr_offsets.append(offsets[i])
                        curr_doc_stats["offsets"] = curr_offsets

            if curr_doc_stats:
                search_results[doc_path] = curr_doc_stats

    # Sort descending by sum of query word frequencies
    sorted_res = sorted(search_results.items(),
                        key=lambda pair: pair[1]["freq"],
                        reverse=True)

    # TODO: fix this function
    display_results(sorted_res)


if __name__ == "__main__":
    # Uncomment the following line if you haven't already installed the punkt nltk addon
    # nltk.download('punkt')

    conn = sqlite3.connect("../data/pa3.db")
    c = conn.cursor()

    t1 = time()
    build_index(conn)
    t2 = time()
    print("Time spent building index: {:.5f}s...".format(t2 - t1))

    # TODO: add more queries when testing and creating report (also, use more, like 5 or 10, reps
    # TODO: when measuring time for report
    for test_query in ["social services"]:
        t1 = time()
        search_index("social services", conn)
        t2 = time()
        print("-----------------")
        t3 = time()
        search_naive("social services")
        t4 = time()

        print("Query '{}' took {:.5f}s using inverted index and {:.5f}s "
              "using naive approach...".format(test_query, t2 - t1, t4 - t3))

    # c.execute("SELECT * FROM IndexWord")
    # print(c.fetchall())
    # c.execute("SELECT * FROM Posting")
    # print(c.fetchall())
    conn.close()


