import sqlite3
import nltk
from bs4 import BeautifulSoup
from os import listdir
from os.path import join, isfile
from time import time


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
    doc_name = document_path.split("/")[-1]
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
                                                              doc_name,
                                                              curr_count,
                                                              curr_offsets))

    conn.commit()


def build_index(conn):
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


if __name__ == "__main__":
    # Uncomment the following line if you haven't already installed the punkt nltk addon
    # nltk.download('punkt')

    conn = sqlite3.connect("../data/pa3.db")
    c = conn.cursor()
    cleanup_tables(conn)

    t1 = time()
    build_index(conn)
    t2 = time()
    print("Time taken: %.5f" % (t2 - t1))

    # print(process_document("data/evem.gov.si/evem.gov.si.55.html", conn))
    # c.execute("SELECT * FROM IndexWord")
    # print(c.fetchall())
    c.execute("SELECT * FROM Posting")
    # print(c.fetchall())
    conn.close()


