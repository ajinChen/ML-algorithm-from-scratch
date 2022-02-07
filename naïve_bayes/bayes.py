import sys
import re
import string
import os
import numpy as np
import codecs
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from typing import Sequence
import time

# From scikit learn that got words from:
# http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words
ENGLISH_STOP_WORDS = frozenset([
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves"])


class defaultintdict(dict):
    """
    Behaves exactly like defaultdict(int) except d['foo'] does NOT
    add 'foo' to dictionary d. (Booo for that default behavior in
    defaultdict!)
    """
    def __init__(self):
        self._factory = int
        super().__init__()

    def __missing__(self, key):
        return 0


def filelist(root) -> Sequence[str]:
    """Return a fully-qualified list of filenames under root directory; sort names alphabetically."""
    allfiles = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            allfiles.append(os.path.join(path, name))
    return sorted(allfiles)


def get_text(filename:str) -> str:
    """
    Load and return the text of a text file, assuming latin-1 encoding as that
    is what the BBC corpus uses.
    """
    f = codecs.open(filename, encoding='latin-1', mode='r')
    s = f.read()
    f.close()
    return s


def words(text:str) -> Sequence[str]:
    """
    Given a string, return a list of words normalized as follows.
    Split the string to make words first by using regex compile() function
    and string.punctuation + '0-9\\r\\t\\n]' to replace all those
    char with a space character.
    Split on space to get word list.
    Ignore words < 3 char long and owercase all words, and remove English stop words
    """
    ctrl_chars = '\x00-\x1f'
    regex = re.compile(r'[' + ctrl_chars + string.punctuation + '0-9\r\t\n]')
    nopunct = regex.sub(" ", text)  # delete stuff but leave at least a space to avoid clumping together
    words = nopunct.split(" ")
    words = [w for w in words if len(w) > 2]  # ignore a, an, to, at, be, ...
    words = [w.lower() for w in words]
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    return words


def load_docs(docs_dirname:str) -> Sequence[Sequence]:
    """
    Load all .txt files under docs_dirname and return a list of word lists, one per doc.
    Ignore empty and non ".txt" files.
    """
    docs = []
    for file in filelist(docs_dirname):
        if file.endswith('.txt'):
            docs.append(words(get_text(file)))
    return docs


def vocab(neg:Sequence[Sequence], pos:Sequence[Sequence]) -> dict:
    """
    Given neg and pos lists of word lists, construct a mapping from word to word index.
    Use index 0 to mean unknown word, '__unknown__'. The real words start from index one.
    The words should be sorted so the first vocabulary word is index one.
    The length of the dictionary is |uniquewords|+1 because of "unknown word".
    |V| is the length of the vocabulary including the unknown word slot.

    Sort the unique words in the vocab alphabetically so we standardize which
    word is associated with which word vector index.

    E.g., given neg = [['hi']] and pos=[['mom']], return:
    V = {'__unknown__':0, 'hi':1, 'mom:2}
    and so |V| is 3
    """
    V = defaultintdict()
    V['__unknown__'] = 0
    count = 0
    vocabs = []
    for wordlist in neg:
        vocabs += wordlist
    for wordlist in pos:
        vocabs += wordlist
    uni_words = sorted(list(set(vocabs)))
    for word in uni_words:
        count += 1
        V[word] = count
    return V


def vectorize(V:dict, docwords:Sequence) -> np.ndarray:
    """
    Return a row vector (based upon V) for docwords. The first element of the
    returned vector is the count of unknown words. So |V| is |uniquewords|+1.
    """
    n = len(V)
    vec = np.zeros(n)
    for word in docwords:
        idx = V[word]
        vec[idx] += 1
    return vec


def vectorize_docs(docs:Sequence, V:dict) -> np.ndarray:
    """
    Return a matrix where each row represents a documents word vector.
    Each column represents a single word feature. There are |V|+1
    columns because we leave an extra one for the unknown word in position 0.
    Invoke vector(V,docwords) to vectorize each doc for each row of matrix
    :param docs: list of word lists, one per doc
    :param V: Mapping from word to index; e.g., first word -> index 1
    :return: numpy 2D matrix with word counts per doc: ndocs x nwords
    """
    D = []
    for doc in docs:
        D.append(vectorize(V, doc))
    return np.array(D)


class NaiveBayes621:
    """
    This object behaves like a sklearn model with fit(X,y) and predict(X) functions.
    Limited to two classes, 0 and 1 in the y target.
    """
    def __init__(self, pwc1=None, pwc0=None, p_c=None):
        self.pwc1 = pwc1
        self.pwc0 = pwc0
        self.p_c = p_c

    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        """
        Given 2D word vector matrix X, one row per document, and 1D binary vector y
        train a Naive Bayes classifier assuming a multinomial distribution for
        p(w,c), the probability of word exists in class c. p(w,c) is estimated by
        the number of times w occurs in all documents of class c divided by the
        total words in class c. p(c) is estimated by the number of documents
        in c divided by the total number of documents.

        The first column of X is a column of zeros to represent missing vocab words.
        """
        X[:, 0] = 0
        # p(c)
        row_num = len(y)
        p_c0 = np.count_nonzero(y == 0) / row_num
        p_c1 = 1 - p_c0
        self.p_c = (p_c0, p_c1)

        # p(w,c)
        vocab_num = len(X[0]) - 1
        wc_wc1 = np.dot(y.reshape(1, -1), X) + 1
        wc_c1 = np.sum(wc_wc1) + vocab_num + 1
        p_wc1 = wc_wc1 / wc_c1
        self.pwc1 = p_wc1

        y_0 = (y - 1) * -1
        wc_wc0 = np.dot(y_0.reshape(1, -1), X) + 1
        wc_c0 = np.sum(wc_wc0) + vocab_num + 1
        p_wc0 = wc_wc0 / wc_c0
        self.pwc0 = p_wc0

    def predict(self, X:np.ndarray) -> np.ndarray:
        """
        Given 2D word vector matrix X, one row per document, return binary vector
        indicating class 0 or 1 for each row of X.
        """
        log_pwc1 = np.log(self.pwc1).reshape(-1, 1)
        pro_1 = np.dot(X, log_pwc1) + np.log(self.p_c[1])
        log_pwc0 = np.log(self.pwc0).reshape(-1, 1)
        pro_0 = np.dot(X, log_pwc0) + np.log(self.p_c[0])
        pred = []
        for i in range(len(pro_1)):
            if pro_1[i] >= pro_0[i]:
                pred.append(1)
            else:
                pred.append(0)
        return np.array(pred)


def kfold_CV(model, X:np.ndarray, y:np.ndarray, k=4) -> np.ndarray:
    """
    Run k-fold cross validation using model and 2D word vector matrix X and binary
    y class vector. Return a 1D numpy vector of length k with the accuracies, the
    ratios of correctly-identified documents to the total number of documents. You
    can use KFold from sklearn to get the splits but must loop through the splits
    with a loop to implement the cross-fold testing.  Pass random_state=999 to KFold
    so we always get same sequence (wrong in practice) so student eval unit tests
    are consistent. Shuffle the elements before walking the folds.
    """
    accuracies = []
    kf = KFold(n_splits=k, shuffle=True, random_state=999)
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        predict = model.predict(X_test)
        accuracies.append(np.sum(y_test == predict) / len(y_test))
    return np.array(accuracies)
