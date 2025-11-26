import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

#Section: Constants
RANDOM_STATE = 42 #seed used for all random operations (splits, CV) to ensure reproducible results
TEST_SIZE = 0.10 #fraction of the full dataset used as test data (10%)
DEV_FRAC_OF_REMAINDER = 0.1111 #fraction of the remaining data (after removing test set) used as dev set

#Section: Regular expression patterns used in preprocessing
email_regex = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b") #pattern to detect email addresses
url_regex = re.compile(r"http\S+|www\.\S+") #pattern to detect URLs starting with http, https or www
HEADER_TOKENS = [ #header tokens that should be removed from the raw email text
    "from",
    "organization",
    "organisation",
    "distribution",
    "lines",
    "sender",
    "reply-to",
    "nntp-posting-host",
    "keywords",
    "references",
    "subject"
]

header_pattern = re.compile( #regex that matches any of the header tokens followed by a colon ("From:", "Subject:" etc.)
    r"\b(" + "|".join(map(re.escape, HEADER_TOKENS)) + r")\s*:",
    flags=re.IGNORECASE
)

#Section: Functions
def format_email_text(text: str) -> str:
    """
    Clean a single raw email string.

    Operations:
    1) Ensure the input is a string; if not, return an empty string.
    2) Lowercase the text.
    3) Remove common header fields such as "From:", "Subject:" using header_pattern.
    4) Remove email addresses using email_regex.
    5) Remove URLs using url_regex.
    6) Replace non-word, non-whitespace and non-apostrophe characters with spaces.
    7) Collapse multiple spaces into one and strip leading/trailing spaces.

    Returns:
        Cleaned text as a single whitespace-normalised string.
    """
    if not isinstance(text, str): 
        return ""

    text = text.lower() #normalise text to lowercase
    text = header_pattern.sub(" ", text) #remove header tokens
    text = email_regex.sub(" ", text) #remove email addresses
    text = url_regex.sub(" ", text) #remove URLs
    text = re.sub(r"[^\w\s']", " ", text) #remove non-word, non-whitespace, non-apostrophe chars
    text = re.sub(r"\s+", " ", text).strip() #collapse multiple spaces and trim

    return text

def load_and_preprocess(path: str) -> pd.DataFrame:
    """
    Load the CSV file from the given path and apply preprocessing.

    Steps:
    1) Read the CSV into a DataFrame.
    2) Keep only the 'text' and 'label' columns and drop rows with missing values.
    3) Create a 'clean_text' column by applying format_email_text to each row.
    4) Create a 'doc_length' column with the token count (word count) of each clean text.

    Returns:
        A pandas DataFrame with columns: ['text', 'label', 'clean_text', 'doc_length'].
    """
    df = pd.read_csv(path) #load csv into DataFrame

    df = df[["text", "label"]].dropna(subset=["text", "label"]) #keep only relevant columns and drop NAs
    df["clean_text"] = df["text"].astype(str).apply(format_email_text) #preprocess text
    df["doc_length"] = df["clean_text"].apply(lambda t: len(t.split())) #compute document lengths

    return df

def split_raw_data(df: pd.DataFrame):
    """
    Split the preprocessed DataFrame into train, dev and test sets.

    Splitting strategy:
    - First, split off a test set of size TEST_SIZE (stratified).
    - Then, split the remaining data into traini and dev sets,
      with DEV_FRAC_OF_REMAINDER chosen so dev is roughly 10% overall.

    Returns:
        X_train, X_dev, X_test, y_train, y_dev, y_test
        where X_* are cleaned texts and y_* are labels.
    """
    X = df["clean_text"].values
    y = df["label"].values

    #first split: train and dev vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split( 
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    #second split: train vs dev
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train_val,
        y_train_val,
        test_size=DEV_FRAC_OF_REMAINDER,
        stratify=y_train_val,
        random_state=RANDOM_STATE
    )

    return X_train, X_dev, X_test, y_train, y_dev, y_test

def fetch_text_length(texts):
    """
    Given an array-like of text strings, return their document lengths.

    This is used inside a FunctionTransformer to provide doc_length as a numeric
    feature in the scikit-learn pipeline.

    Args:
        texts: Iterable of strings.

    Returns:
        A NumPy array of shape (n_samples, 1), where each value is the word count.
    """
    lengths = [len(t.split()) for t in texts]
    return np.array(lengths).reshape(-1, 1)

#Section: Model building and evaluation functions
def build_feature_choices(use_unigrams=True, use_bigrams=False, use_doclen=False) -> Pipeline:
    """
    Build a scikit-learn Pipeline for a given combination of features.

    Feature options:
      - Unigram TF-IDF
      - Bigram TF-IDF
      - Document length (word count) via fetch_text_length

    Args:
        use_unigrams: whether to include unigram TF-IDF features
        use_bigrams: whether to include bigram TF-IDF features
        use_doclen: whether to include document-length as a numeric feature

    Returns:
        A scikit-learn Pipeline with:
            - a FeatureUnion (or single transformer) as 'features'
            - a LinearSVC classifier as 'clf'
    """
    feature_blocks = []

     #choose the appropriate tfidf configuration based on the flags
    if use_unigrams and not use_bigrams:
        feature_blocks.append(
            ("unigrams", TfidfVectorizer(ngram_range=(1, 1), min_df=2))
        )
    elif use_unigrams and use_bigrams:
        feature_blocks.append(
            ("uni_bi", TfidfVectorizer(ngram_range=(1, 2), min_df=2))
        )
    elif use_bigrams and not use_unigrams:
        feature_blocks.append(
            ("bigrams", TfidfVectorizer(ngram_range=(2, 2), min_df=2))
        )

    #optionally add the document length feature
    if use_doclen:
        feature_blocks.append(
            ("doc_length", FunctionTransformer(fetch_text_length, validate=False))
        )

    #if there is only one feature block, use it directly. otherwise, merge them
    if len(feature_blocks) == 1:
        features = feature_blocks[0][1]
    else:
        features = FeatureUnion(feature_blocks)

    #final pipeline! features to LinearSVC classifier
    model = Pipeline([
        ("features", features),
        ("clf", LinearSVC(C=0.5, max_iter=10000, tol=1e-2))
    ])

    return model

def evaluate_feature_model(name, model, X_train, y_train, X_dev, y_dev):
    """
    Train a given model on train and evaluate it on dev.

    Args:
        name: Descriptive name of the feature configuration.
        model: A scikit-learn Pipeline returned by build_feature_choices().
        X_train, y_train: Training texts and labels.
        X_dev, y_dev: Development texts and labels.

    Prints:
        Dev accuracy and dev macro F1.

    Returns:
        A dictionary with model name, the fitted model, accuracy and macro F1.
    """
    print(f"\nFeature Choice: {name}")
    model.fit(X_train, y_train)
    y_dev_pred = model.predict(X_dev)

    acc = accuracy_score(y_dev, y_dev_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_dev, y_dev_pred, average="macro", zero_division=0
    )

    print(f"Dev Accuracy: {acc:.4f}")
    print(f"Dev Macro F1: {macro_f1:.4f}")

    return {
        "name": name,
        "model": model,
        "accuracy": acc,
        "macro_f1": macro_f1,
    }

def run_kfold_cv(name, model, X_train, y_train, k_folds=5, random_state=42):
    """
    Run k-fold cross-validation on the train set only using accuracy as the metric.

    Args:
        name: Descriptive name of the configuration (for logging).
        model: A scikit-learn Pipeline.
        X_train, y_train: Training texts and labels.
        k_folds: Number of folds for KFold cross-validation.
        random_state: Seed used to shuffle the data in KFold.

    Prints:
        Accuracy for each fold and the mean accuracy across folds.

    Returns:
        A NumPy array with the accuracy scores for each fold.
    """
    print(f"\nFeature {name}: {k_folds}-fold CV on train ---")
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=kf,
        scoring="accuracy",
        n_jobs=-1
    )

    print("Fold accuracies:", scores)
    print("Mean accuracy: {:.4f}".format(scores.mean()))

    return scores
