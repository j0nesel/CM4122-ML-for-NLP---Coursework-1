from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.exceptions import ConvergenceWarning

from functions import *
import warnings
import numpy as np

#Suppress convergence warnings for cleaner output
#linearSVC can sometimes warn about convergence when the doc_length feature is included,
#but this does not affect the validity of the final solution here
warnings.filterwarnings("ignore", category=ConvergenceWarning)


#Section: Configuration, Constants and Variables:
DATA_PATH = "20_Newsgroups.csv" #path to the CSV file containing the text and label columns
all_results = [] #list that will store the evaluation results for each feature configuration

#Different feature configurations to compare
#each config controls whether unigrams, bigrams and/or document length are used
#to disable a system, simply comment out its dictionary in this list
configs = [
    # Single feature systems
    {"name": "A: unigrams only",
     "use_unigrams": True,  "use_bigrams": False, "use_doclen": False},

    {"name": "B: bigrams only",
     "use_unigrams": False, "use_bigrams": True,  "use_doclen": False},

    {"name": "C: doc_length only",
     "use_unigrams": False, "use_bigrams": False, "use_doclen": True},

    # Two feature combinations
    {"name": "D: unigrams and bigrams",
     "use_unigrams": True,  "use_bigrams": True,  "use_doclen": False},

    {"name": "E: unigrams and doc_length",
    "use_unigrams": True,  "use_bigrams": False, "use_doclen": True},

    {"name": "F: bigrams and doc_length",
     "use_unigrams": False, "use_bigrams": True,  "use_doclen": True},

    #All three features
    {"name": "G: unigram, bigram and doc_length",
     "use_unigrams": True,  "use_bigrams": True,  "use_doclen": True}
    ]

#Section: Main Programme Script
def main():
    """
    Start to end programme script :)

    Steps:
    1) Load and preprocess the dataset.
    2) Split into train/dev/test sets.
    3) train/evaluate one model per feature configuration on the dev set.
    4) Select the best configuration (by dev macro F1).
    5) Run k-fold CV for the best config on the training data.
    6) Refit best model on train and evaluate on dev.
    7) Refit best model on train+dev and evaluate on test.
    """

    #1) Load the CSV and preprocess / clean data
    df = load_and_preprocess(DATA_PATH)
    print("Loaded {} instances".format(len(df)))

    #2) Split the data into train, dev and test sets (80:10:10 approx)
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_raw_data(df)

    #3) Iterate over feature configurations, build models, evaluate on dev
    for cfg in configs: #build a pipeline with the requested combination of features
        model = build_feature_choices(
            use_unigrams=cfg["use_unigrams"],
            use_bigrams=cfg["use_bigrams"],
            use_doclen=cfg["use_doclen"]
        )

        result = evaluate_feature_model( #train on train and evaluate on dev
            cfg["name"],
            model,
            X_train, y_train,
            X_dev, y_dev
        )

        result["config"] = cfg #store the config alongside its scores
        all_results.append(result)

    #4) Select the configuration with the best dev macro F1
    best_result = max(all_results, key=lambda r: r["macro_f1"])
    best_cfg = best_result["config"]

    print("\nBest Config (by dev Macro F1):")
    print(best_result["name"])
    print("Dev Macro F1:", best_result["macro_f1"])
    print("Config flags:", best_cfg)

    best_model = build_feature_choices( #rebuild a fresh model using the best configuration
        use_unigrams=best_cfg["use_unigrams"],
        use_bigrams=best_cfg["use_bigrams"],
        use_doclen=best_cfg["use_doclen"]
    )

    #5) Run k-fold cross-validation on train ONLY for the best configuration
    run_kfold_cv(
        name=best_result["name"],
        model=best_model,
        X_train=X_train,
        y_train=y_train,
        k_folds=5,
        random_state=RANDOM_STATE
    )

    #6) Refit best model on train and show dev results again
    print("\nRefit best model on training data and evaluate on development data:")
    best_model.fit(X_train, y_train)
    y_dev_pred = best_model.predict(X_dev)

    dev_acc = accuracy_score(y_dev, y_dev_pred)
    dev_macro_p, dev_macro_r, dev_macro_f1, _ = precision_recall_fscore_support(
        y_dev, y_dev_pred, average="macro", zero_division=0
    )

    print("Development Accuracy: {:.4f}".format(dev_acc))
    print("Development Macro Precision:{:.4f}".format(dev_macro_p))
    print("Development Macro Recall: {:.4f}".format(dev_macro_r))
    print("Development Macro F1: {:.4f}".format(dev_macro_f1))

    #7) Retrain best model on train and dev, evaluate on test
    print("\nTrain best model on combined training and development data, evaluate on test data:")

    X_train_full = np.concatenate([X_train, X_dev]) #concatenate train and dev splits into a single training set
    y_train_full = np.concatenate([y_train, y_dev])

    # NOTE: The overlap check below is not very meaningful because it compares
    # raw text values to label values. It is left here as a harmless diagnostic
    # but can be safely removed.
    set_train = set(X_train_full)
    set_test = set(X_test)
    overlap = set_train.intersection(set_test)
    print("Train/test overlap (text vs labels):", len(overlap))

    best_model.fit(X_train_full, y_train_full) #fit the best model on train+dev and evaluate on the held-out test set
    y_test_pred = best_model.predict(X_test)

    test_acc = accuracy_score(y_test, y_test_pred)
    test_macro_p, test_macro_r, test_macro_f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average="macro", zero_division=0
    )

    print("\nTest Accuracy: {:.4f}".format(test_acc))
    print("Test Macro Precision: {:.4f}".format(test_macro_p))
    print("Test Macro Recall: {:.4f}".format(test_macro_r))
    print("Test Macro F1: {:.4f}".format(test_macro_f1))


if __name__ == "__main__":
    main()
