import sys
import pickle
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from util import load_data, impute_missing_values, encode_features

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


def train_and_save_model(df):
    train_len = df.shape[0]
    Y = df.CarInsurance[:train_len]
    df.drop(['CarInsurance'], axis=1, inplace=True)
    X = df[:train_len]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=144)
    classifier = LGBMClassifier(n_estimators=100, silent=False, random_state=94, max_depth=20, num_leaves=40,
                                objective='binary', metrics='auc')
    model = classifier.fit(X_train, Y_train, eval_metric='auc')
    test_model_predictions = model.predict(X_test)
    pd.crosstab(pd.Series(Y_test, name='Actual'), pd.Series(test_model_predictions, name='Predict'), margins=True)

    print("Training")
    print("Accuracy is {0:.2f}".format(accuracy_score(Y_test, test_model_predictions)))
    print("Precision is {0:.2f}".format(precision_score(Y_test, test_model_predictions)))
    print("Recall is {0:.2f}".format(recall_score(Y_test, test_model_predictions)))
    print("F1-Score is {0:.2f}".format(f1_score(Y_test, test_model_predictions)))

    # Saving model to disk
    pickle.dump(model, open('models/carInsurance_model.pkl', 'wb'))

    pickle.dump(X.columns, open('models/carInsurance_features.pkl', 'wb'))


def load_dataset():
    print("starting...")
    return load_data("dataset/carInsurance_train.csv")


def cleansed_data(df_pass):
    df = impute_missing_values(df_pass)
    # check for null values
    df.apply(lambda x: sum(x.isnull()), axis=0)
    return df


def execute_steps():

    df = load_dataset()

    df = cleansed_data(df)

    df = encode_features(df)

    df.drop(['Id'], axis=1, inplace=True)

    # train the model
    train_and_save_model(df)