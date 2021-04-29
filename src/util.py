import time
import sys
import pandas as pd

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


def impute_missing_values(df):
    df['Job'].fillna(value='missing', inplace=True)
    df['Education'].fillna(value='missing', inplace=True)
    df['Communication'].fillna(value='cellular', inplace=True)
    df['Outcome'].fillna(value='missing', inplace=True)
    return df


def encode_features(df):
    df = pd.get_dummies(df, columns=['Job', 'Marital', 'Education', 'Communication', 'LastContactMonth', 'Outcome'],
                        prefix_sep='_')
    df['call_duration'] = (pd.to_datetime(df['CallEnd']) - pd.to_datetime(df['CallStart'])).dt.total_seconds()
    df.drop(['CallStart', 'CallEnd'], axis=1, inplace=True)
    return df


def print_unique(df):
    category_names = df.columns
    for category in category_names:
        print("There are %d unique values in %s." % (df[category].nunique(), category))


def load_data(path):
    # Load Data
    start_time = time.time()
    df = pd.read_csv(path)
    print('[{}] Finished to load data'.format(time.time() - start_time))
    print('Train shape: ', df.shape)
    print("loaded training data")
    return df


def check_null_values(df):
    # check for null values
    print(df.apply(lambda x: sum(x.isnull()), axis=0))
