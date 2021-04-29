import pickle
import sys
import numpy as np
import pandas as pd
from src.util import load_data, impute_missing_values, encode_features

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

if __name__ == "__main__":
    print("test starting...")
    test = load_data('../input/carInsurance_test.csv')

    # check for null values
    print(test.apply(lambda x: sum(x.isnull()), axis=0))

    test = impute_missing_values(test)
    print(test.apply(lambda x: sum(x.isnull()), axis=0))

    test = encode_features(test)

    result_df: pd.DataFrame = test[['Id']]
    test.drop(['Id', 'CarInsurance'], axis=1, inplace=True)

    model = pickle.load(open('model.pkl', 'rb'))
    predcitions = model.predict(test)
    result_df['CarInsurance'] = np.array(predcitions)
    result_df.to_csv("../test_dataset_prediction_output/result.csv", index=False)
