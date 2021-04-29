import os
import pickle as p

import numpy as np
import pandas as pd
from flask import Flask, jsonify, make_response, request

import pipeline
from util import encode_features

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        print("entered")
        json_request = request.get_json(force=True)
        print('Data Received: "{data}"'.format(data=json_request))

    df = pd.DataFrame.from_dict(json_request, orient="index").transpose()

    # preprocessing
    df['Communication'] = df['Communication'].replace('', "cellular")
    df['Outcome'] = df['Outcome'].replace('', "missing")
    df = encode_features(df)

    df["Age"] = df["Age"].astype(int)
    df["Default"] = df["Default"].astype(int)
    df["Balance"] = df["Balance"].astype(int)
    df["HHInsurance"] = df["HHInsurance"].astype(int)
    df["CarLoan"] = df["CarLoan"].astype(int)
    df["LastContactDay"] = df["LastContactDay"].astype(int)
    df["NoOfContacts"] = df["NoOfContacts"].astype(int)
    df["DaysPassed"] = df["DaysPassed"].astype(int)
    df["PrevAttempts"] = df["PrevAttempts"].astype(int)

    missing_cols = pd.DataFrame(0, index=np.arange(1), columns=features.difference(df.columns).tolist())
    df = pd.concat([df, missing_cols], axis=1)

    prediction = model.predict(df)

    if prediction == 1:
        response = make_response(
            jsonify(
                {"message": str("customer will mostly likely to be subscribed")}
            )
        )
    else:
        response = make_response(
            jsonify(
                {"message": str("customer will mostly unlikely to be subscribed")}
            )
        )
    return response


if __name__ == '__main__':
    pipeline.execute_steps()
    model = p.load(open('models/carInsurance_model.pkl', 'rb'))
    features = p.load(open('models/carInsurance_features.pkl', 'rb'))
    port = int(os.environ.get('PORT', 5000))
    ip_address = os.environ.get('ip_address', '0.0.0.0')
    app.run(debug=True, host=ip_address, port=port)
