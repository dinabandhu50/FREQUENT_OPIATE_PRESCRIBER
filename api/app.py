from flask import Flask, render_template, request, jsonify

from package_fop.predict import make_prediction
from package_fop import config 

import pandas as pd
import joblib
import sys 

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def front_end():
    return render_template('home.html')

@app.route('/pred', methods=['GET', 'POST'])
def test():
    j_data = request.get_json()
    df = pd.read_json(j_data)
    
    # data management
    X_test, y_test = df.drop(config.TARGET,axis=1), df[config.TARGET]
    
    # pred
    y_pred = make_prediction(X_test)
    
    # printing to console in the server console
    # print(y_pred, file=sys.stdout)

    return {'Actual':str(y_test.values.reshape(-1,)),'Pred':str(y_pred)}


if __name__ == '__main__':
    app.run(debug=True)
    