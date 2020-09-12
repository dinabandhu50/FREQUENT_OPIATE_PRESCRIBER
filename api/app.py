from flask import Flask, render_template, request, jsonify
import joblib
# import pickle
from src import preprocessors as pp
app = Flask(__name__)

# def init():
#     MODEL_PATH = './models/model.save'
#     global model_saved
#     saved_model = joblib.load(MODEL_PATH)


@app.route('/', methods=['GET', 'POST'])
def front_end():
    return render_template('home.html')


@app.route('/pred', methods=['POST'])
def prediction():
    data = request.get_json()
    pred = saved_model.predict(data)
    return jsonify(str(pred))


if __name__ == '__main__':
    MODEL_PATH = './models/model.save'
    # with open(MODEL_PATH, 'rb') as fid:
        # model_saved = pickle.load(fid)
    saved_model = joblib.load(MODEL_PATH)
    # init()
    # app.run(debug=True)
    