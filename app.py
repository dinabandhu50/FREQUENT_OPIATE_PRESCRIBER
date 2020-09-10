from flask import Flask, render_template, request, jsonify
import joblib
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def front_end():
    return render_template('home.html')


@app.route('/pred', methods=['POST'])
def prediction():
    data = request.get_json()
    pred = model_saved.predict(data)
    return jsonify(str(pred))


if __name__ == '__main__':
    MODEL_PATH = './models/model.save'
    # with open(MODEL_PATH, 'rb') as fid:
        # model_saved = pickle.load(fid)
    model_saved = joblib.load(MODEL_PATH)
    app.run(debug=True)
    