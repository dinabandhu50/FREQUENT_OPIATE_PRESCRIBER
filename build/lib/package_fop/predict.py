import pandas as pd 

import joblib 
import package_fop import config

def make_prediction(input_data):
    # load model
    trained_model = joblib.load(config.SAVED_MODEL)
    result = trained_model.predict(input_data)
    return result

if __name__ == "__main__":
    # Data loading
    test_df = pd.read_csv(config.TEST_DATA_PATH)

    # data management
    X_test, y_test = test_df.drop(config.TARGET,axis=1), test_df[config.TARGET]

    # pred
    number_of_sample = 5
    y_pred = make_prediction(X_test.sample(number_of_sample,random_state=1))

    # printing actual and predicted
    df_ys = pd.DataFrame({
        'Actual': y_test.sample(number_of_sample,random_state=1).values.reshape(-1,),
        'Pred': y_pred
    })
    print(df_ys)
    