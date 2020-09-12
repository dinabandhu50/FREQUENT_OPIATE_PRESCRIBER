import pandas as pd 
import numpy as np 


import pipeline 
import config 
import joblib

# training pipeline 
def run_training():
    # Data loading
    train_df = pd.read_csv(config.TRAIN_DATA_PATH)

    # data management
    X_train, y_train = train_df.drop(config.TARGET,axis=1), train_df[config.TARGET]

    # training
    pipeline.pipe.fit(X_train,y_train)

    # saving model
    joblib.dump(pipeline.pipe,config.SAVED_MODEL)

if __name__ == "__main__":
    run_training()