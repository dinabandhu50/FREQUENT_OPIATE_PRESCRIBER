import pandas as pd 
import numpy as np 


# from package_fop import pipeline
# from package_fop import config
import package_fop.pipeline as pipeline
import package_fop.config as config

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