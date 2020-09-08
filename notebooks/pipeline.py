from sklearn.pipeline import Pipeline

import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.ensemble import GradientBoostingClassifier


import preprocessors as pp
import config

lb = LabelBinarizer()
pipe = Pipeline([
    ('label_enc',lb),
    # ('label_enc',ce.BinaryEncoder(cols=config.ORDINAL_ENCODING_VARIABLES)),
    # ('taget_enc',ce.TargetEncoder(cols=config.TARGET_ENCODING_VARIABLES)),
    # ('num_enc',pp.PCATransformer(config.NUM_COLS,n_components=0.8)),
    ('scaler', MinMaxScaler()),
    # ('gbc',GradientBoostingClassifier())
])