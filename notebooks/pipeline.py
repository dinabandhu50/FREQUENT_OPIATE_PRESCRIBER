from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier


import preprocessors as pp
import config

pipe = Pipeline([
    ('drop',pp.FeaturesToDrop(config.DROP_COLS)),
    ('ohe',pp.OneHotCatEncoder(cols=config.BINARY_CAT_VARIABLES,drop='if_binary')),
    ('rare',pp.RareLabelCatEncoder(cols=config.MULTI_CAT_VARIABLES)),
    ('freq',pp.FrequencyCatEncoder(cols=config.MULTI_CAT_VARIABLES)),
    ('pca',pp.PCATransformer(cols=config.NUM_COLS,n_components=0.8)),
    ('scaler',MinMaxScaler((0, 100))),
    ('clf',GradientBoostingClassifier())
])