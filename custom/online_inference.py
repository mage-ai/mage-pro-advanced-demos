import importlib
import pandas as pd
import polars as pl

from default_repo.transformers import prepare_data_for_training

importlib.reload(prepare_data_for_training)

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@custom
def transform_custom(data, preparation, *args, **kwargs):
    model = data[0]
    # feature_names = data[1]

    features = preparation[2]
    preprocessor = preparation[3]

    row = {}
    for idx, feature in enumerate(features):
        print(f'{idx}. {feature}')
        value = kwargs.get(feature)
        row[feature] = value    

    df = pd.DataFrame([row])
    df_prepared = preprocessor.transform(df)

    y = model.predict(df_prepared)

    df['survived'] = y

    return df