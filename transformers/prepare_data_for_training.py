from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
from typing import Any
import polars as pl
import numpy as np

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


def flatten_column_values(series):
    """Convert any non-scalar values in a series to strings."""
    def convert_value(val):
        # Check for DataFrame/Series first to avoid pd.isna() ambiguity
        if isinstance(val, (pd.DataFrame, pd.Series)):
            return str(val.to_dict()) if not val.empty else ""
        elif isinstance(val, (list, dict, tuple, set)):
            return str(val)
        elif not isinstance(val, (str, int, float, bool, type(None))):
            return str(val)
        else:
            # Only check pd.isna() for scalar values
            try:
                if pd.isna(val):
                    return val
            except Exception:
                pass
            return val
    return series.apply(convert_value)


def prepare_data(data_df):
    # Identify numerical and categorical columns
    numerical_cols = data_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = data_df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Handle missing values
    # For numerical columns, fill missing with median
    for col in numerical_cols:
        median_value = data_df[col].median()
        data_df[col].fillna(median_value, inplace=True)

    # For categorical columns, fill missing with mode
    for col in categorical_cols:
        # Flatten any complex values in the column first
        data_df[col] = flatten_column_values(data_df[col])
        
        # Calculate mode safely
        mode_series = data_df[col].mode()
        if len(mode_series) > 0:
            mode_value = mode_series.iloc[0]
            
            # Handle cases where mode might return a non-scalar
            if isinstance(mode_value, pd.DataFrame):
                mode_value = str(mode_value.to_dict()) if not mode_value.empty else ""
            elif isinstance(mode_value, pd.Series):
                mode_value = str(mode_value.to_dict()) if not mode_value.empty else ""
            elif pd.isna(mode_value):
                mode_value = ""
            elif not isinstance(mode_value, (str, int, float, bool)):
                mode_value = str(mode_value)
        else:
            mode_value = ""
        
        data_df[col].fillna(mode_value, inplace=True)

    # Define preprocessing for numerical data: scaling
    numerical_transformer = StandardScaler()

    # Define preprocessing for categorical data: one-hot encoding
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    # Fit and transform the data
    processed_array = preprocessor.fit_transform(data_df)

    # Get feature names after encoding
    encoded_categorical_cols = preprocessor.named_transformers_[
        "cat"
    ].get_feature_names_out(categorical_cols)
    feature_names = numerical_cols + list(encoded_categorical_cols)

    if not isinstance(processed_array, np.ndarray):
        processed_array = processed_array.toarray()

    # Create a DataFrame with processed features
    processed_df = pl.DataFrame(
        processed_array, schema=feature_names,
    )

    return processed_df, preprocessor


@transformer
def prepare_data_for_training(data: Any, *args, **kwargs) -> pd.DataFrame:
    """
    Preprocesses core data users for training:
    - Handles missing values
    - Encodes categorical variables
    - Scales numerical features
    Returns a cleaned DataFrame ready for model training.
    """
    # Copy the input DataFrame to avoid modifying original data
    data = pd.concat(data['load_api_data_from_ingestion'])
    data_df = data.copy().drop_duplicates()

    labels = data_df['survived']

    data_df = data_df.drop(['survived'], axis=1)
    features = data_df.columns

    processed_df, preprocessor = prepare_data(data_df)

    return processed_df, labels, list(features), preprocessor