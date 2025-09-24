from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import Dict
import polars as pl
import xgboost as xgb

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from typing import Dict, Any, Tuple
import polars as pl


@transformer
def train_ml_model(data_tuple: Tuple[pl.DataFrame, pl.Series], *args, **kwargs):
    """
    Train an XGBoost model on the preprocessed data.
    
    Args:
        data_tuple: Tuple containing (features_df, target_series)
        
    Returns:
        Dictionary containing trained model and performance metrics
    """
    # Extract features and target from the input tuple
    features_df = data_tuple[0]
    target_series = data_tuple[1]
    
    # Convert Polars DataFrame to pandas for XGBoost compatibility
    X = features_df.to_pandas()
    y = target_series
    
    # Validate that we have data
    if X.empty or len(y) == 0:
        raise ValueError("Input data is empty")
    
    if len(X) != len(y):
        raise ValueError("Features and target must have the same length")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() < len(y) * 0.5 else None
    )
    
    # Determine if it's classification or regression based on target variable
    is_classification = y.dtype == 'object' or y.nunique() <= 10
    
    if is_classification:
        # Initialize XGBoost classifier
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        metrics = {
            'model_type': 'classification',
            'accuracy': accuracy,
            'classification_report': report,
            'test_predictions': y_pred.tolist(),
            'prediction_probabilities': y_pred_proba.tolist()
        }
        
    else:
        # Initialize XGBoost regressor
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='rmse'
        )
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate regression metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'model_type': 'regression',
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2,
            'test_predictions': y_pred.tolist()
        }
    
    # Get feature importance
    feature_importance = dict(zip(X.columns, model.feature_importances_))
    
    feature_names = list(X.columns)
    # Return comprehensive results
    results = {
        'model': model,
        'feature_names': feature_names,
        'feature_importance': feature_importance,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'target_classes': y.unique().tolist() if is_classification else None,
        **metrics
    }

    for key, value in results.items():
        print(f'{key}: {value}')

    return model, feature_names