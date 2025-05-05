import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

@dataclass
class PredictionMetadata:
    """Stores metadata about a prediction run"""
    timestamp: str
    model_version: str
    num_samples: int
    protected_attributes: List[str]
    feature_names: List[str]

class ModelTester:
    def __init__(self, protected_attributes: List[str] = None):
        self.protected_attributes = protected_attributes or ['age', 'gender', 'race']
        self.logger = self._setup_logging()
        self.label_encoders = {}  # Store label encoders for each categorical variable

    def _setup_logging(self) -> logging.Logger:
        """Configure logging for the testing process"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data by converting categorical variables to numerical values.
        Stores the label encoders for later use in interpreting results.
        """
        processed_data = data.copy()
        
        # Convert categorical columns to numerical
        categorical_columns = data.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
            
            processed_data[column] = self.label_encoders[column].fit_transform(
                processed_data[column]
            )
            
            # Log the encoding mappings for reference
            mapping = dict(zip(
                self.label_encoders[column].classes_,
                self.label_encoders[column].transform(
                    self.label_encoders[column].classes_
                )
            ))
            self.logger.info(f"Encoding mapping for {column}: {mapping}")
        
        return processed_data

    def test_model(self, 
                  model: Any, 
                  test_data: pd.DataFrame,
                  feature_columns: List[str] = None) -> Tuple[pd.DataFrame, PredictionMetadata]:
        """
        Test a model with synthetic data and collect predictions with metadata.
        Now includes preprocessing of categorical variables.
        """
        self.logger.info("Starting model testing process")
        
        # Validate input data
        self._validate_test_data(test_data)
        
        # Determine feature columns if not specified
        if feature_columns is None:
            feature_columns = [col for col in test_data.columns 
                             if col not in self.protected_attributes]
        
        # Preprocess the data
        processed_data = self._preprocess_data(test_data)
        features_df = processed_data[feature_columns].copy()
        
        # Record the initial state of protected attributes
        protected_data = processed_data[self.protected_attributes].copy()
        
        try:
            # Generate predictions
            self.logger.info("Generating predictions")
            predictions = self._get_predictions(model, features_df)
            
            # Create results DataFrame - use original (non-encoded) data for results
            results = test_data.copy()
            results['prediction'] = predictions
            
            # Create metadata
            metadata = PredictionMetadata(
                timestamp=datetime.now().isoformat(),
                model_version=self._get_model_version(model),
                num_samples=len(test_data),
                protected_attributes=self.protected_attributes,
                feature_names=feature_columns
            )
            
            # Verify prediction compliance using processed data
            self._verify_prediction_compliance(
                processed_data.assign(prediction=predictions), 
                protected_data
            )
            
            self.logger.info("Model testing completed successfully")
            return results, metadata
            
        except Exception as e:
            self.logger.error(f"Error during model testing: {str(e)}")
            raise



    def _validate_test_data(self, data: pd.DataFrame) -> None:
        """
        Validate that test data meets requirements for fairness testing.
        Ensures we have sufficient representation of different groups.
        
        Args:
            data: DataFrame to validate
        """
        # Check for protected attributes
        missing_protected = [attr for attr in self.protected_attributes 
                           if attr not in data.columns]
        if missing_protected:
            raise ValueError(f"Missing protected attributes: {missing_protected}")
        
        # Check for sufficient representation of different groups
        for attr in self.protected_attributes:
            value_counts = data[attr].value_counts()
            if len(value_counts) < 2:
                raise ValueError(f"Insufficient diversity in {attr}")
            
            # Check for minimum representation (1% of data)
            min_representation = len(data) * 0.01
            if any(value_counts < min_representation):
                self.logger.warning(
                    f"Some groups in {attr} have less than 1% representation"
                )

    def _get_predictions(self, model: Any, features: pd.DataFrame) -> np.ndarray:
        """
        Get predictions from the model while handling different model types.
        Supports both probability and class predictions.
        
        Args:
            model: The model object
            features: DataFrame of feature values
        """
        # Handle different types of predict methods
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(features)[:, 1]  # Get positive class probability
        else:
            return model.predict(features)

    def _verify_prediction_compliance(self, 
                                   results: pd.DataFrame,
                                   protected_data: pd.DataFrame) -> None:
        """
        Verify that predictions comply with fairness requirements.
        Checks for concerning patterns in predictions across protected groups.
        
        Args:
            results: DataFrame with predictions
            protected_data: Original protected attributes
        """
        # Check for suspicious correlation with protected attributes
        for attr in self.protected_attributes:
            correlation = np.corrcoef(
                results['prediction'],
                pd.factorize(protected_data[attr])[0]
            )[0, 1]
            
            if abs(correlation) > 0.95:  # Threshold for concerning correlation
                raise ValueError(
                    f"Predictions show suspicious correlation with {attr}"
                )
        
        # Check for balanced predictions across protected groups
        for attr in self.protected_attributes:
            group_means = results.groupby(attr)['prediction'].mean()
            max_diff = group_means.max() - group_means.min()
            
            if max_diff > 0.3:  # Threshold for concerning prediction disparity
                self.logger.warning(
                    f"Large prediction disparity detected across {attr} groups: "
                    f"{max_diff:.2f}"
                )

    def _get_model_version(self, model: Any) -> str:
        """
        Extract version information from model if available.
        Falls back to 'unknown' if version info isn't found.
        """
        if hasattr(model, 'metadata'):
            return getattr(model, 'metadata').get('version', 'unknown')
        return 'unknown'
