import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric

@dataclass
class FairnessMetrics:
    """Stores comprehensive fairness evaluation results"""
    timestamp: str
    protected_attribute: str
    disparate_impact_ratio: float
    statistical_parity_difference: float
    selection_rate_ratio: float
    group_metrics: Dict[str, Dict[str, float]]

class FairnessEvaluator:
    def __init__(self, prediction_threshold: float = 0.5):
        self.prediction_threshold = prediction_threshold
        self.logger = self._setup_logging()
        self.label_encoders = {}
        
        # Define regulatory thresholds
        self.min_disparate_impact_ratio = 0.8
        self.max_statistical_parity_diff = 0.1

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def evaluate_fairness(self, 
                         data: pd.DataFrame,
                         predictions: np.ndarray,
                         protected_attributes: List[str]) -> Dict[str, FairnessMetrics]:
        """
        Evaluate model fairness across all protected attributes.
        This is the main entry point for fairness evaluation.
        """
        self.logger.info("Starting fairness evaluation")
        
        # Preprocess the data
        processed_data = self._preprocess_data(data)
        
        # Convert probabilities to binary predictions if needed
        binary_predictions = (predictions >= self.prediction_threshold).astype(int)
        
        fairness_results = {}
        for attribute in protected_attributes:
            self.logger.info(f"Evaluating fairness for {attribute}")
            
            # Prepare data for AIF360
            dataset = self.prepare_aif360_dataset(
                processed_data, 
                binary_predictions, 
                attribute
            )
            
            # Calculate fairness metrics
            metrics = self._calculate_fairness_metrics(dataset, attribute)
            
            # Verify compliance
            self._verify_compliance(metrics, attribute)
            
            fairness_results[attribute] = metrics
        
        self.logger.info("Fairness evaluation completed")
        return fairness_results

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data by converting categorical variables to numerical"""
        processed_data = data.copy()
        
        categorical_columns = data.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
            
            processed_data[column] = self.label_encoders[column].fit_transform(
                processed_data[column]
            )
            
            mapping = dict(zip(
                self.label_encoders[column].classes_,
                self.label_encoders[column].transform(
                    self.label_encoders[column].classes_
                )
            ))
            self.logger.info(f"Fairness encoding mapping for {column}: {mapping}")
        
        return processed_data

    def prepare_aif360_dataset(self, 
                             data: pd.DataFrame,
                             predictions: np.ndarray,
                             protected_attribute: str) -> BinaryLabelDataset:
        """Prepare data in AIF360 format for fairness evaluation"""
        # Create a copy of data with predictions
        df = data.copy()
        df['prediction'] = predictions
        
        # Get privileged and unprivileged groups
        privileged_value = self._identify_privileged_groups(df, protected_attribute)
        
        # Create privileged and unprivileged groups lists
        privileged_groups = [{protected_attribute: privileged_value}]
        unprivileged_groups = [{protected_attribute: val} for val in 
                              df[protected_attribute].unique() if val != privileged_value]
        
        # Convert to AIF360 format
        dataset = BinaryLabelDataset(
            df=df,
            label_names=['prediction'],
            protected_attribute_names=[protected_attribute],
            privileged_protected_attributes=[[privileged_value]],
            favorable_label=1,
            unfavorable_label=0
        )
        
        # Set the privileged and unprivileged groups
        dataset.privileged_groups = privileged_groups
        dataset.unprivileged_groups = unprivileged_groups
        
        return dataset

    def _identify_privileged_groups(self, data: pd.DataFrame, attribute: str) -> int:
        """Identify privileged groups for fairness comparison"""
        if attribute.lower() == 'gender':
            return self.label_encoders[attribute].transform(['M'])[0]
        elif attribute.lower() == 'race':
            return self.label_encoders[attribute].transform(['White'])[0]
        elif attribute.lower() == 'age':
            return int(data[attribute].median())
        else:
            return data[attribute].mode()[0]

    def _calculate_fairness_metrics(self, 
                                  dataset: BinaryLabelDataset,
                                  attribute: str) -> FairnessMetrics:
        """Calculate fairness metrics using AIF360"""
        metrics = BinaryLabelDatasetMetric(
            dataset,
            unprivileged_groups=dataset.unprivileged_groups,
            privileged_groups=dataset.privileged_groups
        )
        
        # Calculate group-specific metrics
        group_metrics = {}
        for value in np.unique(dataset.protected_attributes):
            condition = dataset.protected_attributes == value
            group_predictions = dataset.labels[condition]
            
            # Map numerical value back to original category if possible
            if attribute in self.label_encoders:
                original_value = self.label_encoders[attribute].inverse_transform([int(value)])[0]
            else:
                original_value = str(value)
            
            group_metrics[original_value] = {
                'selection_rate': float(np.mean(group_predictions)),
                'sample_size': int(np.sum(condition))
            }
        
        # Calculate selection rates for privileged and unprivileged groups
        priv_mask = dataset.protected_attributes == self._identify_privileged_groups(
            pd.DataFrame(dataset.features, columns=dataset.feature_names),
            attribute
        )
        priv_selection_rate = float(np.mean(dataset.labels[priv_mask]))
        unpriv_selection_rate = float(np.mean(dataset.labels[~priv_mask]))
        
        # Calculate selection rate ratio
        selection_rate_ratio = (
            unpriv_selection_rate / priv_selection_rate 
            if priv_selection_rate > 0 
            else 0.0
        )
        
        return FairnessMetrics(
            timestamp=datetime.now().isoformat(),
            protected_attribute=attribute,
            disparate_impact_ratio=metrics.disparate_impact(),
            statistical_parity_difference=metrics.statistical_parity_difference(),
            selection_rate_ratio=selection_rate_ratio,
            group_metrics=group_metrics
        )

    def _verify_compliance(self, metrics: FairnessMetrics, attribute: str) -> None:
        """Verify that fairness metrics meet regulatory requirements"""
        if metrics.disparate_impact_ratio < self.min_disparate_impact_ratio:
            message = (
                f"Disparate impact ratio ({metrics.disparate_impact_ratio:.2f}) "
                f"for {attribute} is below regulatory threshold of "
                f"{self.min_disparate_impact_ratio}"
            )
            self.logger.error(message)
            raise ValueError(message)
        
        if abs(metrics.statistical_parity_difference) > self.max_statistical_parity_diff:
            self.logger.warning(
                f"High statistical parity difference detected for {attribute}: "
                f"{metrics.statistical_parity_difference:.2f}"
            )