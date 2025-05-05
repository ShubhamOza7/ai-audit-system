import shap
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import json

@dataclass
class ExplanationResult:
    """Stores model explanation results"""
    timestamp: str
    model_type: str
    feature_importance: Dict[str, float]
    sample_explanations: Dict[int, Dict[str, float]]
    global_impact: Dict[str, Dict[str, float]]
    protected_attribute_influence: Dict[str, float]

class ModelExplainer:
    def __init__(self, 
                 output_path: str = "explanation_outputs",
                 protected_attributes: List[str] = None):
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        self.protected_attributes = protected_attributes or ['age', 'gender', 'race']
        self.logger = self._setup_logging()
        self.label_encoders = {}

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def explain_model(self, 
                     model: Any,
                     data: pd.DataFrame,
                     feature_names: List[str]) -> ExplanationResult:
        self.logger.info("Starting model explanation process")
        
        try:
            # Preprocess the data
            processed_data = self._preprocess_data(data)
            
            # Initialize SHAP explainer
            explainer = self._initialize_explainer(model, processed_data)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(processed_data)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Ensure we're working with 2D array
            if len(shap_values.shape) == 1:
                shap_values = shap_values.reshape(1, -1)
            
            # Generate explanations
            explanation_result = self._generate_explanations(
                shap_values,
                processed_data,
                feature_names
            )
            
            # Save explanation artifacts
            self._save_explanation_artifacts(explanation_result)
            
            return explanation_result
            
        except Exception as e:
            self.logger.error(f"Error during model explanation: {str(e)}")
            raise

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
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
            self.logger.info(f"Explanation encoding mapping for {column}: {mapping}")
        
        return processed_data

    def _initialize_explainer(self, model: Any, data: pd.DataFrame) -> shap.Explainer:
        try:
            if hasattr(model, "predict_proba"):
                if hasattr(model, "estimators_"):
                    return shap.TreeExplainer(model)
                
                background_data = shap.sample(data, min(100, len(data)))
                predict_fn = lambda x: model.predict_proba(x)[:, 1]
                return shap.KernelExplainer(predict_fn, background_data)
            
            return shap.KernelExplainer(model.predict, 
                                      shap.sample(data, min(100, len(data))))
        except Exception as e:
            self.logger.error(f"Failed to initialize explainer: {str(e)}")
            raise

    def _generate_explanations(self,
                             shap_values: np.ndarray,
                             data: pd.DataFrame,
                             feature_names: List[str]) -> ExplanationResult:
        try:
            # Calculate feature importance
            feature_importance = self.calculate_feature_importance(
                shap_values,
                feature_names
            )
            
            # Generate sample explanations
            sample_explanations = self.generate_sample_explanations(
                shap_values,
                data,
                feature_names
            )
            
            # Calculate global impact
            global_impact = self.analyze_global_impact(
                shap_values,
                data,
                feature_names
            )
            
            # Analyze protected attributes
            protected_influence = self.analyze_protected_attributes(
                shap_values,
                data,
                feature_names
            )
            
            return ExplanationResult(
                timestamp=datetime.now().isoformat(),
                model_type=str(type(data).__name__),
                feature_importance=feature_importance,
                sample_explanations=sample_explanations,
                global_impact=global_impact,
                protected_attribute_influence=protected_influence
            )
        except Exception as e:
            self.logger.error(f"Error generating explanations: {str(e)}")
            raise

    def calculate_feature_importance(self,
                                   shap_values: np.ndarray,
                                   feature_names: List[str]) -> Dict[str, float]:
        try:
            importance_values = np.mean(np.abs(shap_values), axis=0)
            importance_dict = {}
            
            for name, value in zip(feature_names, importance_values):
                if isinstance(value, np.ndarray):
                    if value.size == 1:
                        importance_dict[name] = float(value.item())
                    else:
                        importance_dict[name] = float(np.mean(value))
                else:
                    importance_dict[name] = float(value)
            
            return importance_dict
        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {str(e)}")
            raise

    def generate_sample_explanations(self,
                                   shap_values: np.ndarray,
                                   data: pd.DataFrame,
                                   feature_names: List[str]) -> Dict[int, Dict[str, float]]:
        try:
            sample_explanations = {}
            for idx in range(min(5, len(data))):
                contributions = {}
                for name, value in zip(feature_names, shap_values[idx]):
                    if isinstance(value, np.ndarray):
                        if value.size == 1:
                            contributions[name] = float(value.item())
                        else:
                            contributions[name] = float(np.mean(value))
                    else:
                        contributions[name] = float(value)
                sample_explanations[idx] = contributions
                
            return sample_explanations
        except Exception as e:
            self.logger.error(f"Error generating sample explanations: {str(e)}")
            raise

    def analyze_global_impact(self,
                            shap_values: np.ndarray,
                            data: pd.DataFrame,
                            feature_names: List[str]) -> Dict[str, Dict[str, float]]:
        try:
            impact_analysis = {}
            for idx, feature in enumerate(feature_names):
                feature_values = shap_values[:, idx]
                
                if isinstance(feature_values, np.ndarray):
                    impact_analysis[feature] = {
                        'mean_impact': float(np.mean(feature_values)),
                        'abs_mean_impact': float(np.mean(np.abs(feature_values))),
                        'max_impact': float(np.max(np.abs(feature_values))),
                        'std_impact': float(np.std(feature_values))
                    }
                else:
                    impact_analysis[feature] = {
                        'mean_impact': float(feature_values),
                        'abs_mean_impact': float(abs(feature_values)),
                        'max_impact': float(abs(feature_values)),
                        'std_impact': 0.0
                    }
            
            return impact_analysis
        except Exception as e:
            self.logger.error(f"Error analyzing global impact: {str(e)}")
            raise

    def analyze_protected_attributes(self,
                                  shap_values: np.ndarray,
                                  data: pd.DataFrame,
                                  feature_names: List[str]) -> Dict[str, float]:
        try:
            protected_influence = {}
            for attr in self.protected_attributes:
                if attr in feature_names:
                    idx = feature_names.index(attr)
                    feature_values = shap_values[:, idx]
                    
                    if isinstance(feature_values, np.ndarray):
                        influence = float(np.mean(np.abs(feature_values)))
                    else:
                        influence = float(abs(feature_values))
                        
                    protected_influence[attr] = influence
                    
                    if influence > 0.1:
                        self.logger.warning(
                            f"High influence detected for protected attribute {attr}"
                        )
            
            return protected_influence
        except Exception as e:
            self.logger.error(f"Error analyzing protected attributes: {str(e)}")
            raise

    def _save_explanation_artifacts(self, explanation: ExplanationResult) -> None:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.output_path / f"explanation_{timestamp}.json"
            
            explanation_dict = {
                'timestamp': explanation.timestamp,
                'model_type': explanation.model_type,
                'feature_importance': explanation.feature_importance,
                'sample_explanations': explanation.sample_explanations,
                'global_impact': explanation.global_impact,
                'protected_attribute_influence': explanation.protected_attribute_influence
            }
            
            with open(filepath, 'w') as f:
                json.dump(explanation_dict, f, indent=4)
            
            self.logger.info(f"Explanation artifacts saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving explanation artifacts: {str(e)}")
            raise