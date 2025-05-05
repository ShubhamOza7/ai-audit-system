import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
import logging
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

@dataclass
class DataGenerationConfig:
    """Configuration settings for synthetic data generation"""
    num_samples: int
    include_protected_attributes: bool = True
    include_edge_cases: bool = True
    edge_case_percentage: float = 0.1
    random_seed: Optional[int] = None

class SyntheticFinancialDataGenerator:
    """
    Generates synthetic financial data for model testing while ensuring privacy 
    compliance and adequate representation of edge cases.
    """
    
    def __init__(self, config: DataGenerationConfig):
        """
        Initialize the data generator with configuration parameters.
        
        Args:
            config: Configuration parameters for data generation
        """
        self.config = config
        self.logger = self._setup_logging()
        
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
            
        # Define protected attribute categories for realistic distributions
        self.gender_categories = ['M', 'F', 'Other']
        self.race_categories = ['Asian', 'Black', 'Hispanic', 'White', 'Other']
        self.age_ranges = [(18, 25), (26, 35), (36, 50), (51, 65), (66, 90)]

    def _setup_logging(self) -> logging.Logger:
        """Configure logging for the data generation process"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def generate_loan_data(self) -> pd.DataFrame:
        """
        Generate synthetic loan application data with realistic correlations
        and edge cases.
        
        Returns:
            DataFrame containing synthetic loan data
        """
        self.logger.info(f"Generating {self.config.num_samples} synthetic loan records")
        
        # Generate main financial features
        data = self._generate_base_financial_features()
        
        # Add protected attributes if configured
        if self.config.include_protected_attributes:
            data = self._add_protected_attributes(data)
        
        # Add edge cases if configured
        if self.config.include_edge_cases:
            data = self._inject_edge_cases(data)
        
        self.logger.info("Synthetic data generation completed")
        return data

    def _generate_base_financial_features(self) -> pd.DataFrame:
        """Generate base financial features with realistic correlations"""
        n = self.config.num_samples
        
        # Generate correlated income and credit score
        income = np.exp(np.random.normal(10.5, 0.5, n))  # Log-normal distribution
        credit_score_noise = np.random.normal(0, 20, n)
        credit_score = np.clip(600 + 100 * stats.norm.cdf(np.log(income/40000)) + 
                             credit_score_noise, 300, 850)
        
        # Generate loan amount requests based on income
        loan_amount = np.clip(income * np.random.uniform(0.5, 2.5, n), 1000, 1000000)
        
        # Generate debt-to-income ratio
        dti = np.clip(np.random.normal(0.3, 0.1, n) + 
                     0.2 * (1 - stats.norm.cdf(np.log(income/50000))), 0, 0.6)
        
        # Create employment length with realistic distribution
        employment_length = np.random.gamma(shape=2, scale=5, size=n)
        employment_length = np.clip(employment_length, 0, 40)
        
        return pd.DataFrame({
            'annual_income': income.round(2),
            'credit_score': credit_score.astype(int),
            'loan_amount': loan_amount.round(2),
            'debt_to_income': dti.round(4),
            'employment_length': employment_length.round(1)
        })

    def _add_protected_attributes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add protected attributes while maintaining realistic distributions"""
        n = len(data)
        
        # Generate age with realistic distribution
        age = np.random.gamma(shape=15, scale=2, size=n) + 18
        age = np.clip(age, 18, 90).astype(int)
        
        # Generate gender with realistic proportions
        gender = np.random.choice(
            self.gender_categories,
            size=n,
            p=[0.49, 0.49, 0.02]  # Approximate real-world proportions
        )
        
        # Generate race with approximate US demographic proportions
        race = np.random.choice(
            self.race_categories,
            size=n,
            p=[0.06, 0.13, 0.19, 0.57, 0.05]
        )
        
        data['age'] = age
        data['gender'] = gender
        data['race'] = race
        
        return data

    def _inject_edge_cases(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Inject edge cases to ensure comprehensive model testing.
        Edge cases include unusual but possible scenarios.
        """
        n_edge_cases = int(len(data) * self.config.edge_case_percentage)
        
        edge_cases = pd.DataFrame({
            # Young high-income individuals
            'annual_income': np.random.uniform(150000, 300000, n_edge_cases),
            'age': np.random.randint(23, 30, n_edge_cases),
            'credit_score': np.random.randint(700, 850, n_edge_cases),
            
            # Add other protected attributes
            'gender': np.random.choice(self.gender_categories, n_edge_cases),
            'race': np.random.choice(self.race_categories, n_edge_cases),
            
            # Generate other features
            'loan_amount': np.random.uniform(500000, 1000000, n_edge_cases),
            'debt_to_income': np.random.uniform(0.1, 0.2, n_edge_cases),
            'employment_length': np.random.uniform(1, 5, n_edge_cases)
        })
        
        # Replace some original records with edge cases
        indices_to_replace = np.random.choice(
            len(data), n_edge_cases, replace=False
        )
        
        for col in edge_cases.columns:
            data.loc[indices_to_replace, col] = edge_cases[col].values
        
        self.logger.info(f"Injected {n_edge_cases} edge cases into the dataset")
        return data

class DataGenerator:
    """Generates synthetic data based on model features"""
    def __init__(self, model_loader=None, num_samples=1000):
        self.logger = logging.getLogger(__name__)
        
        if model_loader is None:
            raise ValueError("Model loader is required")
        
        if not hasattr(model_loader, 'expected_features') or model_loader.expected_features is None:
            raise ValueError("Model loader must be initialized with a loaded model")
        
        self.model_loader = model_loader
        self.num_samples = num_samples
        self.edge_case_percentage = 0.1
        self.expected_features = list(model_loader.expected_features)
        
        # Always include these protected attributes
        self.protected_attributes = {
            'age': {'min': 18, 'max': 80},
            'gender': {'categories': ['F', 'M', 'Other']},
            'race': {'categories': ['Asian', 'Black', 'Hispanic', 'White', 'Other']}
        }
        
        # Analyze model features to build configuration
        self.feature_config = self._analyze_model_features(self.expected_features)
    
    def _analyze_model_features(self, feature_names):
        """Automatically analyze model features to create generation rules"""
        config = {
            'numeric_features': [],
            'categorical_features': [],
            'onehot_groups': {}
        }
        
        # Group features by type and identify one-hot encoded groups
        prefixes = {}
        for feature in feature_names:
            # Identify one-hot encoded groups
            if '_' in feature:
                prefix = feature.rsplit('_', 1)[0] + '_'
                if prefix not in prefixes:
                    prefixes[prefix] = []
                prefixes[prefix].append(feature.split(prefix)[1])
            else:
                # Analyze feature name to determine type
                if any(x in feature.lower() for x in ['amount', 'income', 'age', 'length', 'percent']):
                    config['numeric_features'].append(feature)
                else:
                    config['categorical_features'].append(feature)
        
        # Convert identified prefixes to onehot groups
        for prefix, values in prefixes.items():
            if len(values) > 1:  # Only if multiple values exist
                group_name = prefix.rstrip('_')
                config['onehot_groups'][group_name] = {
                    'prefix': prefix,
                    'categories': sorted(values)  # Sort to ensure consistent order
                }
        
        return config

    def generate_data(self):
        """Generate synthetic data based on model's expected features"""
        data = {}
        
        # First generate all required model features
        for feature in self.expected_features:
            if feature in self.feature_config['numeric_features']:
                data[feature] = self._generate_numeric_feature(feature, self.num_samples)
            elif feature in self.feature_config['categorical_features']:
                data[feature] = self._generate_categorical_feature(feature, self.num_samples)
            else:
                # Initialize one-hot encoded features with zeros
                data[feature] = np.zeros(self.num_samples)
        
        # Handle one-hot encoded groups
        for group_config in self.feature_config['onehot_groups'].values():
            prefix = group_config['prefix']
            categories = group_config['categories']
            
            # For each sample, set one category to 1
            for i in range(self.num_samples):
                selected = np.random.choice(categories)
                data[prefix + selected][i] = 1
        
        # Generate protected attributes
        for attr, rules in self.protected_attributes.items():
            if 'categories' in rules:
                data[attr] = np.random.choice(rules['categories'], self.num_samples)
            else:
                data[attr] = np.random.randint(rules['min'], rules['max'], self.num_samples)
        
        # Create DataFrame with all features
        df = pd.DataFrame(data)
        
        # Add correlations and dependencies
        self._add_correlations(df)
        
        # Generate edge cases
        self._generate_edge_cases(df)
        
        # Ensure all required features are present and in correct order
        df_model = df[self.expected_features]
        df_protected = df[list(self.protected_attributes.keys())]
        
        # Combine and return
        final_df = pd.concat([df_model, df_protected], axis=1)
        self.logger.info(f"Generated {self.num_samples} synthetic records with {len(final_df.columns)} features")
        
        return final_df
    
    def _generate_numeric_feature(self, feature_name, n_samples):
        """Generate numeric features based on feature name"""
        if 'age' in feature_name.lower():
            return np.random.randint(18, 80, n_samples)
        elif 'income' in feature_name.lower():
            return np.random.uniform(20000, 200000, n_samples)
        elif 'amount' in feature_name.lower() or 'amnt' in feature_name.lower():
            return np.random.uniform(1000, 100000, n_samples)
        elif 'length' in feature_name.lower():
            return np.random.randint(0, 30, n_samples)
        elif 'percent' in feature_name.lower():
            return np.random.uniform(0, 1, n_samples)
        else:
            return np.random.uniform(0, 100, n_samples)
    
    def _generate_categorical_feature(self, feature_name, n_samples):
        """Generate categorical features based on feature name"""
        if 'default' in feature_name.lower():
            return np.random.choice(['Y', 'N'], n_samples)
        else:
            return np.random.choice(['A', 'B', 'C'], n_samples)
    
    def _add_correlations(self, df):
        """Add realistic correlations between features"""
        if 'person_income' in df.columns and 'age' in df.columns:
            age_factor = (df['age'] - 18) / 62
            df['person_income'] *= (1 + 0.5 * age_factor)
        
        if 'loan_percent_income' in df.columns and 'loan_amnt' in df.columns and 'person_income' in df.columns:
            df['loan_percent_income'] = df['loan_amnt'] / df['person_income']
            df['loan_percent_income'] = np.clip(df['loan_percent_income'], 0, 1)
    
    def _generate_edge_cases(self, df):
        """Generate edge cases for numeric features"""
        n_edge_cases = int(len(df) * self.edge_case_percentage)
        edge_indices = np.random.choice(len(df), n_edge_cases, replace=False)
        
        for col in df.select_dtypes(include=[np.number]).columns:
            if 'percent' in col.lower() or any(col.startswith(p) for p in ['person_home_ownership_', 'loan_intent_', 'loan_grade_']):
                continue
            df.loc[edge_indices, col] = df.loc[edge_indices, col] * 1.5
            
        self.logger.info(f"Generated {n_edge_cases} edge cases")

    def generate_biased_data(self, n_samples=1000, bias_factor=0.7):
        """Generate biased synthetic data"""
        df = self.generate_data()
        
        # Find a suitable numeric feature to bias
        numeric_features = [f for f in self.model_loader.expected_features 
                          if 'income' in f.lower() or 'amount' in f.lower()]
        
        if numeric_features:
            bias_feature = numeric_features[0]
            mask = np.random.random(n_samples) < bias_factor
            df.loc[mask, bias_feature] *= 0.5
            
        return df
