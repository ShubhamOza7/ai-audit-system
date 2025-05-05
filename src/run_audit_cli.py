#!/usr/bin/env python3

import sys
import os
import argparse
from datetime import datetime

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.model_loader import ModelLoader
from src.data_generator import DataGenerator
from tests.test_complete_system import (
    calculate_fairness_metrics,
    calculate_shap_values,
    check_compliance,
    generate_audit_report
)

def run_audit(model_file):
    print(f"Starting audit for model: {model_file}")
    
    try:
        # Initialize model loader
        model_loader = ModelLoader()
        model, model_id = model_loader.load_model(model_file)
        print(f"Model loaded: {model_id.version}")
        
        # Generate synthetic data
        print("Generating test data...")
        data_generator = DataGenerator(model_loader=model_loader)
        test_data = data_generator.generate_data()
        print(f"Generated {len(test_data)} test samples")
        
        # Generate predictions
        print("Running model predictions...")
        model_features = test_data[model_loader.expected_features]
        predictions = model.predict(model_features)
        print(f"Average prediction: {predictions.mean():.4f}")
        
        # Calculate fairness metrics
        print("Calculating fairness metrics...")
        fairness_analysis = {}
        for attr in ['gender', 'race', 'age']:
            print(f"  - Analyzing {attr}...")
            metrics = calculate_fairness_metrics(test_data, predictions, attr)
            fairness_analysis[attr] = metrics
            print(f"    Disparate impact: {metrics['disparate_impact_ratio']:.4f}")
            print(f"    Statistical parity difference: {metrics['statistical_parity_difference']:.4f}")
        
        # Calculate feature importance
        print("Calculating feature importance...")
        feature_importance = calculate_shap_values(model, test_data, model_loader)
        top_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        print("Top 5 most important features:")
        for feature, importance in top_features:
            print(f"  - {feature}: {importance:.4f}")
        
        # Check compliance
        print("Checking compliance...")
        results = {
            'test_data': test_data,
            'predictions': predictions,
            'model': model,
            'model_id': model_id,
            'model_loader': model_loader,
            'fairness_analysis': fairness_analysis,
            'feature_importance': feature_importance
        }
        
        compliance_results = check_compliance(
            results['fairness_analysis'],
            results['feature_importance'],
            results['predictions']
        )
        
        for standard, status in compliance_results.items():
            print(f"  - {standard}: {'PASS' if status else 'FAIL'}")
        
        # Generate report
        print("Generating audit report...")
        report, report_path = generate_audit_report(results, feature_importance)
        
        print(f"\nAudit completed successfully!")
        print(f"Report saved to: {report_path}")
        
        return True
        
    except Exception as e:
        print(f"Error during audit: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Model Audit CLI")
    parser.add_argument("model_file", help="Path to the model file to audit")
    args = parser.parse_args()
    
    if not os.path.exists(args.model_file):
        print(f"Error: Model file '{args.model_file}' not found.")
        sys.exit(1)
    
    success = run_audit(args.model_file)
    sys.exit(0 if success else 1) 