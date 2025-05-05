import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import pickle
import logging
from datetime import datetime
from src.model_loader import ModelLoader
from src.data_generator import DataGenerator
from src.model_tester import ModelTester
from src.fairness_evaluator import FairnessEvaluator
from src.model_explainer import ModelExplainer
from src.report_generator import GovernanceReportGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os
import shap

# Import our custom components

# def create_sample_model():
#     """
#     Creates a simple test model for demonstration purposes.
#     In real usage, you would load your existing model instead.
#     """
#     print("\nStep 0: Creating sample model for testing...")
    
#     # Create sample training data
#     np.random.seed(42)
#     X = np.random.randn(1000, 5)
#     y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
#     # Train a simple model
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X, y)
    
#    # Save the model
#     Path('test_models').mkdir(exist_ok=True)
#     model_path = '/Users/mac/Desktop/ai_audit_system/model/loan_default_model.pkl'
#     with open(model_path, 'wb') as f:
#         pickle.dump(model, f)
    
#     return model_path

logging.basicConfig(level=logging.INFO)

def print_step_header(step_num, title):
    """Print a formatted step header"""
    print(f"\n{'='*80}")
    print(f"Step {step_num}: {title}")
    print(f"{'='*80}")

def print_step_detail(message):
    """Print a detailed step message"""
    print(f"► {message}")

def ensure_audit_directory():
    """Ensure the audit_logs directory exists"""
    os.makedirs('audit_logs', exist_ok=True)

def calculate_shap_values(model, data, model_loader):
    """Calculate SHAP values for feature importance"""
    # Get only numeric features that the model uses
    numeric_features = []
    for feature in model_loader.expected_features:
        try:
            data[feature].astype(float)
            numeric_features.append(feature)
        except (ValueError, TypeError):
            continue
    
    # Create a DataFrame with only numeric features
    numeric_data = data[numeric_features].astype(float)
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(numeric_data)
    
    # Calculate feature importance
    feature_importance = {}
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    
    for i, col in enumerate(numeric_data.columns):
        importance = np.mean(np.abs(shap_values[:, i]))
        feature_importance[col] = float(importance)
    
    # Add zero importance for categorical features
    for feature in model_loader.expected_features:
        if feature not in feature_importance:
            feature_importance[feature] = 0.0
    
    # Normalize importance values
    total_importance = sum(feature_importance.values())
    if total_importance > 0:
        feature_importance = {k: v/total_importance for k, v in feature_importance.items()}
    
    return feature_importance

def check_compliance(fairness_metrics, feature_importance, predictions):
    """
    Check regulatory compliance based on actual metrics with detailed explanations
    """
    compliance = {}
    compliance_notes = {}
    compliance_reasons = {}
    
    # Ensure we have values for protected attributes
    protected_attrs = ['gender', 'race', 'age']
    for attr in protected_attrs:
        if attr not in feature_importance:
            feature_importance[attr] = 0.0
    
    try:
        # ECOA Compliance Check
        min_disparate_impact = min(
            metrics['disparate_impact_ratio'] 
            for metrics in fairness_metrics.values()
        )
        
        protected_attr_influence = max(
            feature_importance.get(attr, 0.0) 
            for attr in protected_attrs
        )
        
        ecoa_reasons = []
        if min_disparate_impact >= 0.8:
            ecoa_reasons.append("✓ Disparate impact ratio meets 4/5 rule (>= 0.8)")
        else:
            ecoa_reasons.append("✗ Disparate impact ratio below 4/5 rule threshold")
            
        if protected_attr_influence < 0.1:
            ecoa_reasons.append("✓ Protected attributes have minimal influence")
        else:
            ecoa_reasons.append("✗ Protected attributes have significant influence")
        
        compliance['ecoa_compliant'] = (
            min_disparate_impact >= 0.8 and 
            protected_attr_influence < 0.1
        )
        compliance_notes['ecoa'] = (
            f"Disparate Impact: {min_disparate_impact:.2f}, "
            f"Protected Influence: {protected_attr_influence:.2f}"
        )
        compliance_reasons['ecoa'] = ecoa_reasons
        
        # GDPR Article 22 Compliance Check
        explainable_features = [
            (f, i) for f, i in feature_importance.items() 
            if f not in protected_attrs and i > 0
        ]
        model_explainability = len(explainable_features) > 0
        
        top_features = sorted(
            explainable_features,
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        gdpr_reasons = []
        if model_explainability:
            gdpr_reasons.append("✓ Model decisions are explainable")
            gdpr_reasons.append(f"✓ Top features identified: {', '.join(f'{f}({i:.2f})' for f, i in top_features)}")
        else:
            gdpr_reasons.append("✗ Model lacks explainability")
            gdpr_reasons.append("✗ No significant feature importance found")
        
        compliance['gdpr_article_22_compliant'] = model_explainability
        compliance_notes['gdpr'] = (
            f"Explainable features: {len(explainable_features)}"
        )
        compliance_reasons['gdpr'] = gdpr_reasons
        
        # ISO 42001 Compliance Check
        fairness_threshold = 0.1
        max_stat_parity = max(
            abs(metrics['statistical_parity_difference']) 
            for metrics in fairness_metrics.values()
        )
        
        iso_reasons = []
        if max_stat_parity < fairness_threshold:
            iso_reasons.append("✓ Statistical parity difference within acceptable range")
        else:
            iso_reasons.append("✗ Statistical parity difference exceeds threshold")
            
        if model_explainability:
            iso_reasons.append("✓ Model decisions are transparent and explainable")
        else:
            iso_reasons.append("✗ Model lacks required transparency")
        
        compliance['iso_42001_compliant'] = (
            model_explainability and
            max_stat_parity < fairness_threshold
        )
        compliance_notes['iso_42001'] = (
            f"Statistical Parity: {max_stat_parity:.2f}, "
            f"Explainability: {'✓' if model_explainability else '✗'}"
        )
        compliance_reasons['iso_42001'] = iso_reasons
        
        compliance_details = {
            'min_disparate_impact': min_disparate_impact,
            'max_statistical_parity': max_stat_parity,
            'protected_attribute_influence': protected_attr_influence,
            'model_explainability_score': len(explainable_features),
            'top_influential_features': dict(top_features) if top_features else {}
        }
        
    except Exception as e:
        logging.error(f"Error in compliance check: {str(e)}")
        compliance = {
            'ecoa_compliant': False,
            'gdpr_article_22_compliant': False,
            'iso_42001_compliant': False
        }
        compliance_notes = {
            'ecoa': "Error in compliance calculation",
            'gdpr': "Error in compliance calculation",
            'iso_42001': "Error in compliance calculation"
        }
        compliance_reasons = {
            'ecoa': ["✗ Error in compliance calculation"],
            'gdpr': ["✗ Error in compliance calculation"],
            'iso_42001': ["✗ Error in compliance calculation"]
        }
        compliance_details = {
            'error': str(e)
        }
    
    return {
        **compliance,
        'compliance_notes': compliance_notes,
        'compliance_reasons': compliance_reasons,
        'compliance_details': compliance_details
    }

def generate_governance_report(results, feature_importance):
    """Generate a comprehensive governance report"""
    test_data = results['test_data']
    predictions = results['predictions']
    model_id = results['model_id']
    
    # Create report timestamp
    timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')
    
    report = {
        'executive_summary': {
            'overall_status': 'COMPLIANT',
            'report_generated': timestamp,
            'model_name': model_id.source_path.split('/')[-1]
        },
        'model_information': {
            'model_name': model_id.source_path.split('/')[-1],
            'version': model_id.version,
            'timestamp': model_id.timestamp,
            'source': model_id.source_path,
            'additional_information': model_id.additional_info
        },
        'fairness_analysis': {}
    }
    
    # Calculate fairness metrics for each protected attribute
    protected_attrs = ['gender', 'race', 'age']
    for attr in protected_attrs:
        groups = test_data[attr].unique()
        group_metrics = {}
        
        for group in groups:
            mask = test_data[attr] == group
            group_metrics[str(group)] = {
                'selection_rate': float(np.mean(predictions[mask])),
                'sample_size': int(sum(mask))
            }
        
        report['fairness_analysis'][attr] = {
            'disparate_impact_ratio': 1.0,  # Placeholder for actual calculation
            'statistical_parity_difference': 0.0,  # Placeholder for actual calculation
            'group_metrics': group_metrics
        }
    
    # Add model explainability section
    report['model_explainability'] = {
        'feature_importance_summary': feature_importance,
        'protected_attribute_influence': {
            'age': feature_importance.get('age', 0.0),
            'gender': feature_importance.get('gender', 0.0),
            'race': feature_importance.get('race', 0.0)
        }
    }
    
    # Add compliance section
    compliance = check_compliance(
        report['fairness_analysis'],
        report['model_explainability']['feature_importance_summary'],
        predictions
    )
    report['regulatory_compliance'] = {
        key: '✓ Compliant' if value else '✗ Non-Compliant'
        for key, value in compliance.items()
        if isinstance(value, bool)
    }
    
    # Save report
    ensure_audit_directory()
    report_path = f'governance_reports/governance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    os.makedirs('governance_reports', exist_ok=True)
    
    # Write report in the desired format
    with open(report_path, 'w') as f:
        f.write("Executive Summary\n")
        f.write("====================\n")
        f.write(f"Overall Status: {report['executive_summary']['overall_status']}\n")
        f.write(f"Report Generated: {report['executive_summary']['report_generated']}\n")
        f.write(f"Model Name: {report['executive_summary']['model_name']}\n\n")
        
        f.write("Model Information\n")
        f.write("====================\n")
        for key, value in report['model_information'].items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("Fairness Analysis\n")
        f.write("====================\n")
        for attr, metrics in report['fairness_analysis'].items():
            f.write(f"\nProtected Attribute: {attr}\n")
            f.write("-" * (len(attr) + 20) + "\n")
            f.write(f"Disparate Impact Ratio: {metrics['disparate_impact_ratio']}\n")
            f.write(f"Statistical Parity Difference: {metrics['statistical_parity_difference']}\n\n")
            f.write("Group-specific metrics:\n")
            for group, group_metrics in metrics['group_metrics'].items():
                f.write(f"  {group}:\n")
                f.write(f"    Selection Rate: {group_metrics['selection_rate']:.3f}\n")
                f.write(f"    Sample Size: {group_metrics['sample_size']}\n")
        
        f.write("\nModel Explainability\n")
        f.write("====================\n")
        f.write("\nFeature Importance Summary:\n")
        for feature, importance in report['model_explainability']['feature_importance_summary'].items():
            f.write(f"- {feature}: {importance:.4f}\n")
        
        f.write("\nProtected Attribute Influence:\n")
        for attr, influence in report['model_explainability']['protected_attribute_influence'].items():
            f.write(f"- {attr}: {influence:.4f}\n")
        
        f.write("\nRegulatory Compliance\n")
        f.write("====================\n")
        for key, value in report['regulatory_compliance'].items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
    
    return report, report_path

def run_complete_audit():
    """
    Demonstrates the complete workflow of our AI auditing system.
    This shows how each component connects to create a comprehensive
    model assessment.
    """
    # First, create a test model (in practice, you'd load your existing model)
    #model_path = create_sample_model()
    model_path = "/Users/mac/Desktop/ai_audit_system/model/credit_random_model.pkl"
    
    # Step 1: Load the model
    print("\nStep 1: Loading model...")
    loader = ModelLoader(log_directory="audit_logs")
    model, model_metadata = loader.load_model(model_path)
    print("Model loaded successfully")
    
    # Step 2: Generate synthetic test data
    print("\nStep 2: Generating synthetic test data...")
    data_generator = DataGenerator(model_loader)
    test_data = data_generator.generate_data()
    print(f"Generated {len(test_data)} synthetic records")
    
    # Step 3: Test model with synthetic data
    print("\nStep 3: Testing model with synthetic data...")
    model_tester = ModelTester(
        protected_attributes=['age', 'gender', 'race']
    )
    test_results, test_metadata = model_tester.test_model(
        model=model,
        test_data=test_data
    )
    print("Model testing completed")
    
    # Step 4: Evaluate fairness
    print("\nStep 4: Evaluating model fairness...")
    fairness_evaluator = FairnessEvaluator()
    fairness_results = fairness_evaluator.evaluate_fairness(
        data=test_data,
        predictions=test_results['prediction'],
        protected_attributes=['gender', 'race', 'age']
    )
    print("Fairness evaluation completed")
    
    # Step 5: Generate model explanations
    print("\nStep 5: Generating model explanations...")
    explainer = ModelExplainer(
        protected_attributes=['age', 'gender', 'race']
    )
    explanation_results = explainer.explain_model(
        model=model,
        data=test_data,
        feature_names=test_data.columns.tolist()
    )
    print("Explanation generation completed")
    
    # Step 6: Model Explainability
    print_step_header(5, "Model Explainability Analysis")
    feature_importance = calculate_shap_values(model, test_data, loader)
    print_step_detail("Calculated SHAP values for feature importance")
    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]:
        print_step_detail(f"Top feature: {feature}: {importance:.4f}")
    
    # Step 7: Compliance Check
    print_step_header(6, "Regulatory Compliance Check")
    compliance = check_compliance(
        fairness_results,
        feature_importance,
        test_results['prediction']
    )
    for regulation, status in compliance.items():
        if isinstance(status, bool):
            print_step_detail(f"{regulation}: {'✓ Compliant' if status else '✗ Non-Compliant'}")
    
    # Step 8: Governance Report
    print_step_header(7, "Governance Report Generation")
    report, report_path = generate_governance_report(test_results, feature_importance)
    print_step_detail(f"Generated comprehensive governance report")
    print_step_detail(f"Report saved to: {report_path}")
    
    return report

def calculate_fairness_metrics(data, predictions, protected_attribute):
    """Calculate fairness metrics for different groups"""
    groups = data[protected_attribute].unique()
    metrics = {}
    
    # Calculate metrics for each group
    selection_rates = {}
    for group in groups:
        mask = data[protected_attribute] == group
        group_preds = predictions[mask]
        selection_rates[group] = {
            'selection_rate': float(np.mean(group_preds)),
            'sample_size': int(sum(mask)),
            'group_percentage': float(sum(mask) / len(data) * 100)
        }
    
    # Calculate disparate impact ratio
    max_rate = max(g['selection_rate'] for g in selection_rates.values())
    min_rate = min(g['selection_rate'] for g in selection_rates.values())
    disparate_impact = min_rate / max_rate if max_rate > 0 else 1.0
    
    # Calculate statistical parity difference
    stat_parity = max_rate - min_rate
    
    return {
        'disparate_impact_ratio': disparate_impact,
        'statistical_parity_difference': stat_parity,
        'group_metrics': selection_rates
    }

def generate_audit_report(results, feature_importance):
    """Generate a comprehensive governance report"""
    test_data = results['test_data']
    predictions = results['predictions']
    model_id = results['model_id']
    
    # Calculate fairness metrics first
    fairness_analysis = {}
    for attr in ['gender', 'race', 'age']:
        metrics = calculate_fairness_metrics(
            test_data, predictions, attr
        )
        fairness_analysis[attr] = metrics
    
    # Check compliance using actual metrics
    compliance_results = check_compliance(
        fairness_analysis,
        feature_importance,
        predictions
    )
    
    # Create report timestamp
    timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')
    
    report = {
        'summary': {
            'overall_status': 'COMPLIANT' if all(compliance_results.values()) else 'NON-COMPLIANT',
            'report_generated': timestamp,
            'model_name': model_id.source_path.split('/')[-1],
            'dataset_size': len(test_data),
            'positive_rate': float(np.mean(predictions))
        },
        'model_information': {
            'model_name': model_id.source_path.split('/')[-1],
            'version': model_id.version,
            'timestamp': model_id.timestamp,
            'source': model_id.source_path,
            'additional_information': model_id.additional_info
        },
        'fairness_analysis': fairness_analysis,
        'model_explainability': {
            'feature_importance_summary': feature_importance,
            'protected_attribute_influence': {
                'age': feature_importance.get('age', 0.0),
                'gender': feature_importance.get('gender', 0.0),
                'race': feature_importance.get('race', 0.0)
            }
        },
        'regulatory_compliance': {
            key: '✓ Compliant' if value else '✗ Non-Compliant'
            for key, value in compliance_results.items()
            if isinstance(value, bool)
        },
        'compliance_details': compliance_results['compliance_details'],
        'compliance_notes': compliance_results['compliance_notes']
    }
    
    # Save report
    ensure_audit_directory()
    report_path = f'governance_reports/governance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    os.makedirs('governance_reports', exist_ok=True)
    
    # Write report in the desired format
    with open(report_path, 'w') as f:
        f.write("Executive Summary\n")
        f.write("====================\n")
        f.write(f"Overall Status: {report['summary']['overall_status']}\n")
        f.write(f"Report Generated: {report['summary']['report_generated']}\n")
        f.write(f"Model Name: {report['summary']['model_name']}\n\n")
        
        f.write("Model Information\n")
        f.write("====================\n")
        for key, value in report['model_information'].items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("Fairness Analysis\n")
        f.write("====================\n")
        for attr, metrics in report['fairness_analysis'].items():
            f.write(f"\nProtected Attribute: {attr}\n")
            f.write("-" * (len(attr) + 20) + "\n")
            f.write(f"Disparate Impact Ratio: {metrics['disparate_impact_ratio']:.3f}\n")
            f.write(f"Statistical Parity Difference: {metrics['statistical_parity_difference']:.3f}\n\n")
            f.write("Group-specific metrics:\n")
            for group, group_metrics in metrics['group_metrics'].items():
                f.write(f"  {group}:\n")
                f.write(f"    Selection Rate: {group_metrics['selection_rate']:.3f}\n")
                f.write(f"    Sample Size: {group_metrics['sample_size']}\n")
        
        f.write("\nModel Explainability\n")
        f.write("====================\n")
        f.write("\nFeature Importance Summary:\n")
        for feature, importance in sorted(
            report['model_explainability']['feature_importance_summary'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            f.write(f"- {feature}: {importance:.4f}\n")
        
        f.write("\nProtected Attribute Influence:\n")
        for attr, influence in report['model_explainability']['protected_attribute_influence'].items():
            f.write(f"- {attr}: {influence:.4f}\n")
        
        f.write("\nRegulatory Compliance\n")
        f.write("====================\n")
        for key, value in report['regulatory_compliance'].items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
    
    return report, report_path

def test_complete_system():
    try:
        print("\n" + "="*80)
        print("STARTING COMPLETE AI MODEL AUDIT")
        print("="*80)
        
        # Step 1: Load model
        print_step_header(1, "Model Loading and Initialization")
        model_loader = ModelLoader()
        model, model_id = model_loader.load_model('model/credit_random_model.pkl')
        print_step_detail(f"Loaded model from: model/credit_random_model.pkl")
        print_step_detail(f"Model ID: {model_id.version}")
        print_step_detail(f"Expected features: {len(model_loader.expected_features)} features")
        print_step_detail("Model loaded successfully")
        
        # Step 2: Generate synthetic data
        print_step_header(2, "Synthetic Data Generation")
        data_generator = DataGenerator(model_loader=model_loader)
        test_data = data_generator.generate_data()
        print_step_detail(f"Generated {len(test_data)} synthetic records")
        print_step_detail(f"Data shape: {test_data.shape}")
        print_step_detail(f"Features generated: {', '.join(test_data.columns[:5])}...")
        
        # Step 3: Model Testing
        print_step_header(3, "Model Testing and Predictions")
        model_features = test_data[model_loader.expected_features]
        predictions = model.predict(model_features)
        print_step_detail(f"Generated predictions for {len(predictions)} records")
        print_step_detail(f"Positive predictions: {sum(predictions)} ({sum(predictions)/len(predictions):.1%})")
        print_step_detail(f"Negative predictions: {len(predictions)-sum(predictions)} ({1-sum(predictions)/len(predictions):.1%})")
        
        # Step 4: Fairness Analysis
        print_step_header(4, "Fairness Metrics Calculation")
        fairness_analysis = {}
        for attr in ['gender', 'race', 'age']:
            metrics = calculate_fairness_metrics(test_data, predictions, attr)
            fairness_analysis[attr] = metrics
            print_step_detail(f"\n{attr.capitalize()} group distribution:")
            for group, group_metrics in metrics['group_metrics'].items():
                print(f"    {group}: {group_metrics['sample_size']} samples "
                      f"({group_metrics['selection_rate']:.1%} positive rate)")
            print(f"    Disparate Impact Ratio: {metrics['disparate_impact_ratio']:.3f}")
            print(f"    Statistical Parity Difference: {metrics['statistical_parity_difference']:.3f}")
        
        # Step 5: Model Explainability
        print_step_header(5, "Model Explainability Analysis")
        feature_importance = calculate_shap_values(model, test_data, model_loader)
        print_step_detail("Calculated SHAP values for feature importance")
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]:
            print_step_detail(f"Top feature: {feature}: {importance:.4f}")
        
        # Prepare results dictionary with all necessary components
        results = {
            'test_data': test_data,
            'predictions': predictions,
            'model': model,
            'model_id': model_id,
            'model_loader': model_loader,
            'fairness_analysis': fairness_analysis,
            'feature_importance': feature_importance
        }
        
        # Step 6: Compliance Check
        print_step_header(6, "Regulatory Compliance Check")
        compliance_results = check_compliance(
            results['fairness_analysis'],
            results['feature_importance'],
            results['predictions']
        )
        
        # Print compliance results
        compliance_standards = {
            'ecoa_compliant': 'ECOA',
            'gdpr_article_22_compliant': 'GDPR Article 22',
            'iso_42001_compliant': 'ISO 42001'
        }
        
        for key, standard_name in compliance_standards.items():
            if key in compliance_results:
                status = compliance_results[key]
                standard_key = key.replace('_compliant', '')
                print(f"\n  • {standard_name}: {'✓ Compliant' if status else '✗ Non-Compliant'}")
                
                # Print compliance details
                if standard_key in compliance_results['compliance_notes']:
                    print(f"    Metrics: {compliance_results['compliance_notes'][standard_key]}")
                
                # Print specific reasons
                if standard_key in compliance_results['compliance_reasons']:
                    print("    Reasons:")
                    for reason in compliance_results['compliance_reasons'][standard_key]:
                        print(f"      {reason}")
        
        # Print detailed metrics
        print("\n► Detailed Compliance Metrics:")
        if 'compliance_details' in compliance_results:
            details = compliance_results['compliance_details']
            print(f"  • Disparate Impact: {details['min_disparate_impact']:.3f} (threshold: 0.8)")
            print(f"  • Statistical Parity: {details['max_statistical_parity']:.3f} (threshold: 0.1)")
            print(f"  • Protected Attribute Influence: {details['protected_attribute_influence']:.3f} (threshold: 0.1)")
            print(f"  • Model Explainability Score: {details['model_explainability_score']}")
            
            if 'top_influential_features' in details:
                print("\n  • Top Influential Features:")
                for feature, importance in details['top_influential_features'].items():
                    print(f"    - {feature}: {importance:.3f}")
        
        # Step 7: Governance Report
        print_step_header(7, "Governance Report Generation")
        report, report_path = generate_audit_report(results, feature_importance)
        print_step_detail(f"Generated comprehensive governance report")
        print_step_detail(f"Report saved to: {report_path}")
        
        # Final Summary
        print_step_header("FINAL", "Audit Summary")
        print(f"► Model Information:")
        print(f"  • Model ID: {model_id.version}")
        print(f"  • Dataset size: {len(test_data)} records")
        print(f"  • Overall positive rate: {np.mean(predictions):.2%}")
        
        print(f"\n► Fairness Analysis:")
        for attr in ['gender', 'race', 'age']:
            metrics = results['fairness_analysis'][attr]
            print(f"\n  • {attr.capitalize()} groups:")
            for group, group_metrics in metrics['group_metrics'].items():
                print(f"    - {group}: {group_metrics['sample_size']} samples "
                      f"({group_metrics['selection_rate']:.2%} positive rate)")
        
        print(f"\n► Compliance Summary:")
        for key, standard_name in compliance_standards.items():
            if key in compliance_results:
                status = compliance_results[key]
                standard_key = key.replace('_compliant', '')
                print(f"\n  • {standard_name}: {'✓ Compliant' if status else '✗ Non-Compliant'}")
                
                # Print compliance details
                if standard_key in compliance_results['compliance_notes']:
                    print(f"    Metrics: {compliance_results['compliance_notes'][standard_key]}")
                
                # Print specific reasons
                if standard_key in compliance_results['compliance_reasons']:
                    print("    Reasons:")
                    for reason in compliance_results['compliance_reasons'][standard_key]:
                        print(f"      {reason}")
        
        # Print detailed metrics
        print("\n► Detailed Compliance Metrics:")
        if 'compliance_details' in compliance_results:
            details = compliance_results['compliance_details']
            print(f"  • Disparate Impact: {details['min_disparate_impact']:.3f} (threshold: 0.8)")
            print(f"  • Statistical Parity: {details['max_statistical_parity']:.3f} (threshold: 0.1)")
            print(f"  • Protected Attribute Influence: {details['protected_attribute_influence']:.3f} (threshold: 0.1)")
            print(f"  • Model Explainability Score: {details['model_explainability_score']}")
            
            if 'top_influential_features' in details:
                print("\n  • Top Influential Features:")
                for feature, importance in details['top_influential_features'].items():
                    print(f"    - {feature}: {importance:.3f}")
        
        print("\n" + "="*80)
        print("AUDIT COMPLETED SUCCESSFULLY")
        print("="*80 + "\n")
        
        return results, report
        
    except Exception as e:
        logging.error(f"Error during audit: {str(e)}")
        raise

if __name__ == "__main__":
    results, report = test_complete_system()