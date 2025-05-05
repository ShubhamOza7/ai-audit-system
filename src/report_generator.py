import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom dataclasses
from src.model_loader import ModelMetadata
from src.model_explainer import ExplanationResult
from src.fairness_evaluator import FairnessMetrics

@dataclass
class GovernanceReport:
    """Structured container for the complete governance report"""
    timestamp: str
    model_info: Dict[str, Any]
    fairness_metrics: Dict[str, Any]
    explainability_results: Dict[str, Any]
    compliance_status: Dict[str, bool]
    recommendations: List[str]

class GovernanceReportGenerator:
    """
    Generates comprehensive AI governance reports that combine fairness and 
    explainability analyses while ensuring regulatory compliance.
    """
    
    def __init__(self, output_directory: str = "governance_reports"):
        """Initialize the report generator"""
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logging()
        
        # Store results for compliance checking
        self.model_metadata = None
        self.explanation_results = None
        
        # Define compliance thresholds
        self.compliance_thresholds = {
            'disparate_impact_min': 0.8,
            'statistical_parity_max': 0.1,
            'protected_attr_influence_max': 0.2
        }

    def _setup_logging(self) -> logging.Logger:
        """Configure logging for report generation process"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def generate_report(self,
                       model_metadata: ModelMetadata,
                       fairness_results: Dict[str, FairnessMetrics],
                       explanation_results: ExplanationResult,
                       test_data_summary: Dict[str, Any]) -> GovernanceReport:
        """Generate a comprehensive governance report"""
        self.logger.info("Starting governance report generation")
        
        try:
            # Store results for compliance checking
            self.model_metadata = model_metadata
            self.explanation_results = explanation_results
            
            # Assess compliance status
            compliance_status = self._verify_compliance(fairness_results, None)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                compliance_status,
                fairness_results,
                explanation_results
            )
            
            # Create report structure
            report = GovernanceReport(
                timestamp=datetime.now().isoformat(),
                model_info=self._format_model_info(model_metadata),
                fairness_metrics=self._format_fairness_metrics(fairness_results),
                explainability_results=self._format_explainability_results(
                    explanation_results
                ),
                compliance_status=compliance_status,
                recommendations=recommendations
            )
            
            # Save report artifacts
            self._save_report_artifacts(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating governance report: {str(e)}")
            raise

    def _format_model_info(self, model_metadata: ModelMetadata) -> Dict[str, Any]:
        """Format model metadata for the report"""
        return {
            'model_name': model_metadata.source_path.split('/')[-1] if model_metadata.source_path else 'Unknown',
            'version': model_metadata.version,
            'timestamp': model_metadata.timestamp,
            'source': model_metadata.source_path,
            'additional_info': model_metadata.additional_info if model_metadata.additional_info else {}
        }

    def _format_fairness_metrics(self, fairness_results: Dict[str, FairnessMetrics]) -> Dict:
        """Format fairness metrics for the report"""
        formatted_metrics = {}
        for attr, metrics in fairness_results.items():
            formatted_metrics[attr] = {
                'disparate_impact_ratio': metrics.disparate_impact_ratio,
                'statistical_parity_difference': metrics.statistical_parity_difference,
                'selection_rate_ratio': metrics.selection_rate_ratio,
                'group_metrics': metrics.group_metrics
            }
        return formatted_metrics

    def _format_explainability_results(self, explanation_results: ExplanationResult) -> Dict:
        """Format explainability results for the report"""
        return {
            'feature_importance': explanation_results.feature_importance,
            'sample_explanations': explanation_results.sample_explanations,
            'global_impact': explanation_results.global_impact,
            'protected_attribute_influence': explanation_results.protected_attribute_influence,
            'timestamp': explanation_results.timestamp,
            'model_type': explanation_results.model_type
        }

    def _verify_compliance(self, 
                          metrics: Dict[str, FairnessMetrics],
                          attr: str) -> Dict[str, bool]:
        """Verify compliance with regulatory requirements"""
        compliance_status = {}
        
        # Check ECOA compliance (fairness across protected groups)
        ecoa_compliant = True
        for metric in metrics.values():
            if metric.disparate_impact_ratio < self.compliance_thresholds['disparate_impact_min']:
                ecoa_compliant = False
                break
        compliance_status['ecoa_compliant'] = ecoa_compliant
        
        # Check GDPR Article 22 compliance (explainability)
        compliance_status['gdpr_article_22_compliant'] = (
            hasattr(self.explanation_results, 'feature_importance') and
            hasattr(self.explanation_results, 'sample_explanations') and
            self.explanation_results.feature_importance is not None and
            self.explanation_results.sample_explanations is not None
        )
        
        # Check ISO 27001 compliance (documentation and transparency)
        compliance_status['iso_27001_compliant'] = all([
            self.model_metadata is not None,
            metrics is not None,
            self.explanation_results is not None
        ])
        
        return compliance_status

    def _generate_recommendations(self,
                                compliance_status: Dict[str, bool],
                                fairness_results: Dict[str, FairnessMetrics],
                                explanation_results: ExplanationResult) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Check fairness-related issues
        for protected_attr, metrics in fairness_results.items():
            if metrics.disparate_impact_ratio < self.compliance_thresholds['disparate_impact_min']:
                recommendations.append(
                    f"Address disparate impact issues for {protected_attr} group. "
                    f"Current ratio: {metrics.disparate_impact_ratio:.2f}, "
                    f"Required: ≥ {self.compliance_thresholds['disparate_impact_min']}"
                )
        
        # Check explainability-related issues
        if not compliance_status['gdpr_article_22_compliant']:
            recommendations.append(
                "Enhance model documentation to fully comply with GDPR Article 22 "
                "right to explanation requirements"
            )
        
        # Add general recommendations if needed
        if not compliance_status['iso_27001_compliant']:
            recommendations.append(
                "Improve documentation and transparency to meet ISO 27001 requirements"
            )
        
        return recommendations

    def generate_human_readable_report(self, report: GovernanceReport) -> str:
        """Generate a detailed, human-readable report"""
        sections = []
        
        # Executive Summary
        sections.append(self._generate_executive_summary(report))
        
        # Model Information
        sections.append(self._generate_model_info_section(report.model_info))
        
        # Fairness Analysis
        sections.append(self._generate_fairness_section(report.fairness_metrics))
        
        # Explainability Analysis
        sections.append(self._generate_explainability_section(
            report.explainability_results
        ))
        
        # Compliance Status
        sections.append(self._generate_compliance_section(
            report.compliance_status
        ))
        
        # Recommendations
        if report.recommendations:
            sections.append(self._generate_recommendations_section(
                report.recommendations
            ))
        
        return "\n\n".join(sections)

    def _generate_executive_summary(self, report: GovernanceReport) -> str:
        """Generate the executive summary section"""
        summary = ["Executive Summary", "=" * 20]
        
        # Overall compliance status
        compliant = all(report.compliance_status.values())
        status = "COMPLIANT" if compliant else "NON-COMPLIANT"
        
        summary.append(f"Overall Status: {status}")
        summary.append(f"Report Generated: {report.timestamp}")
        
        # Get model name from model_info
        model_name = report.model_info['model_name']
        summary.append(f"Model Name: {model_name}")
        
        # Key findings
        summary.append("\nKey Findings:")
        if not compliant:
            failed_standards = [
                standard for standard, status in report.compliance_status.items()
                if not status
            ]
            summary.append("- Non-compliant with: " + ", ".join(failed_standards))
        
        if report.recommendations:
            summary.append(f"- {len(report.recommendations)} recommendations provided")
        
        return "\n".join(summary)

    def _generate_model_info_section(self, model_info: Dict[str, Any]) -> str:
        """Generate the model information section"""
        section = ["Model Information", "=" * 20]
        
        for key, value in model_info.items():
            if key != 'additional_info':
                section.append(f"{key.replace('_', ' ').title()}: {value}")
        
        if model_info.get('additional_info'):
            section.append("\nAdditional Information:")
            for key, value in model_info['additional_info'].items():
                section.append(f"- {key}: {value}")
        
        return "\n".join(section)

    def _generate_fairness_section(self, fairness_metrics: Dict[str, Any]) -> str:
        """Generate the fairness analysis section"""
        section = ["Fairness Analysis", "=" * 20]
        
        for attr, metrics in fairness_metrics.items():
            section.append(f"\nProtected Attribute: {attr}")
            section.append("-" * len(f"Protected Attribute: {attr}"))
            
            section.append(f"Disparate Impact Ratio: {metrics['disparate_impact_ratio']:.3f}")
            section.append(
                f"Statistical Parity Difference: {metrics['statistical_parity_difference']:.3f}"
            )
            
            section.append("\nGroup-specific metrics:")
            for group, group_metrics in metrics['group_metrics'].items():
                section.append(
                    f"  {group}:\n"
                    f"    Selection Rate: {group_metrics['selection_rate']:.3f}\n"
                    f"    Sample Size: {group_metrics['sample_size']}"
                )
        
        return "\n".join(section)

    def _generate_explainability_section(self, 
                                       explainability_results: Dict[str, Any]) -> str:
        """Generate the explainability analysis section"""
        section = ["Model Explainability", "=" * 20]
        
        # Feature importance
        if 'feature_importance' in explainability_results:
            section.append("\nFeature Importance Summary:")
            importance_items = sorted(
                explainability_results['feature_importance'].items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            for feature, importance in importance_items[:10]:  # Top 10 features
                section.append(f"- {feature}: {importance:.4f}")
        
        # Protected attribute influence
        if 'protected_attribute_influence' in explainability_results:
            section.append("\nProtected Attribute Influence:")
            for attr, influence in explainability_results['protected_attribute_influence'].items():
                section.append(f"- {attr}: {influence:.4f}")
        
        return "\n".join(section)

    def _generate_compliance_section(self, compliance_status: Dict[str, bool]) -> str:
        """Generate the compliance status section"""
        section = ["Regulatory Compliance", "=" * 20]
        
        for standard, status in compliance_status.items():
            status_text = "✓ Compliant" if status else "✗ Non-Compliant"
            section.append(f"{standard.replace('_', ' ').title()}: {status_text}")
        
        return "\n".join(section)

    def _generate_recommendations_section(self, recommendations: List[str]) -> str:
        """Generate the recommendations section"""
        section = ["Recommendations", "=" * 20]
        
        for i, recommendation in enumerate(recommendations, 1):
            section.append(f"{i}. {recommendation}")
        
        return "\n".join(section)

    def _save_report_artifacts(self, report: GovernanceReport) -> None:
        """Save report artifacts for documentation"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save JSON version
            json_path = self.output_dir / f"governance_report_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(
                    {
                        'timestamp': report.timestamp,
                        'model_info': report.model_info,
                        'fairness_metrics': report.fairness_metrics,
                        'explainability_results': report.explainability_results,
                        'compliance_status': report.compliance_status,
                        'recommendations': report.recommendations
                    },
                    f,
                    indent=4
                )
            
            # Save human-readable version
            text_path = self.output_dir / f"governance_report_{timestamp}.txt"
            with open(text_path, 'w') as f:
                f.write(self.generate_human_readable_report(report))
            
            self.logger.info(f"Report artifacts saved to {self.output_dir}")
        except Exception as e:
            self.logger.error(f"Error saving report artifacts: {str(e)}")
            raise