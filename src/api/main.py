from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
from typing import Dict, List, Optional
import json
from datetime import datetime
import os
import sys
import shutil

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.model_loader import ModelLoader
from src.data_generator import DataGenerator
from tests.test_complete_system import (
    calculate_fairness_metrics,
    calculate_shap_values,
    check_compliance,
    generate_audit_report
)

app = FastAPI(
    title="AI Audit System API",
    description="API for conducting comprehensive AI model audits",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Root endpoint
@app.get("/")
async def home(request: Request):
    """Serve the upload form"""
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload-model")
async def upload_model(model_file: UploadFile = File(...)):
    """Handle model file upload"""
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)
        
        # Save uploaded file
        file_path = f"uploads/{model_file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(model_file.file, buffer)
        
        return {"filename": model_file.filename, "status": "uploaded"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start-audit")
async def start_audit(model_file: str = Form(...)):
    """Start audit process for uploaded model"""
    try:
        model_path = f"uploads/{model_file}"
        
        # Initialize model loader
        model_loader = ModelLoader()
        model, model_id = model_loader.load_model(model_path)
        
        # Generate synthetic data
        data_generator = DataGenerator(model_loader=model_loader)
        test_data = data_generator.generate_data()
        
        # Generate predictions
        model_features = test_data[model_loader.expected_features]
        predictions = model.predict(model_features)
        
        # Calculate fairness metrics
        fairness_analysis = {}
        for attr in ['gender', 'race', 'age']:
            metrics = calculate_fairness_metrics(test_data, predictions, attr)
            fairness_analysis[attr] = metrics
        
        # Calculate feature importance
        feature_importance = calculate_shap_values(model, test_data, model_loader)
        
        # Check compliance
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
        
        # Generate report
        report, report_path = generate_audit_report(results, feature_importance)
        
        # Prepare response
        response = {
            'audit_id': f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'results': {
                'model_info': {
                    'model_id': model_id.version,
                    'dataset_size': len(test_data),
                    'positive_rate': float(predictions.mean())
                },
                'fairness_analysis': fairness_analysis,
                'compliance_results': compliance_results,
                'report_path': report_path
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/audit/{audit_id}")
async def get_audit_report(audit_id: str):
    """Get the detailed report for a specific audit"""
    try:
        report_path = f"governance_reports/governance_report_{audit_id}.txt"
        if not os.path.exists(report_path):
            raise HTTPException(status_code=404, detail="Report not found")
            
        with open(report_path, 'r') as f:
            report_content = f.read()
            
        return {
            'audit_id': audit_id,
            'report_content': report_content
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/audit/{audit_id}/status")
async def get_audit_status(audit_id: str):
    """Get the current status of an audit"""
    try:
        # In a real implementation, this would check a database
        report_path = f"governance_reports/governance_report_{audit_id}.txt"
        status = "completed" if os.path.exists(report_path) else "not_found"
        
        return {
            'audit_id': audit_id,
            'status': status,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_available_metrics():
    """Get list of available metrics and their descriptions"""
    return {
        'fairness_metrics': [
            {'name': 'disparate_impact_ratio', 'description': 'Measures the ratio of selection rates between groups'},
            {'name': 'statistical_parity_difference', 'description': 'Measures the difference in selection rates'},
        ],
        'compliance_standards': [
            {'name': 'ECOA', 'description': 'Equal Credit Opportunity Act compliance'},
            {'name': 'GDPR Article 22', 'description': 'General Data Protection Regulation compliance'},
            {'name': 'ISO 42001', 'description': 'AI system governance standard compliance'}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting server on http://0.0.0.0:12345")
    uvicorn.run(app, host="0.0.0.0", port=12345) 