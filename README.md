# AI Audit System

A comprehensive system for auditing AI models for fairness, explainability, and regulatory compliance.

## Features

- **Fairness Assessment**: Evaluate models for bias across protected attributes (gender, race, age)
- **Feature Importance Analysis**: Understand which features are driving model decisions
- **Regulatory Compliance**: Check compliance with standards like ECOA, GDPR, and ISO 42001
- **Report Generation**: Generate detailed governance reports
- **REST API Interface**: Access functionality through an API
- **CLI Tool**: Run audits directly from the command line

## Setup

### Prerequisites

- Python 3.9+
- pip (Python package manager)

### Installation

1. Clone the repository:

   ```
   git clone https://github.com/your-username/ai-audit-system.git
   cd ai-audit-system
   ```

2. Create and activate a virtual environment:

   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install fastapi uvicorn python-multipart aif360 fairlearn "numpy<2.0.0"
   ```

## Usage

### Command Line Interface

Run an audit on a model using the CLI:

```
python3 src/run_audit_cli.py path/to/your/model.pkl
```

Example:

```
python3 src/run_audit_cli.py uploads/credit_random_model.pkl
```

The tool will:

1. Load the model
2. Generate test data
3. Analyze for fairness issues
4. Calculate feature importance
5. Check regulatory compliance
6. Generate a detailed report

### REST API

Start the API server:

```
python3 src/api/main.py
```

The API will be available at http://localhost:8000 with the following endpoints:

- `GET /` - Home page (upload form)
- `POST /upload-model` - Upload a model for analysis
- `POST /start-audit` - Start an audit on an uploaded model
- `GET /audit/{audit_id}` - Retrieve a specific audit report
- `GET /audit/{audit_id}/status` - Check the status of an audit
- `GET /metrics` - List available metrics and compliance standards

