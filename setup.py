from setuptools import setup, find_packages

setup(
    name='ai_audit_system',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # List your core dependencies here
        'aif360',
        'scikit-learn',
        # Add other dependencies from your requirements.txt
    ],
    extras_require={
        'adversarial': ['tensorflow'],
        'reductions': ['fairlearn'],
        'infairness': ['inFairness']
    }
)