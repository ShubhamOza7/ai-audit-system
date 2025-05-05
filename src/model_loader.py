import pickle
import json
import datetime
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Tuple
import joblib
import pandas as pd

@dataclass
class ModelMetadata:
    """
    A structured container for storing essential model metadata.
    This helps maintain consistent tracking of model information.
    """
    timestamp: str          # When the model was loaded
    version: str           # Version identifier for the model
    source_path: str       # Where the model file came from
    additional_info: Dict[str, Any] = None  # Any extra information we want to track

class ModelLoader:
    """
    Handles loading of pickle-format ML models and tracks their metadata.
    This class ensures consistent model loading and proper logging for audit purposes.
    """
    
    def __init__(self, log_directory: str = "audit_logs"):
        """
        Initialize the model loader with a specified log directory.
        
        Args:
            log_directory: Where to store audit logs and metadata files
        """
        # Create log directory if it doesn't exist
        self.log_path = Path(log_directory)
        self.log_path.mkdir(exist_ok=True)
        
        # Set up logging configuration
        self._setup_logging()
        
        self.model = None
        self.expected_features = None
        
    def _setup_logging(self) -> None:
        """
        Configure the logging system for tracking model operations.
        Creates a formatted logger that writes to both file and console.
        """
        logging.basicConfig(
            filename=self.log_path / "model_loader.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Add console handler for immediate feedback
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)

    def load_model(self, model_path: str) -> Tuple[Any, ModelMetadata]:
        """
        Load a pickled model and generate its metadata.
        
        Args:
            model_path: Path to the pickle file containing the model
            
        Returns:
            tuple: (loaded_model, model_metadata)
            
        Raises:
            ValueError: If the file is not a pickle file
            FileNotFoundError: If the model file doesn't exist
        """
        # Validate file extension
        if not model_path.endswith('.pkl'):
            raise ValueError("Only .pkl files are supported")
            
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            # Load the model
            self.logger.info(f"Loading model from {model_path}")
            self.model = joblib.load(model_path)
            
            # Get and log the model's features
            self.expected_features = getattr(self.model, 'feature_names_in_', None)
            if self.expected_features is None:
                # Try alternative attribute names
                self.expected_features = getattr(self.model, 'feature_names', None)
                if self.expected_features is None:
                    raise ValueError("Could not determine model's feature names")
            
            self.logger.info(f"Model expects these features: {self.expected_features}")
            
            # Generate metadata
            metadata = self._create_metadata(model_path)
            
            # Log the metadata
            self._save_metadata(metadata)
            
            self.logger.info(f"Successfully loaded model: {metadata.version}")
            return self.model, metadata
            
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {str(e)}")
            raise

    def _create_metadata(self, model_path: Path) -> ModelMetadata:
        """
        Create metadata for a loaded model.
        
        Args:
            model_path: Path object pointing to the model file
            
        Returns:
            ModelMetadata object containing model information
        """
        return ModelMetadata(
            timestamp=datetime.datetime.now().isoformat(),
            version=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            source_path=str(model_path.absolute()),
            additional_info={
                "file_size": model_path.stat().st_size,
                "last_modified": datetime.datetime.fromtimestamp(
                    model_path.stat().st_mtime
                ).isoformat()
            }
        )

    def _save_metadata(self, metadata: ModelMetadata) -> None:
        """
        Save model metadata to a JSON file and log the action.
        
        Args:
            metadata: ModelMetadata object to be saved
        """
        # Convert metadata to dictionary for JSON storage
        metadata_dict = {
            'timestamp': metadata.timestamp,
            'version': metadata.version,
            'source_path': metadata.source_path,
            'additional_info': metadata.additional_info
        }
        
        # Create metadata filename
        metadata_file = self.log_path / f"model_metadata_{metadata.version}.json"
        
        # Save metadata to JSON file
        with open(metadata_file, 'w') as f:
            json.dump(metadata_dict, f, indent=4)
        
        self.logger.info(f"Metadata saved to {metadata_file}")

    def preprocess_data(self, df):
        """Dynamically adapt input data to match model's expected features"""
        if self.expected_features is None:
            raise ValueError("Model not loaded. Please load model first.")

        processed_df = pd.DataFrame()
        
        # For each feature the model expects
        for feature in self.expected_features:
            if feature in df.columns:
                # If feature exists in input, use it
                processed_df[feature] = df[feature]
            else:
                # If feature doesn't exist, add it with zeros
                # You might want to log this for monitoring
                logging.warning(f"Missing feature '{feature}' in input data. Adding with zeros.")
                processed_df[feature] = 0

        return processed_df