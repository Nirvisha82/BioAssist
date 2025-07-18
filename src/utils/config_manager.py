"""
Configuration manager for the RAG chatbot.
"""
import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger


class ConfigManager:
    """Manages configuration from YAML files and environment variables."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config_data = {}
        
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        self._load_config()
        
        logger.info(f"Configuration loaded from {config_path}")
    
    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, encoding='utf-8') as file:
                    self.config_data = yaml.safe_load(file) or {}
            else:
                logger.warning(f"Config file not found: {self.config_path}")
                self.config_data = {}
                
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            self.config_data = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'llm.model_name')."""
        try:
            # First check environment variables
            env_key = key.upper().replace('.', '_')
            env_value = os.getenv(env_key)
            if env_value is not None:
                return self._convert_env_value(env_value)
            
            # Then check config file
            keys = key.split('.')
            value = self.config_data
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
            
        except Exception as e:
            logger.error(f"Error getting config value for {key}: {str(e)}")
            return default
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer conversion
        try:
            if '.' not in value:
                return int(value)
        except ValueError:
            pass
        
        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # String (default)
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        keys = key.split('.')
        config = self.config_data
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_data_source_paths(self) -> Dict[str, str]:
        """Get paths for different data source directories."""
        base_dir = self.get("document_processing.data_source_dir", "./data_source")
        
        return {
            "pdf": os.path.join(base_dir, "pdf"),
            "docx": os.path.join(base_dir, "docx"),
            "txt": os.path.join(base_dir, "txt"),
            "xlsx": os.path.join(base_dir, "xlsx"),
            "pptx": os.path.join(base_dir, "pptx")
        }
    
    def get_vector_db_path(self) -> str:
        """Get vector database path."""
        return self.get("vector_db.persist_directory", "./vector_db")
    
    def get_streamlit_config(self) -> Dict[str, Any]:
        """Get Streamlit-specific configuration."""
        return {
            "title": self.get("streamlit.title", "RAG Chatbot"),
            "port": self.get("streamlit.port", 8501),
            "chat_history_limit": self.get("streamlit.chat_history_limit", 50),
            "page_config": self.get("streamlit.page_config", {
                "page_title": "RAG Chatbot",
                "page_icon": "🌍",
                "layout": "wide"
            })
        }
    
    def validate_config(self) -> bool:
        """Validate essential configuration values."""
        required_keys = [
            "llm.provider",
            "llm.model_name",
            "embeddings.model_name",
            "vector_db.type",
            "vector_db.persist_directory"
        ]
        
        missing_keys = []
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)
        
        if missing_keys:
            logger.error(f"Missing required configuration keys: {missing_keys}")
            return False
        
        # Check if Google API key is available
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            logger.error("GOOGLE_API_KEY environment variable is required")
            return False
        
        return True
    
    def create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.get_vector_db_path(),
            self.get("logging.file", "./logs/rag_chatbot.log").rsplit('/', 1)[0],
        ]
        
        # Add data source directories
        data_source_paths = self.get_data_source_paths()
        directories.extend(data_source_paths.values())
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")


# Global config instance
config = ConfigManager()