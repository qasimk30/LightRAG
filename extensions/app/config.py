import json
import os
import logging
from typing import Dict, Any
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration class loaded from JSON"""
    tenant_id: str
    lamma_cloud_key: str
    llm: Dict[str, Any]
    embedding: Dict[str, Any]
    storage: Dict[str, str]
    default_kb: str
    paths: Dict[str, str]
    ingestion: str
    kb_list: list[str] = field(default_factory=list)
    config_path: str = field(default="", repr=False)

    @classmethod
    def from_file(cls, config_path: str = "config.json"):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)

            # Set default kb_list if not present
            config_data.setdefault("kb_list", [])

            config = cls(**config_data)
            config.config_path = config_path
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file {config_path} not found")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def save(self):
        """Save only the updated kb_list to the original config file without modifying other fields"""
        try:
            with open(self.config_path, "r") as f:
                existing_config = json.load(f)

            # Only update kb_list in the original config dict
            existing_config["kb_list"] = self.kb_list

            with open(self.config_path, "w") as f:
                json.dump(existing_config, f, indent=2)

            logger.info(f"Updated kb_list and saved config to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise



def initialize_config(config_path: str = "config.json") -> Config:
    """Initialize configuration and create tenant-prefixed directories"""
    config = Config.from_file(config_path)
    tenant_prefix = os.path.join(".", config.tenant_id)

    updated_paths = {}
    for key, path in config.paths.items():
        # Normalize path with tenant_id prefix
        updated_path = os.path.normpath(os.path.join(tenant_prefix, path.lstrip("./\\")))
        updated_paths[key] = updated_path
        os.makedirs(updated_path, exist_ok=True)
        logger.info(f"Created directory for {key}: {updated_path}")

    config.paths = updated_paths
    return config
