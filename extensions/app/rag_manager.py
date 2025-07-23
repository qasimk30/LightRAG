import logging
import os
from pathlib import Path
from typing import Optional
from extensions.app.config import Config
from extensions.app.models import ModelManager
from extensions.app.config import initialize_config
from extensions.lightrag_extension import LightRAG_EXTENSIONS 
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

logger = logging.getLogger(__name__)

class RAGManager:
    """Manages RAG instances and tenant operations with focus on creating new instances"""

    def __init__(self, config: Config):
        self.config = config
        self.model_manager = ModelManager(config)
        self.rag_instance = None
        self._cached_llm_func = None
        self._cached_embedding_func = None
        self._cached_lightrag_params = None
        
        # Instance storage
        self._instances = {}  # { tenant_id: { kb_name: LightRAG_EXTENSIONS } }
        self.lamma_key = config.lamma_cloud_key

    @classmethod
    def from_config_file(cls, config_path: str = None):
        """Create RAGManager by auto-detecting config file"""
        if config_path is None:
            possible_paths = [
                "./config.json",
                "./extensions/app/config.json",
                os.getenv("CONFIG_PATH", "")
            ]
            
            for path in possible_paths:
                if path and os.path.exists(path):
                    config_path = path
                    break
            
            if not config_path:
                raise FileNotFoundError("Config file not found. Please specify config_path or set CONFIG_PATH env var")
        
        try:
            config = initialize_config(config_path)
            config.config_path = config_path
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise
        
        return cls(config)

    def _make_namespace_prefix(self, tenant_id: str, kb_name: str) -> str:
        """Create namespace prefix for tenant and KB combination"""
        return f"{tenant_id}__{kb_name}"

    def _set_environment_variables_from_config(self, storage_params: dict):
        """Set environment variables from config parameters so storage implementations can find them"""
        
        # Mapping of config keys to environment variable names
        config_to_env_mapping = {
            # MongoDB
            'mongo_uri': 'MONGO_URI',
            'mongo_database': 'MONGO_DATABASE', 
            'mongo_collection': 'MONGO_KG_COLLECTION',
            'mongo_vector_collection': 'MONGO_VECTOR_COLLECTION',
            'mongo_doc_status_collection': 'MONGO_DOC_STATUS_COLLECTION',
            
            # Redis
            'redis_url': 'REDIS_URI',
            'redis_host': 'REDIS_HOST',
            'redis_port': 'REDIS_PORT',
            'redis_password': 'REDIS_PASSWORD',
            
            
            # PostgreSQL
            'postgres_url': 'POSTGRES_URL',
            'postgres_host': 'POSTGRES_HOST',
            'postgres_port': 'POSTGRES_PORT',
            'postgres_db': 'POSTGRES_DATABASE',
            'postgres_user': 'POSTGRES_USER',
            'postgres_password': 'POSTGRES_PASSWORD',
            'postgres_table': 'POSTGRES_TABLE',
            
            # Qdrant
            'qdrant_url': 'QDRANT_URL',
            'qdrant_api_key': 'QDRANT_API_KEY',
            'qdrant_collection': 'QDRANT_COLLECTION',
            
            # Chroma
            'chroma_host': 'CHROMA_HOST',
            'chroma_port': 'CHROMA_PORT',
            'chroma_collection': 'CHROMA_COLLECTION',
            
            
            # Milvus
            'milvus_uri': 'MILVUS_URI',
            'milvus_user': 'MILVUS_USER',
            'milvus_password': 'MILVUS_PASSWORD',
            'milvus_db_name': 'MILVUS_DB_NAME',
            
            # FAISS
            'faiss_index_path': 'FAISS_INDEX_PATH',
            'faiss_metric_type': 'FAISS_METRIC_TYPE',
            
            # Neo4j
            'neo4j_url': 'NEO4J_URI',
            'neo4j_username': 'NEO4J_USERNAME',
            'neo4j_password': 'NEO4J_PASSWORD',
    
        }
        
        # Set environment variables from config
        for config_key, env_var in config_to_env_mapping.items():
            if config_key in storage_params and storage_params[config_key] is not None:
                # Convert to string and set environment variable
                env_value = str(storage_params[config_key])
                os.environ[env_var] = env_value
                logger.info(f"Set {env_var} = {env_value}")

    def _prepare_storage_params(self, storage_params: dict) -> dict:
        """Prepare storage parameters for LightRAG constructor"""
        
        # First, set environment variables from config
        self._set_environment_variables_from_config(storage_params)
        
        prepared_params = {}
        
        # Map storage types to LightRAG constructor parameters
        storage_mapping = {
            'kv_storage': 'kv_storage',
            'vector_storage': 'vector_storage', 
            'graph_storage': 'graph_storage',
            'doc_status_storage': 'doc_status_storage'
        }
        
        for config_key, lightrag_param in storage_mapping.items():
            if config_key in storage_params:
                prepared_params[lightrag_param] = storage_params[config_key]
                logger.info(f"Setting {lightrag_param} = {storage_params[config_key]}")
        
        # Add additional storage configuration parameters
        additional_storage_params = [
            'mongo_uri', 'mongo_database', 'mongo_collection',
            'redis_url', 'redis_host', 'redis_port', 'redis_password',
            'postgres_url', 'postgres_host', 'postgres_port', 'postgres_db', 'postgres_database',
            'postgres_user', 'postgres_password', 'postgres_table',
            'qdrant_url', 'qdrant_api_key', 'qdrant_collection',
            'chroma_host', 'chroma_port', 'chroma_collection',
            'milvus_uri', 'milvus_user', 'milvus_password', 'milvus_db_name',
            'faiss_index_path', 'faiss_metric_type',
            'neo4j_url', 'neo4j_username', 'neo4j_password'
        ]
        
        for param in additional_storage_params:
            if param in storage_params:
                prepared_params[param] = storage_params[param]
                logger.info(f"Setting additional storage param {param} = {storage_params[param]}")
        
        # Also check for any existing environment variables (these take precedence)
        env_mappings = {
            'MONGO_URI': 'mongo_uri',
            'MONGO_DATABASE': 'mongo_database', 
            'MONGO_KG_COLLECTION': 'mongo_collection',
            'MONGO_VECTOR_COLLECTION': 'mongo_vector_collection',
            'MONGO_DOC_STATUS_COLLECTION': 'mongo_doc_status_collection',
            'REDIS_URI': 'redis_url',
            'POSTGRES_URL': 'postgres_url',
            'POSTGRES_DATABASE': 'postgres_db',
            'QDRANT_URL': 'qdrant_url',
            'QDRANT_API_KEY': 'qdrant_api_key',
            'CHROMA_HOST': 'chroma_host',
            'CHROMA_PORT': 'chroma_port',
            'MILVUS_URI': 'milvus_uri',
            'MILVUS_HOST': 'milvus_host',
            'MILVUS_PORT': 'milvus_port',
            'MILVUS_USER': 'milvus_user',
            'MILVUS_PASSWORD': 'milvus_password',
            'MILVUS_DB_NAME': 'milvus_db_name',
            'NEO4J_URI': 'neo4j_url',
            'NEO4J_USERNAME': 'neo4j_username',
            'NEO4J_PASSWORD': 'neo4j_password'
        }
        
        # Check existing environment variables and add to prepared params
        for env_var, param_name in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value and param_name not in prepared_params:
                prepared_params[param_name] = env_value
                logger.info(f"Using existing environment variable {env_var} for {param_name}")
        
        return prepared_params

    def _reload_config(self):
        """Reload config from saved path"""
        if not hasattr(self.config, 'config_path') or not self.config.config_path:
            logger.warning("No config path available for reloading")
            return False
            
        try:
            new_config = initialize_config(self.config.config_path)
            new_config.config_path = self.config.config_path
            
            old_tenant_id = getattr(self.config, 'tenant_id', 'N/A')
            new_tenant_id = getattr(new_config, 'tenant_id', 'N/A')
            
            self.config = new_config
            self.model_manager = ModelManager(new_config)
            self.lamma_key = new_config.lamma_cloud_key
            
            # Clear cached model functions to force fresh initialization
            self._cached_llm_func = None
            self._cached_embedding_func = None
            self._cached_lightrag_params = None
            
            logger.info(f"Config reloaded from {self.config.config_path}")
            logger.info(f"Tenant ID: {old_tenant_id} -> {new_tenant_id}")
            logger.info("Model cache cleared")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to reload config: {e}")
            raise

    async def _get_model_functions(self, force_refresh: bool = False):
        """Get model functions with caching"""
        if force_refresh or self._cached_llm_func is None or self._cached_embedding_func is None:
            logger.info(f"Initializing models - LLM: {self.config.llm.get('binding', 'auto')}, "
                       f"Embedding: {self.config.embedding.get('binding', 'auto')}")
            
            self._cached_llm_func = await self.model_manager.get_llm_model_func()
            self._cached_embedding_func = await self.model_manager.get_embedding_func()
            self._cached_lightrag_params = self.model_manager.get_lightrag_params()
            
            logger.info("Model functions initialized")
        
        return self._cached_llm_func, self._cached_embedding_func, self._cached_lightrag_params

    async def _create_new_instance(
        self,
        tenant_id: str,
        kb_name: str,
        working_dir: str,
        screenshot_path: str,
        llm_model_func,
        embedding_func,
        storage_params: dict
    ) -> LightRAG_EXTENSIONS:
        """Create a new LightRAG instance"""
        embedding_dim = self.config.embedding.get('embedding_dim', 384)
        max_token_size = self.config.embedding.get('max_token_size', 8192)
        
        logger.info(f"Creating new LightRAG instance for tenant '{tenant_id}', KB '{kb_name}'")
        logger.info(f"Embedding config: dim={embedding_dim}, max_tokens={max_token_size}")
        
        prepared_storage_params = self._prepare_storage_params(storage_params)
        
        instance_params = {
            'namespace_prefix': self._make_namespace_prefix(tenant_id, kb_name),
            'working_dir': working_dir,
            'screenshot_path': screenshot_path,
            'llm_model_func': llm_model_func,
            'Lamma_Key': self.lamma_key,
            'embedding_func': EmbeddingFunc(
                embedding_dim=embedding_dim,
                max_token_size=max_token_size,
                func=embedding_func,
            ),
            'storage_params': prepared_storage_params
        }
        
        logger.info(f"Instance parameters: {list(instance_params.keys())}")
        logger.info(f"Storage parameters: {list(prepared_storage_params.keys())}")
        
        instance = LightRAG_EXTENSIONS(**instance_params)
        await instance.initialize_storages()
        await initialize_pipeline_status()
        
        logger.info(f"Successfully created LightRAG instance for tenant '{tenant_id}', KB '{kb_name}'")
        return instance

    async def create_instance(
        self,
        kb_name: Optional[str] = None,
        tenant_id: Optional[str] = None,
        reload_config: bool = True
    ) -> LightRAG_EXTENSIONS:
        """Create a new RAG instance (always creates fresh instance)
        
        Args:
            kb_name: Name of the knowledge base (uses default if None)
            tenant_id: Tenant ID (uses config default if None)
            reload_config: Whether to reload config before creating instance
        """
        # Reload config if requested
        if reload_config:
            self._reload_config()
        
        # Use defaults if not provided
        if kb_name is None:
            kb_name = self.config.default_kb
        if tenant_id is None:
            tenant_id = self.config.tenant_id
            
        logger.info(f"Creating new instance for tenant '{tenant_id}', KB '{kb_name}'")
        
        # Get fresh model functions
        llm_func, embedding_func, lightrag_params = await self._get_model_functions(force_refresh=reload_config)
        
        # Build paths
        working_dir, screenshot_path, storage_path = self._build_kb_paths(kb_name)
        
        # Merge storage parameters
        enhanced_storage_params = {
            **self.config.storage,
            **lightrag_params
        }
        
        logger.info(f"Storage config from config file: {self.config.storage}")
        logger.info(f"Enhanced storage parameters: {enhanced_storage_params}")
        
        # Create new instance
        instance = await self._create_new_instance(
            tenant_id=tenant_id,
            kb_name=kb_name,
            working_dir=working_dir,
            screenshot_path=screenshot_path,
            llm_model_func=llm_func,
            embedding_func=embedding_func,
            storage_params=enhanced_storage_params
        )
        
        # Store instance
        if tenant_id not in self._instances:
            self._instances[tenant_id] = {}
        self._instances[tenant_id][kb_name] = instance
        
        # Update reference if this is the default KB
        if kb_name == self.config.default_kb:
            self.rag_instance = instance
        
        # Update kb_list if needed
        if kb_name not in self.config.kb_list:
            self.config.kb_list.append(kb_name)
            if hasattr(self.config, 'save'):
                self.config.save()
            logger.info(f"Added KB '{kb_name}' to config")
        
        return instance

    async def get_instance(
        self,
        kb_name: Optional[str] = None,
        tenant_id: Optional[str] = None,
        create_if_missing: bool = True
    ) -> Optional[LightRAG_EXTENSIONS]:
        """Get existing instance or optionally create if missing
        
        Args:
            kb_name: Name of the knowledge base
            tenant_id: Tenant ID
            create_if_missing: Whether to create instance if it doesn't exist
        """
        if kb_name is None:
            kb_name = self.config.default_kb
        if tenant_id is None:
            tenant_id = self.config.tenant_id
            
        # Check if instance exists
        if (tenant_id in self._instances and 
            kb_name in self._instances[tenant_id]):
            logger.info(f"Found existing instance for tenant '{tenant_id}', KB '{kb_name}'")
            return self._instances[tenant_id][kb_name]
        
        # Create if missing and requested
        if create_if_missing:
            logger.info(f"Instance not found, creating new one for tenant '{tenant_id}', KB '{kb_name}'")
            return await self.create_instance(kb_name, tenant_id, reload_config=False)
        
        return None

    async def initialize(self):
        """Initialize RAG system with all configured KBs"""
        logger.info("Initializing RAG system")
        
        # Get model functions
        llm_func, embedding_func, lightrag_params = await self._get_model_functions()
        
        # Initialize all KBs from config
        initialized = set()
        for kb_name in set(self.config.kb_list + [self.config.default_kb]):
            if kb_name in initialized:
                continue
            initialized.add(kb_name)
            
            working_dir, screenshot_path, storage_path = self._build_kb_paths(kb_name)
            
            enhanced_storage_params = {
                **self.config.storage,
                **lightrag_params
            }
            
            instance = await self._create_new_instance(
                tenant_id=self.config.tenant_id,
                kb_name=kb_name,
                working_dir=working_dir,
                screenshot_path=screenshot_path,
                llm_model_func=llm_func,
                embedding_func=embedding_func,
                storage_params=enhanced_storage_params
            )
            
            # Store instance
            if self.config.tenant_id not in self._instances:
                self._instances[self.config.tenant_id] = {}
            self._instances[self.config.tenant_id][kb_name] = instance
            
            # Set default reference
            if kb_name == self.config.default_kb:
                self.rag_instance = instance
            
            logger.info(f"Initialized KB '{kb_name}' for tenant '{self.config.tenant_id}'")

    async def list_kbs(self, tenant_id: str = None) -> list[str]:
        """List all knowledge bases for a tenant"""
        if tenant_id is None:
            tenant_id = self.config.tenant_id
        return list(self._instances.get(tenant_id, {}).keys())

    def get_all_instances(self) -> dict:
        """Get all instances across all tenants"""
        return self._instances

    def clear_instances(self, tenant_id: str = None, kb_name: str = None):
        """Clear instances (useful for forcing recreation)
        
        Args:
            tenant_id: Clear specific tenant (all if None)
            kb_name: Clear specific KB within tenant (all if None)
        """
        if tenant_id is None:
            self._instances.clear()
            self.rag_instance = None
            logger.info("Cleared all instances")
        elif kb_name is None:
            if tenant_id in self._instances:
                del self._instances[tenant_id]
                logger.info(f"Cleared all instances for tenant '{tenant_id}'")
        else:
            if tenant_id in self._instances and kb_name in self._instances[tenant_id]:
                del self._instances[tenant_id][kb_name]
                logger.info(f"Cleared instance for tenant '{tenant_id}', KB '{kb_name}'")
                
                # Clear default reference if needed
                if (kb_name == self.config.default_kb and 
                    tenant_id == self.config.tenant_id):
                    self.rag_instance = None

    async def get_model_info(self):
        """Get information about currently configured models"""
        return {
            "llm": {
                "binding": self.config.llm.get("binding", "auto-detected"),
                "model": self.config.llm.get("model", "auto-detected"),
                "max_tokens": self.config.llm.get("max_tokens", "default")
            },
            "embedding": {
                "binding": self.config.embedding.get("binding", "auto-detected"),
                "model": self.config.embedding.get("model_name", "auto-detected"),
                "embedding_dim": self.config.embedding.get("embedding_dim", 384),
                "max_token_size": self.config.embedding.get("max_token_size", 8192)
            }
        }

    async def test_model_connections(self):
        """Test if model connections are working properly"""
        try:
            llm_func, embedding_func, _ = await self._get_model_functions(force_refresh=True)
            
            # Test LLM
            try:
                await llm_func("Test prompt", system_prompt="You are a helpful assistant.")
                llm_status = "✓ Working"
            except Exception as e:
                llm_status = f"✗ Error: {str(e)[:100]}"
            
            # Test embedding
            try:
                embedding_response = await embedding_func(["test text"])
                embedding_dim = self.config.embedding.get('embedding_dim', 384)
                actual_dim = embedding_response.shape[1] if hasattr(embedding_response, 'shape') else 'unknown'
                embedding_status = f"✓ Working (configured: {embedding_dim}, actual: {actual_dim})"
            except Exception as e:
                embedding_status = f"✗ Error: {str(e)[:100]}"
            
            return {
                "llm_status": llm_status,
                "embedding_status": embedding_status,
                "model_info": await self.get_model_info()
            }
            
        except Exception as e:
            return {
                "error": f"Failed to test model connections: {e}",
                "model_info": await self.get_model_info()
            }

    def _build_kb_paths(self, kb_name: str):
        """Build tenant+KB-specific paths"""
        working_dir = str(Path(self.config.paths["working_dir"]) / kb_name)
        screenshot_path = str(Path(self.config.paths["screenshot_path"]) / kb_name)
        storage_path = str(Path(self.config.paths["storage_path"]) / kb_name)

        os.makedirs(working_dir, exist_ok=True)
        os.makedirs(screenshot_path, exist_ok=True)
        os.makedirs(storage_path, exist_ok=True)

        return working_dir, screenshot_path, storage_path

    # Backward compatibility methods
    async def get_rag_instance(self, kb_name: Optional[str] = None, force_reload: bool = False):
        """Get RAG instance for specific KB (backward compatibility)
        
        Args:
            kb_name: Name of the knowledge base
            force_reload: If True, reload config and create new instance
        """
        if force_reload:
            # Clear existing instance to force recreation
            self.clear_instances(kb_name=kb_name)
            return await self.create_instance(kb_name, reload_config=True)
        else:
            return await self.get_instance(kb_name, create_if_missing=True)

    async def get_or_create_rag_instance(self, kb_name: Optional[str] = None, force_reload: bool = True):
        """Get or create RAG instance for a KB (backward compatibility)
        
        Args:
            kb_name: Name of the knowledge base
            force_reload: If True, always create new instance
        """
        if force_reload:
            return await self.create_instance(kb_name, reload_config=True)
        else:
            return await self.get_instance(kb_name, create_if_missing=True)


# Simplified convenience functions
async def create_rag_manager(config_path: str = None):
    """Create RAG manager from config file"""
    if config_path:
        manager = RAGManager.from_config_file(config_path)
    else:
        from extensions.app.config import Config
        config = Config()
        manager = RAGManager(config)
    
    await manager.initialize()
    return manager

async def create_new_rag_instance(kb_name: str = None, config_path: str = None):
    """Create a new RAG instance with fresh config"""
    manager = await create_rag_manager(config_path)
    return await manager.create_instance(kb_name, reload_config=True)