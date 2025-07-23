import json
import os
import shutil
import logging
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from fastmcp import FastMCP
try:
    from fastmcp import image_content
except ImportError:
    # Fallback if image_content is not available
    image_content = None
from extensions.app.config import Config
from extensions.app.rag_manager import RAGManager

try:
    from lightrag.base import QueryParam
except ImportError as e:
    print(f"Warning: Could not import QueryParam: {e}")
    QueryParam = None

logger = logging.getLogger(__name__)

# Supported document formats
SUPPORTED_FORMATS = {
    # Base types
    'pdf',
    # Documents and presentations
    '602', 'abw', 'cgm', 'cwk', 'doc', 'docx', 'docm', 'dot', 'dotm', 
    'hwp', 'key', 'lwp', 'mw', 'mcw', 'pages', 'pbd', 'ppt', 'pptm', 
    'pptx', 'pot', 'potm', 'potx', 'rtf', 'sda', 'sdd', 'sdp', 'sdw', 
    'sgl', 'sti', 'sxi', 'sxw', 'stw', 'sxg', 'txt', 'uof', 'uop', 
    'uot', 'vor', 'wpd', 'wps', 'xml', 'zabw', 'epub'
}

# Supported image formats
SUPPORTED_IMAGE_FORMATS = {
    'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'tif', 'webp', 'svg'
}

# MIME type mapping for images
IMAGE_MIME_TYPES = {
    'jpg': 'image/jpeg',
    'jpeg': 'image/jpeg',
    'png': 'image/png',
    'gif': 'image/gif',
    'bmp': 'image/bmp',
    'tiff': 'image/tiff',
    'tif': 'image/tiff',
    'webp': 'image/webp',
    'svg': 'image/svg+xml'
}

class MCPTools:
    """MCP tool implementations"""
    
    def __init__(self, mcp_instance: FastMCP, config: Config, rag_manager: RAGManager):
        self.mcp = mcp_instance
        self.config = config
        self.rag_manager = rag_manager
        self._register_tools()
    
    def _register_tools(self):
        """Register all MCP tools"""
        self.mcp.tool()(self.ingest_document)
        self.mcp.tool()(self.search_documents)
        self.mcp.tool()(self.list_knowledge_bases)
        self.mcp.tool()(self.create_knowledge_base)
        self.mcp.tool()(self.search_all_knowledge_bases)
        self.mcp.tool()(self.ingest_directory)
        self.mcp.tool()(self.get_image)
    
    async def ingest_document(
        self,
        file_path: str,
        kb_name: Optional[str] = None,
        doc_id: Optional[str] = None
    ) -> str:
        """
        Ingest a document into the knowledge base
        
        Args:
            file_path: Path to the document file to ingest
            kb_name: Knowledge base name (optional, uses default if not provided)
            doc_id: Optional document ID for tracking
        """
        kb_name = kb_name or self.config.default_kb
        
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"
        
        try:
            rag = await self.rag_manager.get_or_create_rag_instance(kb_name)
            
            # Check file size
            file_size = os.path.getsize(file_path)
            max_size = 100 * 1024 * 1024  # 100MB default
            if file_size > max_size:
                return f"File size ({file_size} bytes) exceeds maximum ({max_size} bytes)"
            
            # Ingest the document
            await rag.ainsert(
                file_paths=[file_path],
                ids=[doc_id] if doc_id else None,
            )
            
            # Copy to storage
            filename = Path(file_path).name
            storage_path = os.path.join(self.config.paths["storage_path"], filename)
            shutil.copy2(file_path, storage_path)
            
            result = {
                "status": "success",
                "message": f"Document {filename} ingested successfully into {kb_name}",
                "doc_id": doc_id or filename,
                "file_path": storage_path,
                "file_size": file_size,
                "kb_name": kb_name,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"[{self.config.tenant_id}/{kb_name}] Document ingested: {filename}")
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_msg = f"Ingestion failed: {str(e)}"
            logger.error(error_msg)
            return error_msg

    async def search_documents(
        self,
        query: str,
        kb_name: str,
        mode: str = "hybrid",
        top_k: int = 10
    ) -> str:
        """
        Search documents in the knowledge base
        
        Args:
            query: Search query text
            kb_name: Knowledge base name
            mode: Search mode (naive, local, global, hybrid, mix)
            top_k: Number of top results to return (1-100)
        """
        kb_name = kb_name or self.config.default_kb
        
        if not query.strip():
            return "Query cannot be empty"
        
        try:
            rag = await self.rag_manager.get_rag_instance(kb_name)
            if rag is None:
                return f"No such Knowledge Base with name {kb_name} exists"
            
            if QueryParam is None:
                return "QueryParam not available - check LightRAG installation"
            
            param = QueryParam(mode=mode, top_k=top_k, only_need_context=True)
            context = await rag.aretrieve_context(query, param)
            
            chunks = []
            if isinstance(context, str):
                for chunk_text in context.split('\n---\n'):
                    if chunk_text.strip():
                        chunks.append({
                            'content': chunk_text.strip(),
                        })
            elif isinstance(context, list):
                for item in context:
                    if isinstance(item, dict):
                        chunks.append(item)
                    else:
                        chunks.append({
                            'content': str(item),
                        })
            
            result = {
                "query": query,
                "kb_name": kb_name,
                "search_mode": mode,
                "total_chunks": len(chunks),
                "chunks": chunks,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"[{self.config.tenant_id}/{kb_name}] Search successful. Query: '{query}', Chunks: {len(chunks)}")
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            logger.error(error_msg)
            return error_msg

    async def list_knowledge_bases(self) -> str:
        """List all available knowledge bases for the tenant"""
        try:
            kbs = await self.rag_manager.list_kbs()
            
            result = {
                "tenant_id": self.config.tenant_id,
                "knowledge_bases": kbs,
                "total_count": len(kbs),
                "timestamp": datetime.now().isoformat()
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_msg = f"Failed to list knowledge bases: {str(e)}"
            logger.error(error_msg)
            return error_msg

    async def create_knowledge_base(self, kb_name: str) -> str:
        """
        Create a new knowledge base
        
        Args:
            kb_name: Name for the new knowledge base
        """
        try:
            rag = await self.rag_manager.get_or_create_rag_instance(kb_name)
            
            result = {
                "status": "success",
                "message": f"Knowledge base '{kb_name}' created successfully",
                "kb_name": kb_name,
                "tenant_id": self.config.tenant_id,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"[{self.config.tenant_id}] Created knowledge base: {kb_name}")
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_msg = f"Failed to create knowledge base: {str(e)}"
            logger.error(error_msg)
            return error_msg

    async def search_all_knowledge_bases(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 10
    ) -> str:
        """
        Search across all knowledge bases when no specific KB is provided
        
        Args:
            query: Search query text
            mode: Search mode (naive, local, global, hybrid, mix)
            top_k: Number of top results to return per KB (1-100)
        """
        if not query.strip():
            return "Query cannot be empty"
        
        try:
            # First get list of all knowledge bases
            kbs = await self.rag_manager.list_kbs()
            
            if not kbs:
                return "No knowledge bases found"
            
            all_results = []
            
            # Search each knowledge base
            for kb_name in kbs:
                try:
                    rag = await self.rag_manager.get_rag_instance(kb_name)
                    if rag is None:
                        logger.warning(f"Could not access knowledge base: {kb_name}")
                        continue
                    
                    if QueryParam is None:
                        return "QueryParam not available - check LightRAG installation"
                    
                    param = QueryParam(mode=mode, top_k=top_k, only_need_context=True)
                    context = await rag.aretrieve_context(query, param)
                    
                    chunks = []
                    if isinstance(context, str):
                        for chunk_text in context.split('\n---\n'):
                            if chunk_text.strip():
                                chunks.append({
                                    'content': chunk_text.strip(),
                                    'kb_name': kb_name
                                })
                    elif isinstance(context, list):
                        for item in context:
                            if isinstance(item, dict):
                                item['kb_name'] = kb_name
                                chunks.append(item)
                            else:
                                chunks.append({
                                    'content': str(item),
                                    'kb_name': kb_name
                                })
                    
                    if chunks:
                        kb_result = {
                            "kb_name": kb_name,
                            "chunk_count": len(chunks),
                            "chunks": chunks
                        }
                        all_results.append(kb_result)
                        logger.info(f"[{self.config.tenant_id}/{kb_name}] Found {len(chunks)} chunks for query: '{query}'")
                    
                except Exception as e:
                    logger.error(f"Error searching KB {kb_name}: {str(e)}")
                    continue
            
            result = {
                "query": query,
                "search_mode": mode,
                "searched_kbs": len(kbs),
                "kbs_with_results": len(all_results),
                "total_chunks": sum(kb_result["chunk_count"] for kb_result in all_results),
                "results": all_results,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"[{self.config.tenant_id}] Search all KBs completed. Query: '{query}', Total chunks: {result['total_chunks']}")
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_msg = f"Search all knowledge bases failed: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def _is_supported_format(self, file_path: str) -> bool:
        """Check if file format is supported for ingestion"""
        file_extension = Path(file_path).suffix.lower().lstrip('.')
        return file_extension in SUPPORTED_FORMATS

    def _get_supported_files(self, directory: str) -> List[str]:
        """Get all supported files from directory"""
        supported_files = []
        
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    if self._is_supported_format(file_path):
                        supported_files.append(file_path)
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {str(e)}")
        
        return supported_files

    async def ingest_directory(
        self,
        directory_path: str,
        kb_name: Optional[str] = None,
        recursive: bool = True
    ) -> str:
        """
        Ingest all supported documents from a directory
        
        Args:
            directory_path: Path to the directory containing documents
            kb_name: Knowledge base name (optional, uses default if not provided)
            recursive: Whether to search subdirectories (default: True)
        """
        kb_name = kb_name or self.config.default_kb
        
        if not os.path.exists(directory_path):
            return f"Directory not found: {directory_path}"
        
        if not os.path.isdir(directory_path):
            return f"Path is not a directory: {directory_path}"
        
        try:
            # Get all supported files
            if recursive:
                supported_files = self._get_supported_files(directory_path)
            else:
                supported_files = []
                for file in os.listdir(directory_path):
                    file_path = os.path.join(directory_path, file)
                    if os.path.isfile(file_path) and self._is_supported_format(file_path):
                        supported_files.append(file_path)
            
            if not supported_files:
                return f"No supported document formats found in directory: {directory_path}"
            
            # Get RAG instance
            rag = await self.rag_manager.get_or_create_rag_instance(kb_name)
            
            # Results tracking
            successful_ingestions = []
            failed_ingestions = []
            total_size = 0
            max_size = 100 * 1024 * 1024  # 100MB per file
            
            # Process each file
            for file_path in supported_files:
                try:
                    # Check file size
                    file_size = os.path.getsize(file_path)
                    if file_size > max_size:
                        failed_ingestions.append({
                            "file": file_path,
                            "error": f"File size ({file_size} bytes) exceeds maximum ({max_size} bytes)"
                        })
                        continue
                    
                    # Ingest the document
                    filename = Path(file_path).name
                    await rag.ainsert(file_paths=[file_path], ids=[filename])
                    
                    # Copy to storage
                    storage_path = os.path.join(self.config.paths["storage_path"], filename)
                    # Handle duplicate filenames by adding a counter
                    counter = 1
                    original_storage_path = storage_path
                    while os.path.exists(storage_path):
                        name, ext = os.path.splitext(original_storage_path)
                        storage_path = f"{name}_{counter}{ext}"
                        counter += 1
                    
                    shutil.copy2(file_path, storage_path)
                    
                    successful_ingestions.append({
                        "file": file_path,
                        "filename": filename,
                        "storage_path": storage_path,
                        "size": file_size
                    })
                    total_size += file_size
                    
                    logger.info(f"[{self.config.tenant_id}/{kb_name}] Ingested: {filename}")
                    
                except Exception as e:
                    failed_ingestions.append({
                        "file": file_path,
                        "error": str(e)
                    })
                    logger.error(f"Failed to ingest {file_path}: {str(e)}")
            
            result = {
                "status": "completed",
                "directory": directory_path,
                "kb_name": kb_name,
                "recursive": recursive,
                "total_files_found": len(supported_files),
                "successful_ingestions": len(successful_ingestions),
                "failed_ingestions": len(failed_ingestions),
                "total_size_ingested": total_size,
                "successful_files": successful_ingestions,
                "failed_files": failed_ingestions,
                "supported_formats": sorted(list(SUPPORTED_FORMATS)),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"[{self.config.tenant_id}/{kb_name}] Directory ingestion completed. Success: {len(successful_ingestions)}, Failed: {len(failed_ingestions)}")
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_msg = f"Directory ingestion failed: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def _get_image_mime_type(self, file_path: str) -> str:
        """Get MIME type for image file"""
        file_extension = Path(file_path).suffix.lower().lstrip('.')
        return IMAGE_MIME_TYPES.get(file_extension, 'application/octet-stream')

    def _is_supported_image_format(self, file_path: str) -> bool:
        """Check if file format is supported for image serving"""
        file_extension = Path(file_path).suffix.lower().lstrip('.')
        return file_extension in SUPPORTED_IMAGE_FORMATS

    async def get_image(self, image_path: str) -> dict:
        """
        Return an image from local storage as base64 encoded string
        
        Args:
            image_path: Path to the image file (relative to storage or absolute path)
        
        Returns:
            Dictionary with image content for MCP response
        """
        try:
            # Handle relative paths by checking storage directory first
            if not os.path.isabs(image_path):
                # Try storage directory first
                storage_image_path = os.path.join(self.config.paths["storage_path"], image_path)
                if os.path.exists(storage_image_path):
                    image_path = storage_image_path
            
            # Check if file exists
            if not os.path.exists(image_path):
                error_msg = f"Image file not found: {image_path}"
                logger.error(error_msg)
                return {"error": error_msg}
            
            # Check if file is a supported image format
            if not self._is_supported_image_format(image_path):
                error_msg = f"Unsupported image format: {image_path}. Supported formats: {', '.join(sorted(SUPPORTED_IMAGE_FORMATS))}"
                logger.error(error_msg)
                return {"error": error_msg}
            
            # Check file size (limit to 10MB for images)
            file_size = os.path.getsize(image_path)
            max_size = 10 * 1024 * 1024  # 10MB
            if file_size > max_size:
                error_msg = f"Image file too large ({file_size} bytes). Maximum size: {max_size} bytes"
                logger.error(error_msg)
                return {"error": error_msg}
            
            # Log success
            filename = Path(image_path).name
            logger.info(f"[{self.config.tenant_id}] Image served: {filename} ({file_size} bytes)")
            
            # Use FastMCP's image_content helper if available
            if image_content is not None:
                return image_content(path=image_path)
            
            # Fallback to manual base64 encoding
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
                base64_data = base64.b64encode(image_data).decode('utf-8')
            
            # Get MIME type
            mime_type = self._get_image_mime_type(image_path)
            
            return {
                        "type": "image",
                        "data": base64_data,
                        "mimeType": mime_type
            }
            
        except Exception as e:
            error_msg = f"Failed to serve image: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}