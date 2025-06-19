from __future__ import annotations

import traceback
import asyncio
import configparser
import os
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import partial
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Iterator,
    cast,
    final,
    Literal,
    Optional,
    List,
    Dict,
)
from lightrag.constants import (
    DEFAULT_MAX_TOKEN_SUMMARY,
    DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE,
)
from lightrag.utils import get_env_value

from lightrag.kg import (
    STORAGES,
    verify_storage_implementation,
)

from lightrag.kg.shared_storage import (
    get_namespace_data,
    get_pipeline_status_lock,
)

from lightrag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
    QueryParam,
    StorageNameSpace,
    StoragesStatus,
)
from lightrag.namespace import NameSpace, make_namespace
from extensions.operate_extension import (
    chunking_by_token_size,
    extract_entities,
    merge_nodes_and_edges,
    kg_query,
    naive_query,
    query_with_keywords,
    kg_query_chunks_only,
    kg_query_generate_response,
    naive_query_chunks_only,
    naive_query_generate_response,
)
from lightrag.prompt import GRAPH_FIELD_SEP
from lightrag.utils import (
    Tokenizer,
    TiktokenTokenizer,
    EmbeddingFunc,
    always_get_an_event_loop,
    compute_mdhash_id,
    convert_response_to_json,
    lazy_external_import,
    priority_limit_async_func_call,
    get_content_summary,
    clean_text,
    check_storage_env_vars,
    logger,
)
from lightrag.types import KnowledgeGraph
from lightrag.lightrag import LightRAG
from dotenv import load_dotenv

from llama_cloud_services import LlamaParse

class LightRAG_EXTENSIONS(LightRAG):
    def __init__(self, *args, screenshot_path=None, **kwargs):
        self.screenshot_dir = screenshot_path

        # Call the original LightRAG initializer with the rest of the args
        super().__init__(*args, **kwargs)
        self.entities_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.VECTOR_STORE_ENTITIES
            ),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name", "source_id", "content", "file_path", "screenshot_path"},
        )
        self.relationships_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.VECTOR_STORE_RELATIONSHIPS
            ),
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id", "source_id", "content", "file_path", "screenshot_path"},
        )
        self.chunks_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.VECTOR_STORE_CHUNKS
            ),
            embedding_func=self.embedding_func,
            meta_fields={"full_doc_id", "content", "file_path", "screenshot_path"},
        )

        load_dotenv()
        lamma_cloud = os.getenv("LAMMA_CLOUD")

        # Initialize llama_parse 
        self.parser = LlamaParse(
            api_key=lamma_cloud,
            result_type="markdown",  # or "text" depending on your needs
            take_screenshot=True, 
        )

    def insert(
        self,
        file_paths: str | list[str],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        ids: str | list[str] | None = None,
    ) -> None:
        """Sync Insert documents with checkpoint support

        Args:
            split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
            chunk_token_size, it will be split again by token size.
            split_by_character_only: if split_by_character_only is True, split the string by character only, when
            split_by_character is None, this parameter is ignored.
            ids: single string of the document ID or list of unique document IDs, if not provided, MD5 hash IDs will be generated
            file_paths: single string of the file path or list of file paths, used for citation
        """
        loop = always_get_an_event_loop()
        loop.run_until_complete(
            self.ainsert(
               file_paths,  split_by_character, split_by_character_only, ids
            )
        )

    async def ainsert(
        self,
        file_paths: str | list[str],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        ids: str | list[str] | None = None,
    ) -> None:
        """Async Insert documents with checkpoint support

        Args:
            split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
            chunk_token_size, it will be split again by token size.
            split_by_character_only: if split_by_character_only is True, split the string by character only, when
            split_by_character is None, this parameter is ignored.
            ids: list of unique document IDs, if not provided, MD5 hash IDs will be generated
            file_paths: list of file paths corresponding to each document, used for citation
        """
        await self.apipeline_enqueue_documents(file_paths, ids)
        await self.apipeline_process_enqueue_documents(
            file_paths, split_by_character, split_by_character_only
        )

    async def apipeline_enqueue_documents(
        self,
        file_paths: str | list[str],
        ids: list[str] | None = None,
    ) -> None:
        """
        Pipeline for Processing Documents

        1. Validate ids if provided or generate MD5 hash IDs
        2. Remove duplicate contents
        3. Generate document initial status
        4. Filter out already processed documents
        5. Enqueue document in status

        Args:
            file_paths: list of file paths corresponding to each document, used for citation
            ids: list of unique document IDs, if not provided, MD5 hash IDs will be generated
        """

        if isinstance(ids, str):
            ids = [ids]
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        # 1. Validate ids if provided or generate MD5 hash IDs
        if ids is not None:
            # Check if IDs are unique
            if len(ids) != len(set(ids)):
                raise ValueError("IDs must be unique")

            # Generate contents dict of IDs provided by user and documents
            contents = {
                id_: {"file_path": path}
                for id_, path in zip(ids, file_paths)
            }
        else:
            # Clean input text and remove duplicates
            cleaned_input = [
                path for path in file_paths
            ]
            unique_content_with_paths = []

            # Keep track of unique content and their paths
            for path in cleaned_input:
                if path not in unique_content_with_paths:
                    unique_content_with_paths.append(path)

            # Generate contents dict of MD5 hash IDs and documents with paths
            contents = {
                compute_mdhash_id(path, prefix="doc-"): {
                    "file_path": path,
                }
                for path in unique_content_with_paths
            }

        # 2. Remove duplicate contents
        unique_paths = {}
        for id_, content_data in contents.items():
            file_path = content_data["file_path"]
            if file_path not in unique_paths:
                unique_paths[file_path] = (id_)

        # Reconstruct contents with unique content
        contents = {
            id_: {"file_path": file_path}
            for  file_path, id_ in unique_paths.items()
        }

        # 3. Generate document initial status
        new_docs: dict[str, Any] = {
            id_: {
                "status": DocStatus.PENDING,
                "content": content_data["file_path"],
                "content_summary": get_content_summary(content_data["file_path"]),
                "content_length": len(content_data["file_path"]),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "file_path": content_data[
                    "file_path"
                ],  # Store file path in document status
            }
            for id_, content_data in contents.items()
        }

        # 4. Filter out already processed documents
        # Get docs ids
        all_new_doc_ids = set(new_docs.keys())
        # Exclude IDs of documents that are already in progress
        unique_new_doc_ids = await self.doc_status.filter_keys(all_new_doc_ids)

        # Log ignored document IDs
        ignored_ids = [
            doc_id for doc_id in unique_new_doc_ids if doc_id not in new_docs
        ]
        if ignored_ids:
            logger.warning(
                f"Ignoring {len(ignored_ids)} document IDs not found in new_docs"
            )
            for doc_id in ignored_ids:
                logger.warning(f"Ignored document ID: {doc_id}")

        # Filter new_docs to only include documents with unique IDs
        new_docs = {
            doc_id: new_docs[doc_id]
            for doc_id in unique_new_doc_ids
            if doc_id in new_docs
        }

        if not new_docs:
            logger.info("No new unique documents were found.")
            return

        # 5. Store status document
        await self.doc_status.upsert(new_docs)
        logger.info(f"Stored {len(new_docs)} new unique documents")


    async def apipeline_process_enqueue_documents(
        self,
        file_paths: list,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
    ) -> None:
        """
        Process pending documents using llama_parse for chunking while maintaining screenshot_paths.
        
        1. Get all pending, failed, and abnormally terminated processing documents.
        2. Use llama_parse to split document content into chunks with screenshots
        3. Process each chunk for entity and relation extraction
        4. Update the document status
        """

        # Get pipeline status shared data and lock
        pipeline_status = await get_namespace_data("pipeline_status")
        pipeline_status_lock = get_pipeline_status_lock()

        # Check if another process is already processing the queue
        async with pipeline_status_lock:
            if not pipeline_status.get("busy", False):
                processing_docs, failed_docs, pending_docs = await asyncio.gather(
                    self.doc_status.get_docs_by_status(DocStatus.PROCESSING),
                    self.doc_status.get_docs_by_status(DocStatus.FAILED),
                    self.doc_status.get_docs_by_status(DocStatus.PENDING),
                )

                to_process_docs: dict[str, DocProcessingStatus] = {}
                to_process_docs.update(processing_docs)
                to_process_docs.update(failed_docs)
                to_process_docs.update(pending_docs)

                if not to_process_docs:
                    logger.info("No documents to process")
                    return

                pipeline_status.update({
                    "busy": True,
                    "job_name": "Default Job",
                    "job_start": datetime.now(timezone.utc).isoformat(),
                    "docs": 0,
                    "batchs": 0,
                    "cur_batch": 0,
                    "request_pending": False,
                    "latest_message": "",
                })
                del pipeline_status["history_messages"][:]
            else:
                pipeline_status["request_pending"] = True
                logger.info("Another process is already processing the document queue. Request queued.")
                return

        try:
            while True:
                if not to_process_docs:
                    log_message = "All documents have been processed or are duplicates"
                    logger.info(log_message)
                    pipeline_status["latest_message"] = log_message
                    pipeline_status["history_messages"].append(log_message)
                    break

                log_message = f"Processing {len(to_process_docs)} document(s)"
                logger.info(log_message)

                pipeline_status["docs"] = len(to_process_docs)
                pipeline_status["batchs"] = len(to_process_docs)
                pipeline_status["cur_batch"] = 0
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

                first_doc_id, first_doc = next(iter(to_process_docs.items()))
                first_doc_path = first_doc.file_path
                path_prefix = first_doc_path[:20] + ("..." if len(first_doc_path) > 20 else "")
                total_files = len(to_process_docs)
                job_name = f"{path_prefix}[{total_files} files]"
                pipeline_status["job_name"] = job_name

                processed_count = 0
                semaphore = asyncio.Semaphore(self.max_parallel_insert)

                async def process_document(
                    doc_id: str,
                    status_doc: DocProcessingStatus,
                    pipeline_status: dict,
                    pipeline_status_lock: asyncio.Lock,
                    semaphore: asyncio.Semaphore,
                ) -> None:
                    """Process single document using llama_parse"""
                    file_extraction_stage_ok = False
                    async with semaphore:
                        nonlocal processed_count
                        current_file_number = 0
                        try:
                            file_path = getattr(status_doc, "file_path", "unknown_source")

                            async with pipeline_status_lock:
                                processed_count += 1
                                current_file_number = processed_count
                                pipeline_status["cur_batch"] = processed_count

                                log_message = f"Extracting stage {current_file_number}/{total_files}: {file_path}"
                                logger.info(log_message)
                                pipeline_status["history_messages"].append(log_message)
                                log_message = f"Processing d-id: {doc_id}"
                                logger.info(log_message)
                                pipeline_status["latest_message"] = log_message
                                pipeline_status["history_messages"].append(log_message)

                            # Use llama_parse to process the document and get chunks with screenshots
                            chunks = await self._process_with_llama_parse(
                                file_paths, 
                                status_doc.content,
                                doc_id
                            )
                            
                            # Process document chunks
                            doc_status_task = asyncio.create_task(
                                self.doc_status.upsert({
                                    doc_id: {
                                        "status": DocStatus.PROCESSING,
                                        "chunks_count": len(chunks),
                                        "created_at": status_doc.created_at,
                                        "updated_at": datetime.now(timezone.utc).isoformat(),
                                        "file_path": file_path,
                                    }
                                })
                            )
                            
                            chunks_vdb_task = asyncio.create_task(self.chunks_vdb.upsert(chunks))

                            entity_relation_task = asyncio.create_task(
                                self._process_entity_relation_graph(
                                    chunks, pipeline_status, pipeline_status_lock
                                )
                            )
                            
                            full_docs_task = asyncio.create_task(
                                self.full_docs.upsert({doc_id: {"content": status_doc.content}})
                            )
                            text_chunks_task = asyncio.create_task(self.text_chunks.upsert(chunks))
                            
                            tasks = [doc_status_task, chunks_vdb_task, entity_relation_task, 
                                    full_docs_task, text_chunks_task]
                            await asyncio.gather(*tasks)
                            file_extraction_stage_ok = True

                        except Exception as e:
                            logger.error(traceback.format_exc())
                            error_msg = f"Failed to extract document {current_file_number}/{total_files}: {file_path}"
                            logger.error(error_msg)
                            async with pipeline_status_lock:
                                pipeline_status["latest_message"] = error_msg
                                pipeline_status["history_messages"].append(traceback.format_exc())
                                pipeline_status["history_messages"].append(error_msg)

                                for task in [chunks_vdb_task, entity_relation_task, 
                                        full_docs_task, text_chunks_task]:
                                    if not task.done():
                                        task.cancel()

                            if self.llm_response_cache:
                                await self.llm_response_cache.index_done_callback()

                            await self.doc_status.upsert({
                                doc_id: {
                                    "status": DocStatus.FAILED,
                                    "error": str(e),
                                    "content": status_doc.content,
                                    "created_at": status_doc.created_at,
                                    "updated_at": datetime.now(timezone.utc).isoformat(),
                                    "file_path": file_path,
                                }
                            })

                    if file_extraction_stage_ok:
                        try:
                            chunk_results = await entity_relation_task
                            await merge_nodes_and_edges(
                                chunk_results=chunk_results,
                                knowledge_graph_inst=self.chunk_entity_relation_graph,
                                entity_vdb=self.entities_vdb,
                                relationships_vdb=self.relationships_vdb,
                                global_config=asdict(self),
                                pipeline_status=pipeline_status,
                                pipeline_status_lock=pipeline_status_lock,
                                llm_response_cache=self.llm_response_cache,
                                current_file_number=current_file_number,
                                total_files=total_files,
                                file_path=file_path,
                            )
                            
                            await self.doc_status.upsert({
                                doc_id: {
                                    "status": DocStatus.PROCESSED,
                                    "chunks_count": len(chunks),
                                    "created_at": status_doc.created_at,
                                    "updated_at": datetime.now(timezone.utc).isoformat(),
                                    "file_path": file_path,
                                }
                            })

                            await self._insert_done()

                            async with pipeline_status_lock:
                                log_message = f"Completed processing file {current_file_number}/{total_files}: {file_path}"
                                logger.info(log_message)
                                pipeline_status["latest_message"] = log_message
                                pipeline_status["history_messages"].append(log_message)

                        except Exception as e:
                            logger.error(traceback.format_exc())
                            error_msg = f"Merging stage failed in document {current_file_number}/{total_files}: {file_path}"
                            logger.error(error_msg)
                            async with pipeline_status_lock:
                                pipeline_status["latest_message"] = error_msg
                                pipeline_status["history_messages"].append(traceback.format_exc())
                                pipeline_status["history_messages"].append(error_msg)

                            if self.llm_response_cache:
                                await self.llm_response_cache.index_done_callback()

                            await self.doc_status.upsert({
                                doc_id: {
                                    "status": DocStatus.FAILED,
                                    "error": str(e),
                                    "created_at": status_doc.created_at,
                                    "updated_at": datetime.now().isoformat(),
                                    "file_path": file_path,
                                }
                            })

                # Create processing tasks for all documents
                doc_tasks = []
                for doc_id, status_doc in to_process_docs.items():
                    doc_tasks.append(
                        process_document(
                            doc_id,
                            status_doc,
                            pipeline_status,
                            pipeline_status_lock,
                            semaphore,
                        )
                    )

                await asyncio.gather(*doc_tasks)

                has_pending_request = False
                async with pipeline_status_lock:
                    has_pending_request = pipeline_status.get("request_pending", False)
                    if has_pending_request:
                        pipeline_status["request_pending"] = False

                if not has_pending_request:
                    break

                log_message = "Processing additional documents due to pending request"
                logger.info(log_message)
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

                processing_docs, failed_docs, pending_docs = await asyncio.gather(
                    self.doc_status.get_docs_by_status(DocStatus.PROCESSING),
                    self.doc_status.get_docs_by_status(DocStatus.FAILED),
                    self.doc_status.get_docs_by_status(DocStatus.PENDING),
                )

                to_process_docs = {}
                to_process_docs.update(processing_docs)
                to_process_docs.update(failed_docs)
                to_process_docs.update(pending_docs)

        finally:
            log_message = "Document processing pipeline completed"
            logger.info(log_message)
            async with pipeline_status_lock:
                pipeline_status["busy"] = False
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)


    async def _process_with_llama_parse(self, file_paths: str, content: str, doc_id: str) -> dict[str, Any]:
        """
        Process document using llama_parse to get chunks with screenshots.
        
        Args:
            file_path: Path to the document file
            content: Document content
            doc_id: Document ID
        
        Returns:
            Dictionary of chunks with metadata including screenshots
        """
        try:
            
            # Parse the document
            parsed_result = await self.parser.aparse(file_paths)

            chunks = {}
            chunk_size = -1

            if isinstance(parsed_result, list):
                for parsed_doc in parsed_result:
                    await self.process_parsed_document(parsed_doc, chunk_size, chunks)
            else:
                await self.process_parsed_document(parsed_result, chunk_size, chunks)
                            
            return chunks
        
        except Exception as e:
            logger.error(f"Failed to process document with llama_parse: {str(e)}")
            raise


    async def process_parsed_document(self, parsed_doc, chunk_size, chunks):
        file_path = getattr(parsed_doc, "file_name", "unknown_file")
        doc_id = os.path.splitext(os.path.basename(file_path))[0]

        screenshot_dir = f"{self.screenshot_dir}/{doc_id}"
        os.makedirs(screenshot_dir, exist_ok=True)

        # Save all images (screenshots and object images) for this doc
        await parsed_doc.asave_all_images(screenshot_dir)

        for page_index, page in enumerate(parsed_doc.pages):
            page_text = page.text
            screenshot_filename = f"page_{page_index + 1}.jpg"
            screenshot_path = os.path.join(screenshot_dir, screenshot_filename)

            # Allow full page chunking if chunk_size == -1
            actual_chunk_size = len(page_text) if chunk_size == -1 else chunk_size

            for block_index in range(0, len(page_text), actual_chunk_size):
                text_chunk = page_text[block_index:block_index + actual_chunk_size]
                chunk_id = compute_mdhash_id(text_chunk, prefix="chunk-")

                chunks[chunk_id] = {
                    "content": text_chunk,
                    "full_doc_id": doc_id,
                    "file_path": file_path,
                    "page_number": page_index + 1,
                    "screenshot_path": screenshot_path
                }

    async def _process_entity_relation_graph(
        self, chunk: dict[str, Any], pipeline_status=None, pipeline_status_lock=None
    ) -> list:
        try:
            chunk_results = await extract_entities(
                chunk,
                global_config=asdict(self),
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
                llm_response_cache=self.llm_response_cache,
            )
            return chunk_results
        except Exception as e:
            error_msg = f"Failed to extract entities and relationships: {str(e)}"
            logger.error(error_msg)
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = error_msg
                pipeline_status["history_messages"].append(error_msg)
            raise e

    async def aquery(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> str | AsyncIterator[str]:
        """
        Perform a async query.

        Args:
            query (str): The query to be executed.
            param (QueryParam): Configuration parameters for query execution.
                If param.model_func is provided, it will be used instead of the global model.
            system_prompt (Optional[str]): Custom prompts for fine-tuned control over the system's behavior. Defaults to None, which uses PROMPTS["rag_response"].

        Returns:
            str: The result of the query execution.
        """
        # Retrieve context
        context = await self.aretrieve_context(query, param)
        
        # Handle early return for context-only requests
        if param.only_need_context:
            if context is not None:
                return context
            else:
                return PROMPTS["fail_response"]
        
        # Generate response
        response = await self.agenerate_response(query, context, param, system_prompt)
        
        await self._query_done()
        return response

    async def aretrieve_context(
        self,
        query: str,
        param: QueryParam = QueryParam(),
    ) -> str | None:
        """
        Retrieve context/chunks for a query based on the specified mode.
        
        Args:
            query (str): The query to retrieve context for.
            param (QueryParam): Configuration parameters for query execution.
            
        Returns:
            str | None: The retrieved context or None if no context could be built.
        """
        global_config = asdict(self)
        param.original_query = query
        
        if param.mode in ["local", "global", "hybrid", "mix"]:
            return await kg_query_chunks_only(
                query=query.strip(),
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entities_vdb=self.entities_vdb,
                relationships_vdb=self.relationships_vdb,
                text_chunks_db=self.text_chunks,
                query_param=param,
                global_config=global_config,
                hashing_kv=self.llm_response_cache,
                chunks_vdb=self.chunks_vdb,
            )
        elif param.mode == "naive":
            return await naive_query_chunks_only(
                query=query.strip(),
                chunks_vdb=self.chunks_vdb,
                query_param=param,
                global_config=global_config,
                hashing_kv=self.llm_response_cache,
            )
        elif param.mode == "bypass":
            # Bypass mode doesn't need context retrieval
            return None
        else:
            raise ValueError(f"Unknown mode {param.mode}")

    async def agenerate_response(
        self,
        query: str,
        context: str | None,
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> str | AsyncIterator[str]:
        """
        Generate a response using the provided context and query.
        
        Args:
            query (str): The original query.
            context (str | None): The retrieved context.
            param (QueryParam): Configuration parameters for query execution.
            system_prompt (str | None): Custom system prompt.
            
        Returns:
            str | AsyncIterator[str]: The generated response.
        """
        global_config = asdict(self)
        
        if param.mode in ["local", "global", "hybrid", "mix"]:
            return await kg_query_generate_response(
                query=query.strip(),
                context=context,
                query_param=param,
                global_config=global_config,
                hashing_kv=self.llm_response_cache,
                system_prompt=system_prompt,
            )
        elif param.mode == "naive":
            return await naive_query_generate_response(
                query=query.strip(),
                context=context,
                query_param=param,
                global_config=global_config,
                hashing_kv=self.llm_response_cache,
                system_prompt=system_prompt,
            )
        elif param.mode == "bypass":
            # Bypass mode: directly use LLM without knowledge retrieval
            use_llm_func = param.model_func or global_config["llm_model_func"]
            # Apply higher priority (8) to entity/relation summary tasks
            use_llm_func = partial(use_llm_func, _priority=8)

            param.stream = True if param.stream is None else param.stream
            return await use_llm_func(
                query.strip(),
                system_prompt=system_prompt,
                history_messages=param.conversation_history,
                stream=param.stream,
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")


    