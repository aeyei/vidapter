import os
import json
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import logging
from tqdm import tqdm

from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.schema import ImageNode, Document
import openai

@dataclass
class RAGConfig:
    """Configuration for the RAG engine."""
    output_folder: str = "./mixed_data/"
    vector_store_uri: str = "lancedb"
    text_collection: str = "text_collection"
    image_collection: str = "image_collection"
    similarity_top_k: int = 3
    image_similarity_top_k: int = 3
    model_name: str = "gpt-4-turbo"
    max_new_tokens: int = 1500

def get_openai_api_key(api_key: Optional[str] = None) -> str:
    """Get OpenAI API key from parameter or environment variable."""
    if api_key:
        return api_key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key is required. Set it via api_key parameter or OPENAI_API_KEY environment variable."
        )
    return api_key

class RAGEngine:
    def __init__(self, config: Optional[RAGConfig] = None, api_key: Optional[str] = None):
        """Initialize the RAG engine with configuration."""
        self.config = config or RAGConfig()
        self.logger = logging.getLogger(__name__)
        
        # Set OpenAI API key
        openai.api_key = get_openai_api_key(api_key)
            
        self.retriever_engine = None
        self.qa_template = (
            "Given the provided information, including relevant images and retrieved context from the video, "
            "accurately and precisely answer the query without any additional prior knowledge.\n"
            "---------------------\n"
            "Context: {context_str}\n"
            "Metadata for video: {metadata_str}\n"
            "---------------------\n"
            "Query: {query_str}\n"
            "Answer: "
        )
        
        # Ensure output directory exists
        Path(self.config.output_folder).mkdir(parents=True, exist_ok=True)

    def setup_index(self) -> None:
        """Set up the vector stores and create the multi-modal index."""
        try:
            self.logger.info("Setting up vector stores...")
            text_store = LanceDBVectorStore(
                uri=self.config.vector_store_uri,
                table_name=self.config.text_collection
            )
            image_store = LanceDBVectorStore(
                uri=self.config.vector_store_uri,
                table_name=self.config.image_collection
            )
            
            storage_context = StorageContext.from_defaults(
                vector_store=text_store,
                image_store=image_store
            )
            
            self.logger.info("Loading documents...")
            documents = SimpleDirectoryReader(self.config.output_folder).load_data()
            
            self.logger.info("Creating multi-modal index...")
            index = MultiModalVectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context
            )
            
            self.retriever_engine = index.as_retriever(
                similarity_top_k=self.config.similarity_top_k,
                image_similarity_top_k=self.config.image_similarity_top_k
            )
            self.logger.info("Index setup complete")
            
        except Exception as e:
            self.logger.error(f"Error setting up index: {str(e)}")
            raise

    def retrieve(self, query_str: str) -> Tuple[List[str], List[str]]:
        """Retrieve relevant content for a query."""
        if not self.retriever_engine:
            raise ValueError("Index not set up. Call setup_index() first.")
            
        try:
            self.logger.info(f"Retrieving content for query: {query_str}")
            retrieval_results = self.retriever_engine.retrieve(query_str)
            retrieved_image = []
            retrieved_text = []
            
            for res_node in tqdm(retrieval_results, desc="Processing retrieval results"):
                if isinstance(res_node.node, ImageNode):
                    retrieved_image.append(res_node.node.metadata["file_path"])
                else:
                    retrieved_text.append(res_node.text)
                    
            return retrieved_image, retrieved_text
            
        except Exception as e:
            self.logger.error(f"Error during retrieval: {str(e)}")
            raise

    def generate_response(
        self,
        query_str: str,
        context_str: str,
        metadata: Dict[str, Any],
        image_paths: List[str]
    ) -> str:
        """Generate a response using the LLM."""
        try:
            self.logger.info("Loading image documents...")
            image_documents = SimpleDirectoryReader(
                input_dir=self.config.output_folder,
                input_files=image_paths
            ).load_data()
            
            self.logger.info("Initializing OpenAI multi-modal LLM...")
            openai_mm_llm = OpenAIMultiModal(
                model=self.config.model_name,
                api_key=openai.api_key,
                max_new_tokens=self.config.max_new_tokens
            )
            
            self.logger.info("Generating response...")
            response = openai_mm_llm.complete(
                prompt=self.qa_template.format(
                    context_str=context_str,
                    query_str=query_str,
                    metadata_str=json.dumps(metadata, indent=2)
                ),
                image_documents=image_documents
            )
            
            return response.text
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raise 