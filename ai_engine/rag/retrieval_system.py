"""
Retrieval-Augmented Generation (RAG) system.
Uses LangChain + FAISS/ChromaDB for vector storage and hybrid search.
"""
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from langchain.document_loaders import (
        WebBaseLoader,
        TextLoader,
        DirectoryLoader,
        PyPDFLoader,
    )
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS, Chroma
    from langchain.chains import RetrievalQA
    from langchain.llms.base import LLM
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retriever combining keyword and semantic search."""

    def __init__(
        self,
        vectorstore,
        k: int = 5,
        score_threshold: float = 0.7,
    ):
        self.vectorstore = vectorstore
        self.k = k
        self.score_threshold = score_threshold

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve documents using hybrid search."""
        # Semantic search with scores
        try:
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                query, k=self.k * 2
            )

            # Filter by score threshold
            # For FAISS, lower score is better (distance)
            # For ChromaDB, higher score is better (similarity)
            filtered_docs = []
            for doc, score in docs_with_scores:
                # Accept if score indicates good match
                # For distance-based: score < threshold (lower is better)
                # For similarity-based: score > threshold (higher is better)
                # We'll accept if score is reasonable for either case
                if isinstance(score, (int, float)):
                    # Try both interpretations
                    if score < 1.0 or score > self.score_threshold:
                        filtered_docs.append(doc)
                else:
                    filtered_docs.append(doc)

            filtered_docs = filtered_docs[:self.k] if len(filtered_docs) >= self.k else filtered_docs
        except Exception as e:
            logger.warning(f"Hybrid search failed, falling back to simple search: {e}")
            # Fallback to simple similarity search
            filtered_docs = self.vectorstore.similarity_search(query, k=self.k)

        # If not enough results, add more from simple search
        if len(filtered_docs) < self.k:
            keyword_docs = self.vectorstore.similarity_search(
                query, k=self.k - len(filtered_docs)
            )
            # Avoid duplicates
            existing_ids = {id(doc) for doc in filtered_docs}
            for doc in keyword_docs:
                if id(doc) not in existing_ids:
                    filtered_docs.append(doc)
                    if len(filtered_docs) >= self.k:
                        break

        return filtered_docs[:self.k]


class RAGSystem:
    """
    Retrieval-Augmented Generation system.
    Supports multiple document sources and vector stores.
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        vectorstore_type: str = "faiss",
        persist_directory: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize RAG system.

        Args:
            embedding_model: HuggingFace embedding model name
            vectorstore_type: "faiss" or "chroma"
            persist_directory: Directory to persist vectorstore
            chunk_size: Chunk size for document splitting
            chunk_overlap: Overlap between chunks
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain, faiss-cpu (or faiss-gpu), and chromadb are required. "
                "Install with: pip install langchain faiss-cpu chromadb pypdf"
            )

        self.embedding_model = embedding_model
        self.vectorstore_type = vectorstore_type
        self.persist_directory = Path(persist_directory) if persist_directory else None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize embeddings
        logger.info(f"Loading embeddings model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
        )

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        # Load or create vectorstore
        self.vectorstore = None
        self._load_or_create_vectorstore()

    def _load_or_create_vectorstore(self):
        """Load existing vectorstore or create new one."""
        if self.persist_directory and self.persist_directory.exists():
            try:
                if self.vectorstore_type == "faiss":
                    self.vectorstore = FAISS.load_local(
                        str(self.persist_directory),
                        self.embeddings,
                    )
                    logger.info(f"Loaded FAISS vectorstore from {self.persist_directory}")
                elif self.vectorstore_type == "chroma":
                    self.vectorstore = Chroma(
                        persist_directory=str(self.persist_directory),
                        embedding_function=self.embeddings,
                    )
                    logger.info(f"Loaded ChromaDB vectorstore from {self.persist_directory}")
                return
            except Exception as e:
                logger.warning(f"Failed to load existing vectorstore: {e}. Creating new one.")

        # Create new vectorstore
        if self.vectorstore_type == "faiss":
            # Create empty FAISS index (will be populated when documents are added)
            # We'll create it when we add documents
            self.vectorstore = None
        elif self.vectorstore_type == "chroma":
            self.vectorstore = Chroma(
                persist_directory=str(self.persist_directory) if self.persist_directory else None,
                embedding_function=self.embeddings,
            )
        else:
            raise ValueError(f"Unknown vectorstore_type: {self.vectorstore_type}")

    def add_documents(
        self,
        documents: List[Document],
        source: Optional[str] = None,
    ):
        """Add documents to the vectorstore."""
        if not documents:
            return

        # Split documents
        splits = self.text_splitter.split_documents(documents)

        # Add source metadata if provided
        if source:
            for split in splits:
                if "source" not in split.metadata:
                    split.metadata["source"] = source

        logger.info(f"Adding {len(splits)} document chunks to vectorstore")

        # Add to vectorstore
        if self.vectorstore is None:
            # Create new FAISS vectorstore
            if self.vectorstore_type == "faiss":
                self.vectorstore = FAISS.from_documents(splits, self.embeddings)
            else:
                self.vectorstore.add_documents(splits)
        else:
            # Add to existing vectorstore
            if self.vectorstore_type == "faiss":
                # FAISS doesn't support incremental updates easily, need to merge
                new_store = FAISS.from_documents(splits, self.embeddings)
                self.vectorstore.merge_from(new_store)
            else:
                self.vectorstore.add_documents(splits)

        # Persist
        if self.persist_directory:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            if self.vectorstore_type == "faiss":
                self.vectorstore.save_local(str(self.persist_directory))
            # ChromaDB auto-persists

    def load_from_urls(self, urls: List[str]):
        """Load documents from URLs."""
        logger.info(f"Loading documents from {len(urls)} URLs")
        loader = WebBaseLoader(urls)
        documents = loader.load()
        self.add_documents(documents, source="web")

    def load_from_directory(
        self,
        directory: str,
        glob_pattern: str = "**/*.txt",
        file_type: str = "text",
    ):
        """Load documents from a directory."""
        logger.info(f"Loading documents from directory: {directory}")
        if file_type == "text":
            loader = DirectoryLoader(directory, glob=glob_pattern, loader_cls=TextLoader)
        elif file_type == "pdf":
            loader = DirectoryLoader(directory, glob=glob_pattern, loader_cls=PyPDFLoader)
        else:
            raise ValueError(f"Unknown file_type: {file_type}")

        documents = loader.load()
        self.add_documents(documents, source=str(directory))

    def load_from_file(self, file_path: str):
        """Load document from a single file."""
        logger.info(f"Loading document from file: {file_path}")
        file_path_obj = Path(file_path)
        if file_path_obj.suffix == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)

        documents = loader.load()
        self.add_documents(documents, source=file_path)

    def query(
        self,
        query: str,
        llm: LLM,
        k: int = 5,
        use_hybrid: bool = True,
        return_source_documents: bool = False,
    ) -> Dict[str, Any]:
        """
        Query the RAG system.

        Args:
            query: Query string
            llm: Language model for generation
            k: Number of documents to retrieve
            use_hybrid: Use hybrid retrieval
            return_source_documents: Return source documents

        Returns:
            Dictionary with answer and optionally source documents
        """
        if self.vectorstore is None:
            raise RuntimeError("Vectorstore is empty. Add documents first.")

        # Create retriever
        if use_hybrid:
            hybrid_retriever = HybridRetriever(self.vectorstore, k=k)
            # Create a LangChain-compatible retriever wrapper
            from langchain.schema import BaseRetriever
            class HybridLangChainRetriever(BaseRetriever):
                def __init__(self, hybrid_retriever):
                    super().__init__()
                    self.hybrid_retriever = hybrid_retriever
                
                def get_relevant_documents(self, query: str):
                    return self.hybrid_retriever.retrieve(query)
                
                async def aget_relevant_documents(self, query: str):
                    # Async version
                    import asyncio
                    return await asyncio.get_event_loop().run_in_executor(
                        None, self.hybrid_retriever.retrieve, query
                    )
            
            retriever_obj = HybridLangChainRetriever(hybrid_retriever)
        else:
            retriever_obj = self.vectorstore.as_retriever(
                search_kwargs={"k": k}
            )

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever_obj,
            return_source_documents=return_source_documents,
        )

        # Run query
        result = qa_chain({"query": query})

        return result

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search without LLM generation."""
        if self.vectorstore is None:
            return []
        return self.vectorstore.similarity_search(query, k=k)

