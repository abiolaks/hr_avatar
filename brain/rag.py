# rag.py
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from config import CHROMA_DIR, EMBEDDING_MODEL
from logger import logger, log_performance

class RAGManager:
    def __init__(self, persist_dir=CHROMA_DIR):
        self.persist_dir = persist_dir
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.vectorstore = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=self.embeddings
        )
        logger.info(f"RAGManager initialized with persist_dir={persist_dir}")

    @log_performance
    def ingest_documents(self, docs_path):
        """Load all .txt and .pdf files from docs_path, chunk, and add to vectorstore."""
        if not os.path.exists(docs_path):
            logger.error(f"Document path {docs_path} does not exist.")
            return 0

        documents = []
        txt_loader = DirectoryLoader(docs_path, glob="**/*.txt", loader_cls=TextLoader)
        documents += txt_loader.load()
        pdf_loader = DirectoryLoader(docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents += pdf_loader.load()
        logger.info(f"Loaded {len(documents)} documents from {docs_path}")

        if not documents:
            return 0

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")

        self.vectorstore.add_documents(chunks)
        self.vectorstore.persist()
        logger.info(f"Ingested {len(chunks)} chunks into vectorstore")
        return len(chunks)

    @log_performance
    def retrieve(self, query, k=3):
        """Retrieve top-k relevant document chunks."""
        docs = self.vectorstore.similarity_search(query, k=k)
        logger.debug(f"Retrieved {len(docs)} docs for query: {query[:50]}...")
        return docs
