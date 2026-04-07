# rag.py
import os
import tempfile
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from config import CHROMA_DIR, AZURE_STORAGE_CONNECTION_STRING, AZURE_STORAGE_CONTAINER
from logger import logger, log_performance


# Supported file types for both local and Azure ingestion
_LOADERS = {
    ".txt":  TextLoader,
    ".pdf":  PyPDFLoader,
    ".docx": Docx2txtLoader,
}


class RAGManager:
    def __init__(self, persist_dir=CHROMA_DIR):
        self.persist_dir = persist_dir
        # FastEmbedEmbeddings runs in-process (no HTTP round-trip) — ~0.05s vs ~4.6s for OllamaEmbeddings
        self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.vectorstore = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=self.embeddings,
        )
        logger.info(f"RAGManager initialized | persist_dir={persist_dir}")

    # ── Internal: chunk + add to vectorstore ─────────────────────────────────

    def _add_documents(self, documents: list) -> int:
        """Chunk a list of LangChain Documents and add them to the vectorstore."""
        if not documents:
            return 0
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")
        self.vectorstore.add_documents(chunks)
        logger.info(f"Ingested {len(chunks)} chunks into vectorstore")
        return len(chunks)

    # ── Local document folder ─────────────────────────────────────────────────

    @log_performance
    def ingest_documents(self, docs_path: str) -> int:
        """
        Load all .txt, .pdf, and .docx files from a local folder,
        chunk them, and add to the vectorstore.
        """
        if not os.path.exists(docs_path):
            logger.error(f"Document path does not exist: {docs_path}")
            return 0

        documents = []
        for glob, loader_cls in [
            ("**/*.txt",  TextLoader),
            ("**/*.pdf",  PyPDFLoader),
            ("**/*.docx", Docx2txtLoader),
        ]:
            loader = DirectoryLoader(docs_path, glob=glob, loader_cls=loader_cls)
            docs = loader.load()
            documents += docs
            logger.info(f"Loaded {len(docs)} {glob} file(s) from {docs_path}")

        logger.info(f"Total local documents loaded: {len(documents)}")
        return self._add_documents(documents)

    # ── Azure Blob Storage ────────────────────────────────────────────────────

    @log_performance
    def ingest_from_azure(
        self,
        container_name: str = None,
        connection_string: str = None,
    ) -> int:
        """
        Download all supported documents (.txt, .pdf, .docx) from an Azure
        Blob Storage container, chunk them, and add to the vectorstore.

        Credentials are read from environment variables by default:
            AZURE_STORAGE_CONNECTION_STRING
            AZURE_STORAGE_CONTAINER

        You can override either at call time:
            rag.ingest_from_azure(
                container_name="my-container",
                connection_string="DefaultEndpointsProtocol=https;..."
            )
        """
        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError:
            logger.error(
                "azure-storage-blob is not installed. "
                "Run: pip install azure-storage-blob"
            )
            return 0

        conn_str  = connection_string or AZURE_STORAGE_CONNECTION_STRING
        container = container_name    or AZURE_STORAGE_CONTAINER

        if not conn_str:
            logger.error(
                "Azure connection string not set. "
                "Export AZURE_STORAGE_CONNECTION_STRING or pass connection_string=."
            )
            return 0

        logger.info(f"Connecting to Azure Blob Storage | container: {container}")
        try:
            service_client   = BlobServiceClient.from_connection_string(conn_str)
            container_client = service_client.get_container_client(container)
        except Exception as e:
            logger.error(f"Failed to connect to Azure Blob Storage: {e}")
            return 0

        # Download blobs to a temp directory, then run the standard ingest pipeline
        with tempfile.TemporaryDirectory() as tmp_dir:
            downloaded = 0
            for blob in container_client.list_blobs():
                ext = os.path.splitext(blob.name)[1].lower()
                if ext not in _LOADERS:
                    logger.debug(f"Skipping unsupported file: {blob.name}")
                    continue

                # Flatten any folder structure in the blob name to avoid conflicts
                local_name = blob.name.replace("/", "__")
                local_path = os.path.join(tmp_dir, local_name)

                try:
                    with open(local_path, "wb") as f:
                        container_client.download_blob(blob.name).readinto(f)
                    logger.info(f"Downloaded blob: {blob.name}")
                    downloaded += 1
                except Exception as e:
                    logger.error(f"Failed to download blob {blob.name}: {e}")
                    continue

            if downloaded == 0:
                logger.warning(
                    f"No supported documents (.txt, .pdf, .docx) found "
                    f"in container '{container}'"
                )
                return 0

            logger.info(f"Downloaded {downloaded} document(s) from Azure")
            return self.ingest_documents(tmp_dir)

    # ── Combined ingestion: local + Azure ─────────────────────────────────────

    def ingest_all(
        self,
        local_path: str = None,
        azure_container: str = None,
        azure_connection_string: str = None,
    ) -> int:
        """
        Ingest from local folder and/or Azure Blob Storage in one call.
        Either source is optional — pass only what you have.

        Examples:
            rag.ingest_all(local_path="./hr_docs")
            rag.ingest_all(azure_container="hr-documents")
            rag.ingest_all(local_path="./hr_docs", azure_container="hr-documents")
        """
        total = 0
        if local_path:
            total += self.ingest_documents(local_path)
        if azure_container or AZURE_STORAGE_CONNECTION_STRING:
            total += self.ingest_from_azure(
                container_name=azure_container,
                connection_string=azure_connection_string,
            )
        logger.info(f"ingest_all complete | total chunks ingested: {total}")
        return total

    # ── Retrieval ─────────────────────────────────────────────────────────────

    @log_performance
    def retrieve(self, query: str, k: int = 3) -> list:
        """Retrieve top-k relevant document chunks for a query."""
        docs = self.vectorstore.similarity_search(query, k=k)
        logger.debug(f"Retrieved {len(docs)} docs for query: {query[:50]}...")
        return docs
