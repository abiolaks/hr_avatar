# Test the rag system
# tests/test_rag.py
from brain.rag import RAGManager
import tempfile
import os

def test_ingest_and_retrieve():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a dummy text file
        doc_path = os.path.join(tmpdir, "test.txt")
        with open(doc_path, "w") as f:
            f.write("Parental leave is 12 weeks.")

        rag = RAGManager(persist_dir=os.path.join(tmpdir, "chroma"))
        count = rag.ingest_documents(tmpdir)
        assert count == 1  # one chunk

        docs = rag.retrieve("How long is parental leave?")
        assert len(docs) > 0
        assert "12 weeks" in docs[0].page_content
