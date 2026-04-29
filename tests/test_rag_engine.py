"""
RAG Engine Tests - Chunking, Embedding ve FAISS Index testleri
"""
import pytest
import numpy as np
from rag_engine import (
    chunk_text,
    embed_texts,
    create_index,
    search_index,
)


class TestChunking:
    """Metin parçalama (chunking) testleri"""

    def test_chunk_text_returns_chunks_and_metadata(self, sample_text, sample_filename):
        """chunk_text iki değer döndürmeli: chunks ve metadata"""
        chunks, metadata = chunk_text(sample_text, filename=sample_filename)
        assert isinstance(chunks, list)
        assert isinstance(metadata, list)
        assert len(chunks) > 0
        assert len(metadata) == len(chunks)

    def test_chunk_metadata_has_required_fields(self, sample_text, sample_filename):
        """Her metadata kayıt gerekli alanları içermeli"""
        _, metadata = chunk_text(sample_text, filename=sample_filename)
        for m in metadata:
            assert "doc_id" in m
            assert "source" in m
            assert "content" in m
            assert m["source"] == sample_filename

    def test_chunk_size_respects_limit(self, sample_filename):
        """Chunk boyutu belirtilen limiti aşmamalı"""
        long_text = "test " * 1000
        chunks, _ = chunk_text(long_text, filename=sample_filename, chunk_size=600, overlap=100)
        for chunk in chunks:
            assert len(chunk) <= 600


class TestEmbedding:
    """Embedding (vektör) testleri"""

    def test_embed_texts_returns_numpy_array(self):
        """embed_texts numpy array döndürmeli"""
        texts = ["Bu bir test cümlesidir.", "Bu da başka bir cümle."]
        embeddings = embed_texts(texts)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.dtype == np.float32

    def test_embedding_dimension_consistency(self):
        """Her embedding aynı boyutta olmalı"""
        texts = ["Birinci cümle", "İkinci cümle", "Üçüncü cümle"]
        embeddings = embed_texts(texts)
        assert embeddings.shape[0] == 3
        assert all(len(emb) == embeddings.shape[1] for emb in embeddings)

    def test_similar_texts_have_similar_embeddings(self):
        """Benzer cümlelerin embeddingleri yakın olmalı"""
        texts = [
            "Yapay zeka harika bir teknolojidir.",
            "AI çok güzel bir teknolojidir.",
            "Pizza çok lezzetlidir."
        ]
        embeddings = embed_texts(texts)
        sim_1_2 = np.dot(embeddings[0], embeddings[1])
        sim_1_3 = np.dot(embeddings[0], embeddings[2])
        assert sim_1_2 > sim_1_3


class TestFAISSIndex:
    """FAISS index testleri"""

    def test_create_index(self):
        """FAISS index oluşturulmalı"""
        texts = ["Cümle bir", "Cümle iki", "Cümle üç"]
        embeddings = embed_texts(texts)
        index = create_index(embeddings)
        assert index is not None
        assert index.ntotal == 3

    def test_search_index_returns_relevant_results(self, sample_text, sample_filename):
        """Index araması ilgili sonuçları döndürmeli"""
        chunks, metadata = chunk_text(sample_text, filename=sample_filename)
        embeddings = embed_texts(chunks)
        index = create_index(embeddings)
        
        results = search_index(
            index=index,
            query="RAG nedir?",
            metadata=metadata,
            k=2
        )
        assert isinstance(results, list)
        assert len(results) > 0
        assert len(results) <= 2

    def test_search_returns_metadata_format(self, sample_text, sample_filename):
        """Arama sonucu doğru formatta olmalı"""
        chunks, metadata = chunk_text(sample_text, filename=sample_filename)
        embeddings = embed_texts(chunks)
        index = create_index(embeddings)
        
        results = search_index(
            index=index,
            query="yapay zeka",
            metadata=metadata,
            k=1
        )
        if len(results) > 0:
            assert "content" in results[0]
            assert "source" in results[0]