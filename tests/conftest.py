"""
Pytest configuration ve fixtures
"""
import sys
import os
from pathlib import Path

# Backend modüllerini import edebilmek için path ekleme
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pytest


@pytest.fixture
def sample_text():
    """Test için örnek metin"""
    return """
    Yapay zeka, makinelerin insan benzeri öğrenme ve problem çözme 
    yeteneklerini taklit etmesidir. Makine öğrenmesi yapay zekanın bir alt dalıdır.
    Derin öğrenme ise makine öğrenmesinin gelişmiş bir formudur.
    RAG (Retrieval-Augmented Generation) yapay zeka sistemlerinde kullanılan 
    bir tekniktir. Bu teknik, dokümanlardan bilgi çekerek daha doğru cevaplar üretir.
    """


@pytest.fixture
def sample_filename():
    """Test için örnek dosya adı"""
    return "test_document.txt"


@pytest.fixture
def sample_query():
    """Test için örnek sorgu"""
    return "RAG nedir?"