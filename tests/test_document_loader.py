"""
Document Loader Tests - Dosya okuma testleri
"""
import pytest
import tempfile
import os
from document_loader import load_txt, load_file, clean_text


class TestCleanText:
    """Metin temizleme testleri"""

    def test_clean_text_removes_empty_lines(self):
        """Boş satırlar temizlenmeli"""
        dirty = "Satır 1\n\n\nSatır 2\n\n"
        clean = clean_text(dirty)
        assert "\n\n" not in clean
        assert "Satır 1" in clean
        assert "Satır 2" in clean

    def test_clean_text_strips_whitespace(self):
        """Başında ve sonundaki boşluklar temizlenmeli"""
        dirty = "  Test metin  \n   "
        clean = clean_text(dirty)
        assert clean == "Test metin"


class TestLoadTxt:
    """TXT dosya okuma testleri"""

    def test_load_txt_reads_file(self):
        """TXT dosyası okunabilmeli"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False, encoding='utf-8'
        ) as f:
            f.write("Bu bir test metnidir.")
            temp_path = f.name
        
        try:
            content = load_txt(temp_path)
            assert "test metnidir" in content
        finally:
            os.unlink(temp_path)

    def test_load_txt_handles_turkish_chars(self):
        """Türkçe karakterler doğru okunmalı"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False, encoding='utf-8'
        ) as f:
            f.write("Çığşöü ÇIĞŞÖÜ")
            temp_path = f.name
        
        try:
            content = load_txt(temp_path)
            assert "Çığşöü" in content
        finally:
            os.unlink(temp_path)


class TestLoadFile:
    """Genel dosya okuma router testleri"""

    def test_load_file_unsupported_format_raises_error(self):
        """Desteklenmeyen format hata vermeli"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.xyz', delete=False
        ) as f:
            f.write("test")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError):
                load_file(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_file_routes_txt(self):
        """TXT uzantılı dosyalar load_txt'ye yönlenmeli"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False, encoding='utf-8'
        ) as f:
            f.write("İçerik testi")
            temp_path = f.name
        
        try:
            content = load_file(temp_path)
            assert "İçerik testi" in content
        finally:
            os.unlink(temp_path)