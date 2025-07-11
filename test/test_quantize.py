"""
Pruebas unitarias para el pipeline de cuantización.

Este módulo contiene todas las pruebas para verificar el correcto
funcionamiento del sistema de cuantización de modelos.
"""

import pytest
import tempfile
import shutil
import os
from unittest.mock import Mock, patch, MagicMock
import torch

from src.quantize import ModelQuantizer
from src.config import get_model_config

class TestModelQuantizer:
    """Pruebas para la clase ModelQuantizer."""

    def setup_method(self):
        """Configuración antes de cada prueba."""
        self.quantizer = ModelQuantizer()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Limpieza después de cada prueba."""
        if hasattr(self, 'quantizer'):
            self.quantizer.cleanup()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @pytest.fixture
    def mock_tokenizer(self):
        """Mock del tokenizer para pruebas."""
        tokenizer = Mock()
        tokenizer.pad_token = None
        tokenizer.eos_token = "<|endoftext|>"
        tokenizer.pad_token_id = 50256
        tokenizer.eos_token_id = 50256
        tokenizer.save_pretrained = Mock()
        return tokenizer

    @pytest.fixture
    def mock_model(self):
        """Mock del modelo para pruebas."""
        model = Mock()
        model.eval = Mock()
        model.device = "cpu"
        model.save_pretrained = Mock()
        model.generate = Mock()
        return model

    def test_init(self):
        """Prueba la inicialización del cuantizador."""
        assert self.quantizer.config is not None
        assert self.quantizer.tokenizer is None
        assert self.quantizer.model is None
        assert self.quantizer.generation_config is None

    @patch('src.quantize.BitsAndBytesConfig')
    @patch('src.quantize.AutoModelForCausalLM')
    @patch('src.quantize.AutoTokenizer')
    def test_load_and_quantize_model(self, mock_tokenizer_class, mock_model_class, mock_bnb_config_class):
        """Prueba la carga y cuantización del modelo."""
        # Configurar mocks
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_tokenizer.pad_token_id = 50256
        mock_tokenizer.eos_token_id = 50256
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model.device = "cpu"
        mock_model_class.from_pretrained.return_value = mock_model

        # Ejecutar método
        tokenizer, model = self.quantizer.load_and_quantize_model()

        # Verificar resultados
        assert tokenizer is not None
        assert model is not None
        assert self.quantizer.tokenizer is not None
        assert self.quantizer.model is not None
        assert self.quantizer.generation_config is not None

        # Verificar que se llamó a la configuración de bitsandbytes
        mock_bnb_config_class.assert_called_once()

        # Verificar que se configuró el pad_token
        assert mock_tokenizer.pad_token == mock_tokenizer.eos_token

        # Verificar que se llamó eval()
        mock_model.eval.assert_called_once()

    def test_save_quantized_model_without_loaded_model(self):
        """Prueba guardar modelo sin haberlo cargado."""
        with pytest.raises(ValueError, match="El modelo debe estar cargado"):
            self.quantizer.save_quantized_model(self.temp_dir)

    def test_save_quantized_model_with_loaded_model(self, mock_tokenizer, mock_model):
        """Prueba guardar modelo después de cargarlo."""
        # Configurar estado del cuantizador
        self.quantizer.tokenizer = mock_tokenizer
        self.quantizer.model = mock_model

        # Ejecutar método
        self.quantizer.save_quantized_model(self.temp_dir)

        # Verificar que se llamaron los métodos de guardado
        mock_model.save_pretrained.assert_called_once_with(self.temp_dir)
        mock_tokenizer.save_pretrained.assert_called_once_with(self.temp_dir)

    @patch('src.quantize.AutoTokenizer')
    @patch('src.quantize.AutoModelForCausalLM')
    def test_load_quantized_model(self, mock_model_class, mock_tokenizer_class):
        """Prueba cargar modelo cuantizado desde disco."""
        # Configurar mocks
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_tokenizer.pad_token_id = 50256
        mock_tokenizer.eos_token_id = 50256
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model.device = "cpu"
        mock_model_class.from_pretrained.return_value = mock_model

        # Ejecutar método
        tokenizer, model = self.quantizer.load_quantized_model(self.temp_dir)

        # Verificar resultados
        assert tokenizer is not None
        assert model is not None
        assert self.quantizer.tokenizer is not None
        assert self.quantizer.model is not None
        assert self.quantizer.generation_config is not None

        # Verificar que se llamó from_pretrained con la ruta correcta
        mock_tokenizer_class.from_pretrained.assert_called_once_with(
            self.temp_dir, padding_side="left"
        )
        mock_model_class.from_pretrained.assert_called_once()

    def test_validate_model_precision_without_model(self):
        """Prueba validar precisión sin modelo cargado."""
        with pytest.raises(ValueError, match="El modelo debe estar cargado"):
            self.quantizer.validate_model_precision()

    def test_validate_model_precision_with_model(self, mock_tokenizer, mock_model):
        """Prueba validar precisión con modelo cargado."""
        # Configurar mocks
        mock_inputs = {
            'input_ids': torch.tensor([[1, 2, 3, 4]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1]])
        }
        mock_tokenizer.return_value = MagicMock()
        mock_tokenizer.return_value.__getitem__.side_effect = mock_inputs.__getitem__
        mock_tokenizer.return_value.to.return_value = mock_tokenizer.return_value
        mock_tokenizer.decode = Mock(return_value="El futuro de la inteligencia artificial será increíble")

        mock_model.device = "cpu"
        mock_model.generate = Mock(return_value=[torch.tensor([1, 2, 3, 4, 5, 6])])

        # Configurar estado del cuantizador
        self.quantizer.tokenizer = mock_tokenizer
        self.quantizer.model = mock_model
        self.quantizer.generation_config = Mock()

        # Ejecutar método
        result = self.quantizer.validate_model_precision()

        # Verificar resultados
        assert result is not None
        assert "validation_passed" in result
        assert "input_text" in result
        assert "generated_text" in result
        assert "input_tokens" in result
        assert "output_tokens" in result
        assert "new_tokens" in result

        # Verificar que se llamaron los métodos correctos
        mock_tokenizer.assert_called_once()
        mock_model.generate.assert_called_once()
        mock_tokenizer.decode.assert_called_once()

    def test_validate_model_precision_with_exception(self, mock_tokenizer, mock_model):
        """Prueba validar precisión con excepción."""
        # Configurar mocks para lanzar excepción
        mock_tokenizer.side_effect = Exception("Error de tokenización")

        # Configurar estado del cuantizador
        self.quantizer.tokenizer = mock_tokenizer
        self.quantizer.model = mock_model
        self.quantizer.generation_config = Mock()

        # Ejecutar método
        result = self.quantizer.validate_model_precision()

        # Verificar que se manejó la excepción
        assert result is not None
        assert result["validation_passed"] is False
        assert "error" in result

    def test_cleanup(self, mock_tokenizer, mock_model):
        """Prueba la limpieza de recursos."""
        # Configurar estado del cuantizador
        self.quantizer.tokenizer = mock_tokenizer
        self.quantizer.model = mock_model

        # Ejecutar limpieza
        self.quantizer.cleanup()

        # Verificar que se liberaron los recursos
        assert self.quantizer.tokenizer is None
        assert self.quantizer.model is None

    @patch('src.quantize.torch.cuda.is_available')
    @patch('src.quantize.torch.cuda.empty_cache')
    @patch('src.quantize.gc.collect')
    def test_cleanup_with_cuda(self, mock_gc, mock_empty_cache, mock_cuda_available):
        """Prueba la limpieza con CUDA disponible."""
        mock_cuda_available.return_value = True

        # Configurar estado del cuantizador
        self.quantizer.tokenizer = Mock()
        self.quantizer.model = Mock()

        # Ejecutar limpieza
        self.quantizer.cleanup()

        # Verificar que se llamaron los métodos de limpieza
        mock_empty_cache.assert_called_once()
        mock_gc.assert_called_once()

    def test_get_model_config_integration(self):
        """Prueba de integración con la configuración del modelo."""
        config = get_model_config()

        # Verificar que la configuración tiene los campos esperados
        assert "model_name" in config
        assert "quantized_path" in config
        assert "quantization_bits" in config
        assert "device" in config
        assert "torch_dtype" in config

        # Verificar valores por defecto
        assert config["model_name"] == "EleutherAI/gpt-j-6B"
        assert config["quantization_bits"] == 8

class TestQuantizationPipeline:
    """Pruebas de integración para el pipeline completo."""

    def setup_method(self):
        """Configuración antes de cada prueba."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Limpieza después de cada prueba."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('src.quantize.ModelQuantizer')
    def test_main_function(self, mock_quantizer_class):
        """Prueba la función principal del pipeline."""
        # Configurar mock
        mock_quantizer = Mock()
        mock_quantizer.load_and_quantize_model.return_value = (Mock(), Mock())
        mock_quantizer.validate_model_precision.return_value = {"validation_passed": True}
        mock_quantizer.save_quantized_model = Mock()
        mock_quantizer.cleanup = Mock()
        mock_quantizer_class.return_value = mock_quantizer

        # Importar y ejecutar main
        from src.quantize import main
        main()

        # Verificar que se llamaron los métodos correctos
        mock_quantizer.load_and_quantize_model.assert_called_once()
        mock_quantizer.validate_model_precision.assert_called_once()
        mock_quantizer.save_quantized_model.assert_called_once()
        mock_quantizer.cleanup.assert_called_once()

    @patch('src.quantize.ModelQuantizer')
    def test_main_function_with_exception(self, mock_quantizer_class):
        """Prueba la función principal con excepción."""
        # Configurar mock para lanzar excepción
        mock_quantizer = Mock()
        mock_quantizer.load_and_quantize_model.side_effect = Exception("Error de carga")
        mock_quantizer.cleanup = Mock()
        mock_quantizer_class.return_value = mock_quantizer

        # Importar y ejecutar main
        from src.quantize import main

        with pytest.raises(Exception, match="Error de carga"):
            main()

        # Verificar que se llamó cleanup incluso con excepción
        mock_quantizer.cleanup.assert_called_once()

# Función para ejecutar pruebas
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
