"""
Configuración centralizada para el servicio de inferencia optimizada.

Este módulo contiene todas las constantes y configuraciones necesarias
para el funcionamiento del sistema de inferencia con cuantización.
"""

import os
from typing import Dict, Any

# Configuración del modelo
MODEL_NAME = "EleutherAI/gpt-j-6B"
QUANTIZED_MODEL_PATH = "./quantized_model"
QUANTIZATION_BITS = 8  # 8 bits para balance entre velocidad y calidad

# Configuración del servidor
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
MAX_BATCH_SIZE = 4
MAX_QUEUE_SIZE = 100
BATCH_TIMEOUT = 0.5  # segundos para esperar antes de procesar lote parcial

# Configuración de generación de texto
DEFAULT_MAX_LENGTH = 50
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_DO_SAMPLE = True

# Configuración de logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Configuración de hardware
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
TORCH_DTYPE = "float16"

# Configuración de benchmarks
BENCHMARK_REQUESTS = 100
BENCHMARK_CONCURRENCY = 10
BENCHMARK_DURATION = 60  # segundos

# Configuración de pruebas
TEST_TIMEOUT = 30  # segundos
TEST_TEXT = "El futuro de la inteligencia artificial"

def get_model_config() -> Dict[str, Any]:
    """
    Obtiene la configuración específica para el modelo.

    Returns:
        Dict[str, Any]: Diccionario con configuración del modelo
    """
    return {
        "model_name": MODEL_NAME,
        "quantized_path": QUANTIZED_MODEL_PATH,
        "quantization_bits": QUANTIZATION_BITS,
        "device": DEVICE,
        "torch_dtype": TORCH_DTYPE
    }

def get_server_config() -> Dict[str, Any]:
    """
    Obtiene la configuración específica para el servidor.

    Returns:
        Dict[str, Any]: Diccionario con configuración del servidor
    """
    return {
        "host": SERVER_HOST,
        "port": SERVER_PORT,
        "max_batch_size": MAX_BATCH_SIZE,
        "max_queue_size": MAX_QUEUE_SIZE,
        "batch_timeout": BATCH_TIMEOUT
    }

def get_generation_config() -> Dict[str, Any]:
    """
    Obtiene la configuración específica para la generación de texto.

    Returns:
        Dict[str, Any]: Diccionario con configuración de generación
    """
    return {
        "max_length": DEFAULT_MAX_LENGTH,
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "do_sample": DEFAULT_DO_SAMPLE
    }
