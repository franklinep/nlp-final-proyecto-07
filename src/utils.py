"""
Utilidades generales para el servicio de inferencia optimizada.

Este módulo contiene funciones auxiliares para logging, validación,
y otras operaciones comunes del sistema.
"""

import logging
import time
import functools
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import json

from .config import LOG_LEVEL, LOG_FORMAT

def setup_logging(name: str) -> logging.Logger:
    """
    Configura el sistema de logging para el módulo especificado.

    Args:
        name (str): Nombre del módulo para el logger

    Returns:
        logging.Logger: Logger configurado
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    # Evitar duplicar handlers
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(LOG_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

def measure_time(func: Callable) -> Callable:
    """
    Decorador para medir el tiempo de ejecución de funciones.

    Args:
        func (Callable): Función a medir

    Returns:
        Callable: Función decorada con medición de tiempo
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        logger = setup_logging(func.__module__)
        logger.info(f"{func.__name__} ejecutada en {end_time - start_time:.4f} segundos")

        return result
    return wrapper

def validate_text_input(text: str, max_length: int = 1000) -> bool:
    """
    Valida que el texto de entrada sea válido para procesamiento.

    Args:
        text (str): Texto a validar
        max_length (int): Longitud máxima permitida

    Returns:
        bool: True si el texto es válido, False en caso contrario
    """
    if not isinstance(text, str):
        return False

    if len(text.strip()) == 0:
        return False

    if len(text) > max_length:
        return False

    return True

def create_directory_if_not_exists(path: str) -> None:
    """
    Crea un directorio si no existe.

    Args:
        path (str): Ruta del directorio a crear
    """
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Guarda datos en formato JSON.

    Args:
        data (Dict[str, Any]): Datos a guardar
        filepath (str): Ruta del archivo donde guardar
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Carga datos desde un archivo JSON.

    Args:
        filepath (str): Ruta del archivo a cargar

    Returns:
        Optional[Dict[str, Any]]: Datos cargados o None si hay error
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger = setup_logging(__name__)
        logger.error(f"Error cargando JSON desde {filepath}: {e}")
        return None

def batch_list(items: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Divide una lista en lotes de tamaño específico.

    Args:
        items (List[Any]): Lista de elementos a dividir
        batch_size (int): Tamaño de cada lote

    Returns:
        List[List[Any]]: Lista de lotes
    """
    if batch_size <= 0:
        raise ValueError("El tamaño del lote debe ser mayor que 0")

    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

def calculate_percentile(values: List[float], percentile: float) -> float:
    """
    Calcula el percentil especificado de una lista de valores.

    Args:
        values (List[float]): Lista de valores
        percentile (float): Percentil a calcular (0-100)

    Returns:
        float: Valor del percentil
    """
    if not values:
        return 0.0

    sorted_values = sorted(values)
    index = (percentile / 100) * (len(sorted_values) - 1)

    if index.is_integer():
        return sorted_values[int(index)]
    else:
        lower_index = int(index)
        upper_index = lower_index + 1
        weight = index - lower_index
        return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight

def format_latency_stats(latencies: List[float]) -> Dict[str, float]:
    """
    Calcula estadísticas de latencia formateadas.

    Args:
        latencies (List[float]): Lista de latencias en segundos

    Returns:
        Dict[str, float]: Estadísticas de latencia
    """
    if not latencies:
        return {"count": 0, "mean": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0}

    # Convertir a milisegundos
    latencies_ms = [lat * 1000 for lat in latencies]

    return {
        "count": len(latencies_ms),
        "mean": sum(latencies_ms) / len(latencies_ms),
        "p50": calculate_percentile(latencies_ms, 50),
        "p95": calculate_percentile(latencies_ms, 95),
        "p99": calculate_percentile(latencies_ms, 99),
        "max": max(latencies_ms)
    }
