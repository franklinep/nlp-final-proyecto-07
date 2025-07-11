"""
Servidor FastAPI para inferencia optimizada con cuantización y dynamic batching.

Este módulo implementa un servidor web que maneja peticiones de generación
de texto usando un modelo cuantizado con procesamiento en lotes dinámicos.
"""

import asyncio
import time
import uuid
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import uvicorn

from .quantize import ModelQuantizer
from .config import get_server_config, get_model_config
from .utils import setup_logging, measure_time, validate_text_input

logger = setup_logging(__name__)

# Modelos de datos para la API
class GenerationRequest(BaseModel):
    """Modelo para peticiones de generación de texto."""
    text: str = Field(..., description="Texto de entrada para generar")
    max_length: Optional[int] = Field(50, description="Longitud máxima del texto generado")
    temperature: Optional[float] = Field(0.7, description="Temperatura para sampling")
    top_p: Optional[float] = Field(0.9, description="Top-p para sampling")
    do_sample: Optional[bool] = Field(True, description="Usar sampling o greedy")

class GenerationResponse(BaseModel):
    """Modelo para respuestas de generación de texto."""
    request_id: str = Field(..., description="ID único de la petición")
    input_text: str = Field(..., description="Texto de entrada")
    generated_text: str = Field(..., description="Texto generado")
    processing_time: float = Field(..., description="Tiempo de procesamiento en segundos")
    tokens_generated: int = Field(..., description="Número de tokens generados")

class HealthResponse(BaseModel):
    """Modelo para respuestas de salud del servicio."""
    status: str = Field(..., description="Estado del servicio")
    model_loaded: bool = Field(..., description="Si el modelo está cargado")
    queue_size: int = Field(..., description="Tamaño actual de la cola")
    processed_requests: int = Field(..., description="Peticiones procesadas")

# Variables globales para el servidor
model_quantizer: Optional[ModelQuantizer] = None
request_queue: Optional[asyncio.Queue] = None
processing_stats = {
    "processed_requests": 0,
    "total_processing_time": 0.0,
    "average_processing_time": 0.0
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestor de ciclo de vida del servidor FastAPI.

    Args:
        app (FastAPI): Instancia de la aplicación FastAPI
    """
    # Inicialización
    logger.info("Iniciando servidor de inferencia optimizada")

    global model_quantizer, request_queue

    try:
        # Configurar cola de peticiones
        server_config = get_server_config()
        request_queue = asyncio.Queue(maxsize=server_config['max_queue_size'])

        # Inicializar cuantizador
        model_quantizer = ModelQuantizer()

        # Cargar modelo cuantizado
        model_config = get_model_config()
        try:
            # Intentar cargar modelo guardado
            model_quantizer.load_quantized_model(model_config['quantized_path'])
            logger.info("Modelo cuantizado cargado desde disco")
        except Exception as e:
            logger.warning(f"No se pudo cargar modelo guardado: {e}")
            logger.info("Cuantizando modelo desde cero...")
            model_quantizer.load_and_quantize_model()
            model_quantizer.save_quantized_model(model_config['quantized_path'])

        # Iniciar worker de procesamiento
        asyncio.create_task(batch_processing_worker())

        logger.info("Servidor iniciado exitosamente")
        yield

    except Exception as e:
        logger.error(f"Error durante la inicialización: {e}")
        raise
    finally:
        # Limpieza
        logger.info("Cerrando servidor...")
        if model_quantizer:
            model_quantizer.cleanup()
        logger.info("Servidor cerrado")

# Crear aplicación FastAPI
app = FastAPI(
    title="Servicio de Inferencia Optimizada",
    description="API para generación de texto con modelos cuantizados y dynamic batching",
    version="1.0.0",
    lifespan=lifespan
)

async def batch_processing_worker():
    """
    Worker en segundo plano para procesar peticiones en lotes.
    """
    logger.info("Iniciando worker de procesamiento en lotes")

    server_config = get_server_config()
    batch_size = server_config['max_batch_size']
    batch_timeout = server_config['batch_timeout']

    while True:
        try:
            batch = []
            start_time = time.time()

            # Recolectar peticiones para el lote
            while len(batch) < batch_size:
                try:
                    # Esperar por petición con timeout
                    remaining_time = batch_timeout - (time.time() - start_time)
                    if remaining_time <= 0:
                        break

                    request_data = await asyncio.wait_for(
                        request_queue.get(),
                        timeout=remaining_time
                    )
                    batch.append(request_data)

                except asyncio.TimeoutError:
                    break

            # Procesar lote si hay peticiones
            if batch:
                await process_batch(batch)

        except Exception as e:
            logger.error(f"Error en worker de procesamiento: {e}")
            await asyncio.sleep(1)  # Evitar bucle infinito en caso de error

async def process_batch(batch: List[Dict[str, Any]]) -> None:
    """
    Procesa un lote de peticiones de generación.

    Args:
        batch (List[Dict[str, Any]]): Lote de peticiones a procesar
    """
    if not batch:
        return

    logger.info(f"Procesando lote de {len(batch)} peticiones")
    batch_start_time = time.time()

    try:
        # Extraer textos y parámetros
        texts = [req['request'].text for req in batch]
        futures = [req['future'] for req in batch]

        # Procesar lote
        results = await generate_batch(texts, batch[0]['request'])

        # Resolver futures con resultados
        for i, (future, result) in enumerate(zip(futures, results)):
            if not future.done():
                processing_time = time.time() - batch_start_time

                response = GenerationResponse(
                    request_id=batch[i]['request_id'],
                    input_text=batch[i]['request'].text,
                    generated_text=result,
                    processing_time=processing_time,
                    tokens_generated=len(result.split()) - len(batch[i]['request'].text.split())
                )

                future.set_result(response)

        # Actualizar estadísticas
        processing_stats['processed_requests'] += len(batch)
        processing_stats['total_processing_time'] += time.time() - batch_start_time
        processing_stats['average_processing_time'] = (
            processing_stats['total_processing_time'] / processing_stats['processed_requests']
        )

        logger.info(f"Lote procesado en {time.time() - batch_start_time:.4f} segundos")

    except Exception as e:
        logger.error(f"Error procesando lote: {e}")
        # Resolver futures con error
        for req in batch:
            future = req['future']
            if not future.done():
                future.set_exception(e)

async def generate_batch(texts: List[str], sample_request: GenerationRequest) -> List[str]:
    """
    Genera texto para un lote de peticiones.

    Args:
        texts (List[str]): Lista de textos de entrada
        sample_request (GenerationRequest): Petición de ejemplo para parámetros

    Returns:
        List[str]: Lista de textos generados
    """
    if not model_quantizer or not model_quantizer.model or not model_quantizer.tokenizer:
        raise RuntimeError("Modelo no está cargado")

    try:
        # Tokenizar entradas
        inputs = model_quantizer.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model_quantizer.model.device)

        # Configurar parámetros de generación
        generation_config = model_quantizer.generation_config
        generation_config.max_length = sample_request.max_length
        generation_config.temperature = sample_request.temperature
        generation_config.top_p = sample_request.top_p
        generation_config.do_sample = sample_request.do_sample

        # Generar texto
        with torch.no_grad():
            outputs = model_quantizer.model.generate(
                **inputs,
                generation_config=generation_config,
                max_new_tokens=50,
                pad_token_id=model_quantizer.tokenizer.pad_token_id
            )

        # Decodificar resultados
        generated_texts = []
        for output in outputs:
            generated_text = model_quantizer.tokenizer.decode(
                output,
                skip_special_tokens=True
            )
            generated_texts.append(generated_text)

        return generated_texts

    except Exception as e:
        logger.error(f"Error durante la generación: {e}")
        raise

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest) -> GenerationResponse:
    """
    Endpoint para generar texto usando el modelo cuantizado.

    Args:
        request (GenerationRequest): Petición de generación

    Returns:
        GenerationResponse: Respuesta con texto generado

    Raises:
        HTTPException: Si hay errores en la petición o procesamiento
    """
    # Validar entrada
    if not validate_text_input(request.text):
        raise HTTPException(
            status_code=400,
            detail="Texto de entrada inválido"
        )

    # Validar parámetros
    if request.max_length and (request.max_length < 1 or request.max_length > 200):
        raise HTTPException(
            status_code=400,
            detail="max_length debe estar entre 1 y 200"
        )

    if request.temperature and (request.temperature < 0.1 or request.temperature > 2.0):
        raise HTTPException(
            status_code=400,
            detail="temperature debe estar entre 0.1 y 2.0"
        )

    if request.top_p and (request.top_p < 0.1 or request.top_p > 1.0):
        raise HTTPException(
            status_code=400,
            detail="top_p debe estar entre 0.1 y 1.0"
        )

    try:
        # Generar ID único para la petición
        request_id = str(uuid.uuid4())

        # Crear future para la respuesta
        future = asyncio.Future()

        # Añadir a la cola de procesamiento
        request_data = {
            'request_id': request_id,
            'request': request,
            'future': future
        }

        await request_queue.put(request_data)

        # Esperar resultado
        try:
            result = await asyncio.wait_for(future, timeout=30.0)
            return result
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="Timeout procesando la petición"
            )

    except asyncio.QueueFull:
        raise HTTPException(
            status_code=503,
            detail="Servicio ocupado, intente más tarde"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en endpoint /generate: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error interno del servidor"
        )

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Endpoint para verificar el estado del servicio.

    Returns:
        HealthResponse: Estado actual del servicio
    """
    model_loaded = (
        model_quantizer is not None and
        model_quantizer.model is not None and
        model_quantizer.tokenizer is not None
    )

    queue_size = request_queue.qsize() if request_queue else 0

    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        queue_size=queue_size,
        processed_requests=processing_stats['processed_requests']
    )

@app.get("/stats")
async def get_stats() -> Dict[str, Any]:
    """
    Endpoint para obtener estadísticas del servicio.

    Returns:
        Dict[str, Any]: Estadísticas actuales
    """
    return {
        "processing_stats": processing_stats,
        "queue_size": request_queue.qsize() if request_queue else 0,
        "model_loaded": model_quantizer is not None and model_quantizer.model is not None
    }

@app.get("/")
async def root():
    """
    Endpoint raíz con información básica del servicio.

    Returns:
        Dict[str, str]: Información del servicio
    """
    return {
        "service": "Servicio de Inferencia Optimizada",
        "version": "1.0.0",
        "status": "running"
    }

def run_server():
    """
    Función para ejecutar el servidor FastAPI.
    """
    server_config = get_server_config()

    uvicorn.run(
        "src.server:app",
        host=server_config['host'],
        port=server_config['port'],
        log_level="info",
        reload=False
    )

if __name__ == "__main__":
    run_server()
