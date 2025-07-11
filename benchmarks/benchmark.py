"""
Script de benchmarking para el servicio de inferencia optimizada.

Este script utiliza asyncio y httpx para enviar un gran número de peticiones
concurrentes al servidor FastAPI y medir las latencias.
"""

import asyncio
import time
import httpx
import pandas as pd
from typing import List, Dict, Any

from src.config import get_server_config, BENCHMARK_CONCURRENCY, BENCHMARK_REQUESTS
from src.utils import setup_logging, format_latency_stats

logger = setup_logging(__name__)

# Textos de ejemplo para las pruebas de carga
SAMPLE_PROMPTS = [
    "Explica la teoría de la relatividad en términos simples.",
    "¿Cuál es la capital de Mongolia?",
    "Escribe un poema corto sobre el océano.",
    "Resume la trama de 'Cien años de soledad'.",
    "Genera una idea para una startup de tecnología.",
    "Traduce 'hello world' al francés.",
    "¿Quién fue el primer programador de la historia?",
    "Describe cómo funciona un modelo de lenguaje grande.",
    "Crea una receta para una cena rápida y saludable.",
    "¿Qué es la computación cuántica?",
]

async def send_request(client: httpx.AsyncClient, url: str, prompt: str) -> Dict[str, Any]:
    """
    Envía una única petición de generación y mide la latencia.

    Args:
        client (httpx.AsyncClient): Cliente HTTP asíncrono.
        url (str): URL del endpoint /generate.
        prompt (str): Texto de entrada para la generación.

    Returns:
        Dict[str, Any]: Diccionario con el resultado de la petición.
    """
    payload = {"text": prompt}
    start_time = time.monotonic()
    try:
        response = await client.post(url, json=payload, timeout=60.0)
        response.raise_for_status()
        latency = time.monotonic() - start_time
        return {
            "status": "success",
            "latency_s": latency,
            "response_data": response.json()
        }
    except httpx.HTTPStatusError as e:
        latency = time.monotonic() - start_time
        logger.error(f"Error en la petición: {e.response.status_code} - {e.response.text}")
        return {"status": "http_error", "latency_s": latency, "status_code": e.response.status_code}
    except Exception as e:
        latency = time.monotonic() - start_time
        logger.error(f"Error inesperado en la petición: {e}")
        return {"status": "error", "latency_s": latency, "error_message": str(e)}

async def run_benchmark():
    """
    Función principal que ejecuta el benchmark completo.
    """
    server_config = get_server_config()
    base_url = f"http://{server_config['host']}:{server_config['port']}"
    generate_url = f"{base_url}/generate"

    logger.info("Iniciando benchmark...")
    logger.info(f"URL del servicio: {generate_url}")
    logger.info(f"Peticiones totales: {BENCHMARK_REQUESTS}")
    logger.info(f"Concurrencia: {BENCHMARK_CONCURRENCY}")

    # Esperar a que el servidor esté listo
    async with httpx.AsyncClient() as client:
        for _ in range(10):
            try:
                health_response = await client.get(f"{base_url}/health")
                if health_response.status_code == 200 and health_response.json().get("status") == "healthy":
                    logger.info("Servidor listo y saludable.")
                    break
            except httpx.RequestError:
                logger.info("Esperando al servidor...")
                await asyncio.sleep(2)
        else:
            logger.error("El servidor no está disponible. Abortando benchmark.")
            return

    # Ejecutar benchmark
    start_benchmark_time = time.monotonic()
    results = []

    async with httpx.AsyncClient() as client:
        tasks = []
        for i in range(BENCHMARK_REQUESTS):
            prompt = SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]
            tasks.append(send_request(client, generate_url, prompt))

            # Limitar concurrencia
            if len(tasks) >= BENCHMARK_CONCURRENCY:
                completed_tasks = await asyncio.gather(*tasks)
                results.extend(completed_tasks)
                tasks = []

        # Procesar tareas restantes
        if tasks:
            completed_tasks = await asyncio.gather(*tasks)
            results.extend(completed_tasks)

    total_benchmark_time = time.monotonic() - start_benchmark_time
    logger.info(f"Benchmark completado en {total_benchmark_time:.2f} segundos.")

    # Procesar y guardar resultados
    successful_requests = [r for r in results if r["status"] == "success"]
    failed_requests = [r for r in results if r["status"] != "success"]

    logger.info(f"Peticiones exitosas: {len(successful_requests)}")
    logger.info(f"Peticiones fallidas: {len(failed_requests)}")

    if successful_requests:
        latencies = [r["latency_s"] for r in successful_requests]
        stats = format_latency_stats(latencies)

        logger.info("--- Estadísticas de Latencia (ms) ---")
        logger.info(f"  - Peticiones: {stats['count']}")
        logger.info(f"  - Media: {stats['mean']:.2f} ms")
        logger.info(f"  - P50 (Mediana): {stats['p50']:.2f} ms")
        logger.info(f"  - P95: {stats['p95']:.2f} ms")
        logger.info(f"  - P99: {stats['p99']:.2f} ms")
        logger.info(f"  - Máxima: {stats['max']:.2f} ms")

        # Guardar en CSV
        df = pd.DataFrame(successful_requests)
        output_path = "benchmarks/bench_results.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Resultados detallados guardados en '{output_path}'")
    else:
        logger.warning("No hubo peticiones exitosas para generar estadísticas.")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
