"""
Pruebas unitarias y de integración para el servidor FastAPI.

Este módulo contiene pruebas para los endpoints, el manejo de la cola
y el procesamiento en lotes del servidor de inferencia.
"""

import pytest
import asyncio
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from starlette.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock

# Es importante importar la app de esta manera para las pruebas
from src.server import app, lifespan

# Usamos pytest-asyncio para manejar pruebas asíncronas
pytestmark = pytest.mark.asyncio

@pytest_asyncio.fixture(scope="module")
def anyio_backend():
    """Define el backend de asyncio para pytest."""
    return "asyncio"

@pytest_asyncio.fixture(scope="module")
async def client():
    """
    Crea un cliente de prueba asíncrono para la aplicación FastAPI.
    Este fixture simula el ciclo de vida completo de la aplicación, permitiendo
    que el worker real se ejecute pero parcheando la función de generación.
    """
    # Mock para la función que realmente ejecuta el modelo
    async def mock_generate_batch(texts: list, sample_request):
        return [f"Respuesta para: {text}" for text in texts]

    with patch('src.server.ModelQuantizer') as MockModelQuantizer, \
         patch('src.server.generate_batch', new=mock_generate_batch):
        
        # Simular que el modelo se carga correctamente
        mock_quantizer_instance = MagicMock()
        mock_quantizer_instance.load_quantized_model.return_value = (MagicMock(), MagicMock())
        MockModelQuantizer.return_value = mock_quantizer_instance

        # El lifespan de la app iniciará el worker real, que llamará a nuestro
        # `mock_generate_batch` parcheado.
        with TestClient(app):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as test_client:
                yield test_client



class TestServerEndpoints:
    """Pruebas para los endpoints de la API."""

    async def test_root_endpoint(self, client: AsyncClient):
        """Prueba que el endpoint raíz '/' funciona correctamente."""
        response = await client.get("/")
        assert response.status_code == 200
        assert response.json() == {
            "service": "Servicio de Inferencia Optimizada",
            "version": "1.0.0",
            "status": "running"
        }

    async def test_health_check_endpoint(self, client: AsyncClient):
        """Prueba que el endpoint de salud '/health' funciona."""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    async def test_generate_text_success(self, client: AsyncClient):
        """Prueba una petición exitosa al endpoint '/generate'."""
        test_payload = {"text": "Hola, mundo"}
        response = await client.post("/generate", json=test_payload, timeout=10)

        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data
        assert data["input_text"] == "Hola, mundo"
        assert "generated_text" in data
        assert data["generated_text"] == "Respuesta para: Hola, mundo"
        assert "processing_time" in data

    async def test_generate_text_invalid_input(self, client: AsyncClient):
        """Prueba una petición a '/generate' con texto de entrada inválido."""
        test_payload = {"text": " "}
        response = await client.post("/generate", json=test_payload)
        assert response.status_code == 400
        assert response.json() == {"detail": "Texto de entrada inválido"}

    async def test_generate_text_invalid_max_length(self, client: AsyncClient):
        """Prueba una petición a '/generate' con max_length fuera de rango."""
        test_payload = {"text": "Texto válido", "max_length": 999}
        response = await client.post("/generate", json=test_payload)
        assert response.status_code == 400
        assert response.json() == {"detail": "max_length debe estar entre 1 y 200"}

    async def test_concurrent_requests(self, client: AsyncClient):
        """Prueba que el servidor maneja múltiples peticiones concurrentes."""
        tasks = []
        for i in range(5):
            payload = {"text": f"Petición concurrente {i}"}
            tasks.append(client.post("/generate", json=payload, timeout=20))

        responses = await asyncio.gather(*tasks)

        for i, response in enumerate(responses):
            assert response.status_code == 200
            data = response.json()
            assert data["input_text"] == f"Petición concurrente {i}"
            assert data["generated_text"] == f"Respuesta para: Petición concurrente {i}"

    async def test_queue_full_exception(self, client: AsyncClient):
        """Prueba que el servidor responde con 503 si la cola está llena."""
        # Usamos un mock para simular que la cola está llena
        with patch('src.server.request_queue.put', side_effect=asyncio.QueueFull):
            payload = {"text": "Esta petición fallará"}
            response = await client.post("/generate", json=payload)
            assert response.status_code == 503
            assert response.json() == {"detail": "Servicio ocupado, intente más tarde"}

    async def test_processing_timeout(self, client: AsyncClient):
        """Prueba que el servidor responde con 504 si el procesamiento excede el timeout."""
        # Mock para simular un timeout en la espera del future
        with patch('src.server.asyncio.wait_for', side_effect=asyncio.TimeoutError):
            payload = {"text": "Esta petición dará timeout"}
            response = await client.post("/generate", json=payload, timeout=10)
            assert response.status_code == 504
            assert response.json() == {"detail": "Timeout procesando la petición"}

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
