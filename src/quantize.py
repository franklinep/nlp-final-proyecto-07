"""
Pipeline de cuantización para optimizar modelos de lenguaje.

Este módulo implementa la cuantización de modelos usando bitsandbytes
para reducir el uso de memoria y mejorar la velocidad de inferencia.
"""

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig
)
from typing import Tuple, Dict, Any, Optional
import gc

from .config import get_model_config, get_generation_config
from .utils import setup_logging, measure_time, create_directory_if_not_exists

logger = setup_logging(__name__)

class ModelQuantizer:
    """
    Clase para manejar la cuantización y carga de modelos de lenguaje.
    """

    def __init__(self):
        """Inicializa el cuantizador con la configuración por defecto."""
        self.config = get_model_config()
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.generation_config: Optional[GenerationConfig] = None

    @measure_time
    def load_and_quantize_model(self) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """
        Carga y cuantiza el modelo especificado en la configuración.

        Returns:
            Tuple[AutoTokenizer, AutoModelForCausalLM]: Tokenizer y modelo cuantizado

        Raises:
            Exception: Si hay errores durante la carga o cuantización
        """
        try:
            logger.info(f"Cargando tokenizer para {self.config['model_name']}")

            # Cargar tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['model_name'],
                padding_side="left",
                trust_remote_code=True
            )

            # Configurar pad_token si no existe
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Configuración de cuantización
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True if self.config['quantization_bits'] == 8 else False,
                load_in_4bit=True if self.config['quantization_bits'] == 4 else False,
                bnb_4bit_quant_type="nf4" if self.config['quantization_bits'] == 4 else None,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True if self.config['quantization_bits'] == 4 else False,
            )

            logger.info(f"Cuantizando modelo a {self.config['quantization_bits']} bits")

            # Cargar modelo cuantizado
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config['model_name'],
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )

            # Configurar modo de evaluación
            self.model.eval()

            # Configurar generación
            gen_config = get_generation_config()
            self.generation_config = GenerationConfig(
                max_length=gen_config['max_length'],
                temperature=gen_config['temperature'],
                top_p=gen_config['top_p'],
                do_sample=gen_config['do_sample'],
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            logger.info("Modelo cargado y cuantizado exitosamente")
            return self.tokenizer, self.model

        except Exception as e:
            logger.error(f"Error durante la cuantización: {e}")
            raise

    @measure_time
    def save_quantized_model(self, save_path: str) -> None:
        """
        Guarda el modelo cuantizado en disco.

        Args:
            save_path (str): Ruta donde guardar el modelo

        Raises:
            ValueError: Si el modelo no está cargado
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("El modelo debe estar cargado antes de guardarlo")

        create_directory_if_not_exists(save_path)

        try:
            logger.info(f"Guardando modelo cuantizado en {save_path}")

            # Guardar modelo
            self.model.save_pretrained(save_path)

            # Guardar tokenizer
            self.tokenizer.save_pretrained(save_path)

            logger.info("Modelo guardado exitosamente")

        except Exception as e:
            logger.error(f"Error guardando modelo: {e}")
            raise

    @measure_time
    def load_quantized_model(self, load_path: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """
        Carga un modelo cuantizado desde disco.

        Args:
            load_path (str): Ruta del modelo guardado

        Returns:
            Tuple[AutoTokenizer, AutoModelForCausalLM]: Tokenizer y modelo cargado
        """
        try:
            logger.info(f"Cargando modelo cuantizado desde {load_path}")

            # Cargar tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                load_path,
                padding_side="left"
            )

            # Configurar pad_token si no existe
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Cargar modelo
            self.model = AutoModelForCausalLM.from_pretrained(
                load_path,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )

            self.model.eval()

            # Configurar generación
            gen_config = get_generation_config()
            self.generation_config = GenerationConfig(
                max_length=gen_config['max_length'],
                temperature=gen_config['temperature'],
                top_p=gen_config['top_p'],
                do_sample=gen_config['do_sample'],
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            logger.info("Modelo cargado exitosamente")
            return self.tokenizer, self.model

        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise

    def validate_model_precision(self, test_text: str = "El futuro de la inteligencia artificial") -> Dict[str, Any]:
        """
        Valida la precisión del modelo cuantizado comparándolo con texto de prueba.

        Args:
            test_text (str): Texto de prueba para validación

        Returns:
            Dict[str, Any]: Resultados de la validación
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("El modelo debe estar cargado antes de validar")

        try:
            logger.info("Validando precisión del modelo cuantizado")

            # Tokenizar entrada
            inputs = self.tokenizer(
                test_text,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.model.device)

            # Generar texto
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    max_new_tokens=20
                )

            # Decodificar salida
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            # Calcular estadísticas básicas
            input_length = len(inputs['input_ids'][0])
            output_length = len(outputs[0])
            new_tokens = output_length - input_length

            validation_result = {
                "input_text": test_text,
                "generated_text": generated_text,
                "input_tokens": input_length,
                "output_tokens": output_length,
                "new_tokens": new_tokens,
                "validation_passed": len(generated_text) > len(test_text)
            }

            logger.info(f"Validación completada: {validation_result['validation_passed']}")
            return validation_result

        except Exception as e:
            logger.error(f"Error durante la validación: {e}")
            return {
                "validation_passed": False,
                "error": str(e)
            }

    def cleanup(self) -> None:
        """Limpia recursos del modelo para liberar memoria."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Limpiar cache de GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Recolección de basura
        gc.collect()
        logger.info("Recursos del modelo liberados")

def main():
    """
    Función principal para ejecutar el pipeline de cuantización.
    """
    logger.info("Iniciando pipeline de cuantización")

    quantizer = ModelQuantizer()

    try:
        # Cargar y cuantizar modelo
        tokenizer, model = quantizer.load_and_quantize_model()

        # Validar precisión
        validation_result = quantizer.validate_model_precision()
        logger.info(f"Resultado de validación: {validation_result}")

        # Guardar modelo cuantizado
        config = get_model_config()
        quantizer.save_quantized_model(config['quantized_path'])

        logger.info("Pipeline de cuantización completado exitosamente")

    except Exception as e:
        logger.error(f"Error en el pipeline: {e}")
        raise
    finally:
        # Limpiar recursos
        quantizer.cleanup()

if __name__ == "__main__":
    main()
