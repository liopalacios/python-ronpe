# acta_extractor_cpu.py
"""
Extractor de actas electorales - VERSIÓN CPU
Optimizado para máquinas sin GPU NVIDIA
"""

import cv2
import numpy as np
import easyocr
import re
import base64
from io import BytesIO
from PIL import Image
from typing import Dict, Optional, Tuple
from config import config

class ActaExtractorCPU:
    """
    Versión optimizada para CPU
    - Menor consumo de memoria
    - Imágenes redimensionadas agresivamente
    - OCR con parámetros más rápidos
    """
    
    def __init__(self):
        print("🚀 Inicializando extractor en MODO CPU...")
        
        # EasyOCR forzado a CPU
        self.reader = easyocr.Reader(
            ['es'], 
            gpu=False,          # ← Forzar CPU
            verbose=False,      # Silencioso
            model_storage_directory='./models'  # Cache de modelos
        )
        
        # Configuración CPU
        self.max_size = config.MAX_IMAGE_SIZE
        self.use_gpu = False
        
        # Patrones de búsqueda
        self.numero_mesa_pattern = re.compile(config.NUMERO_MESA_PATTERN)
        self.votos_pattern = re.compile(config.VOTOS_PATTERN)
        
        print("✅ Extractor CPU listo")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocesamiento optimizado para CPU"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Redimensionar más agresivamente (ahorra tiempo CPU)
        h, w = gray.shape
        if max(h, w) > self.max_size:
            scale = self.max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # CLAHE con parámetros ligeros
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def extract_numero_mesa(self, ocr_result) -> Optional[int]:
        """Extraer número de mesa impreso"""
        for (bbox, text, confidence) in ocr_result:
            text_upper = text.upper()
            if "MESA" in text_upper:
                if "N°" in text_upper or "Nº" in text_upper:
                    match = self.numero_mesa_pattern.search(text)
                    if match:
                        return int(match.group(1))
        
        # Segunda pasada: buscar solo números grandes
        for (bbox, text, confidence) in ocr_result:
            match = self.numero_mesa_pattern.search(text)
            if match:
                num = int(match.group(1))
                if 1000 <= num <= 99999:
                    return num
        return None
    
    def extract_votos_candidatos(self, ocr_result) -> Tuple[Optional[int], Optional[int]]:
        """Extraer votos de candidatos (manuscrito)"""
        votos_c1 = None
        votos_c2 = None
        
        for i, (bbox, text, confidence) in enumerate(ocr_result):
            text_upper = text.upper()
            
            # Juntos por el Perú
            if "JUNTOS POR EL PERU" in text_upper or "JP" in text_upper:
                for j in range(i + 1, min(i + 4, len(ocr_result))):
                    next_text = ocr_result[j][1]
                    match = self.votos_pattern.search(next_text)
                    if match:
                        votos_c1 = int(match.group(1))
                        break
            
            # Fuerza Popular
            if "FUERZA POPULAR" in text_upper or "K" in text_upper:
                for j in range(i + 1, min(i + 4, len(ocr_result))):
                    next_text = ocr_result[j][1]
                    match = self.votos_pattern.search(next_text)
                    if match:
                        votos_c2 = int(match.group(1))
                        break
        
        return votos_c1, votos_c2
    
    def extract_from_base64(self, image_base64: str) -> Dict:
        """Método principal - extraer datos de imagen base64"""
        try:
            # Decodificar imagen
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_bytes))
            image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Preprocesar
            processed = self.preprocess_image(image_np)
            
            # OCR (configurado para CPU)
            result = self.reader.readtext(
                processed,
                paragraph=False,
                width_ths=0.7,
                height_ths=0.7,
                decoder='greedy'  # Más rápido que beamsearch
            )
            
            if not result:
                return self._empty_result("No se pudo leer la imagen")
            
            # Extraer datos
            numero_mesa = self.extract_numero_mesa(result)
            votos_c1, votos_c2 = self.extract_votos_candidatos(result)
            
            # Validar
            missing = []
            if numero_mesa is None:
                missing.append("número de mesa")
            if votos_c1 is None:
                missing.append("votos Juntos por el Perú")
            if votos_c2 is None:
                missing.append("votos Fuerza Popular")
            
            is_complete = len(missing) == 0
            
            return {
                "success": is_complete,
                "device": "CPU",
                "data": {
                    "numero_mesa": numero_mesa,
                    "votos_candidato_1": votos_c1,
                    "votos_candidato_2": votos_c2
                },
                "confidence": 85 if is_complete else 50,
                "missing_fields": missing,
                "message": "Acta procesada correctamente" if is_complete else f"Faltan: {', '.join(missing)}"
            }
            
        except Exception as e:
            print(f"❌ Error CPU: {e}")
            return self._empty_result(str(e))
    
    def _empty_result(self, error_msg: str) -> Dict:
        return {
            "success": False,
            "device": "CPU",
            "data": {
                "numero_mesa": None,
                "votos_candidato_1": None,
                "votos_candidato_2": None
            },
            "confidence": 0,
            "missing_fields": ["numero_mesa", "votos_candidato_1", "votos_candidato_2"],
            "message": error_msg
        }