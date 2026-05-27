# acta_extractor_gpu.py
"""
Extractor de actas electorales - VERSIÓN GPU NVIDIA
Optimizado para máquinas con GPU NVIDIA CUDA
"""

import cv2
import numpy as np
from paddleocr import PaddleOCR  # Mejor para GPU
import re
import base64
from io import BytesIO
from PIL import Image
from typing import Dict, Optional, Tuple
from config import config

class ActaExtractorGPU:
    """
    Versión optimizada para GPU NVIDIA
    - Usa PaddleOCR (mejor rendimiento en GPU)
    - Mayor tamaño de imagen permitido
    - Mayor precisión
    """
    
    def __init__(self):
        print("🚀 Inicializando extractor en MODO GPU (NVIDIA CUDA)...")
        
        # PaddleOCR con GPU
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='es',
            use_gpu=True,           # ← Usar GPU
            gpu_mem=4000,           # 4GB de memoria GPU
            show_log=False,
            enable_mkldnn=False,    # No necesario con GPU
            cpu_threads=4,          # Backup para CPU
            ir_optim=True           # Optimización de inferencia
        )
        
        # Configuración GPU (imágenes más grandes)
        self.max_size = config.MAX_IMAGE_SIZE_GPU
        self.use_gpu = True
        
        # Patrones de búsqueda
        self.numero_mesa_pattern = re.compile(config.NUMERO_MESA_PATTERN)
        self.votos_pattern = re.compile(config.VOTOS_PATTERN)
        
        # Verificar GPU
        self._check_gpu()
        
        print("✅ Extractor GPU listo")
    
    def _check_gpu(self):
        """Verificar que GPU está disponible"""
        try:
            import paddle
            if paddle.is_compiled_with_cuda():
                print("   🎮 CUDA disponible")
                print(f"   📊 GPU: {paddle.device.get_device()}")
            else:
                print("   ⚠️ CUDA no disponible, cayendo a CPU")
        except:
            print("   ⚠️ No se pudo verificar CUDA")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocesamiento para GPU (más detalle)"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Redimensionar menos agresivo (GPU puede manejar más)
        h, w = gray.shape
        if max(h, w) > self.max_size:
            scale = self.max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # CLAHE con mejor contraste
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def extract_numero_mesa(self, ocr_result) -> Optional[int]:
        """Extraer número de mesa"""
        if not ocr_result or not ocr_result[0]:
            return None
        
        for line in ocr_result[0]:
            text = line[1][0].upper()
            if "MESA" in text:
                if "N°" in text or "Nº" in text:
                    match = self.numero_mesa_pattern.search(text)
                    if match:
                        return int(match.group(1))
        return None
    
    def extract_votos_candidatos(self, ocr_result) -> Tuple[Optional[int], Optional[int]]:
        """Extraer votos de candidatos"""
        if not ocr_result or not ocr_result[0]:
            return None, None
        
        votos_c1 = None
        votos_c2 = None
        
        lines = ocr_result[0]
        for idx, line in enumerate(lines):
            text = line[1][0].upper()
            
            if "JUNTOS POR EL PERU" in text or "JP" in text:
                for j in range(idx + 1, min(idx + 4, len(lines))):
                    next_text = lines[j][1][0]
                    match = self.votos_pattern.search(next_text)
                    if match:
                        votos_c1 = int(match.group(1))
                        break
            
            if "FUERZA POPULAR" in text or "K" in text:
                for j in range(idx + 1, min(idx + 4, len(lines))):
                    next_text = lines[j][1][0]
                    match = self.votos_pattern.search(next_text)
                    if match:
                        votos_c2 = int(match.group(1))
                        break
        
        return votos_c1, votos_c2
    
    def extract_from_base64(self, image_base64: str) -> Dict:
        """Método principal - extraer datos con GPU"""
        try:
            # Decodificar imagen
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_bytes))
            image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Preprocesar
            processed = self.preprocess_image(image_np)
            
            # OCR con PaddleOCR (GPU)
            result = self.ocr.ocr(processed, cls=True)
            
            if not result or not result[0]:
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
                "device": "GPU (NVIDIA CUDA)",
                "data": {
                    "numero_mesa": numero_mesa,
                    "votos_candidato_1": votos_c1,
                    "votos_candidato_2": votos_c2
                },
                "confidence": 95 if is_complete else 60,
                "missing_fields": missing,
                "message": "Acta procesada correctamente" if is_complete else f"Faltan: {', '.join(missing)}"
            }
            
        except Exception as e:
            print(f"❌ Error GPU: {e}")
            return self._empty_result(str(e))
    
    def _empty_result(self, error_msg: str) -> Dict:
        return {
            "success": False,
            "device": "GPU",
            "data": {
                "numero_mesa": None,
                "votos_candidato_1": None,
                "votos_candidato_2": None
            },
            "confidence": 0,
            "missing_fields": ["numero_mesa", "votos_candidato_1", "votos_candidato_2"],
            "message": error_msg
        }