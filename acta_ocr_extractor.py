# acta_ocr_extractor.py
import cv2
import numpy as np
from paddleocr import PaddleOCR
import re
from typing import Dict, Optional, Tuple
import base64
from io import BytesIO
from PIL import Image

class ActaExtractor:
    """
     Versión mejorada para actas electorales ONPE
    """
    
    def __init__(self):
        print("🚀 Inicializando extractor en MODO CPU (mejorado)...")
        
        # EasyOCR con español
        self.reader = easyocr.Reader(
            ['es'], 
            gpu=False,
            verbose=False,
            model_storage_directory='./models'
        )
        
        # Configuración
        self.max_size = 1600  # Aumentado para mejor detalle
        
        # Patrones mejorados
        self.numero_mesa_patterns = [
            re.compile(r'(\d{6})'),      # 6 dígitos exactos (047291)
            re.compile(r'(\d{5,6})'),    # 5-6 dígitos
            re.compile(r'0(\d{5})')      # 0 seguido de 5 dígitos
        ]
        
        self.votos_pattern = re.compile(r'\b(\d{1,3})\b')
        
        print("✅ Extractor CPU mejorado listo")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocesamiento mejorado para detectar números pequeños"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Redimensionar más grande para mejor detalle
        h, w = gray.shape
        if max(h, w) > self.max_size:
            scale = self.max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Aumentar contraste
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Binarización adaptativa para resaltar números
        binary = cv2.adaptiveThreshold(
            enhanced, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 15, 2
        )
        
        return binary
    
    def extract_numero_mesa(self, ocr_result, image: np.ndarray) -> Optional[int]:
        """
        Extraer número de mesa - Múltiples estrategias
        """
        all_text = ""
        
        # Estrategia 1: Buscar línea específica "Mesa de sufragio N°"
        for (bbox, text, confidence) in ocr_result:
            text_upper = text.upper()
            all_text += " " + text
            
            if "MESA" in text_upper and ("N°" in text_upper or "Nº" in text_upper or "N" in text_upper):
                # Buscar números en esta línea
                for pattern in self.numero_mesa_patterns:
                    match = pattern.search(text)
                    if match:
                        num = int(match.group(1))
                        if 1000 <= num <= 999999:
                            return num
                
                # Buscar en las siguientes líneas (ampliar región)
                bbox_center = self._get_bbox_center(bbox)
                for (bbox2, text2, conf2) in ocr_result:
                    bbox2_center = self._get_bbox_center(bbox2)
                    # Si está cerca verticalmente (dentro de 100 píxeles)
                    if abs(bbox2_center[1] - bbox_center[1]) < 100:
                        for pattern in self.numero_mesa_patterns:
                            match = pattern.search(text2)
                            if match:
                                num = int(match.group(1))
                                if 1000 <= num <= 999999:
                                    return num
        
        # Estrategia 2: Buscar en toda la imagen por patrón de 6 dígitos
        for pattern in self.numero_mesa_patterns:
            matches = pattern.findall(all_text)
            for match in matches:
                num = int(match)
                if 1000 <= num <= 999999:
                    return num
        
        # Estrategia 3: Búsqueda regional en la imagen (donde típicamente está el número)
        numero_region = self._find_number_in_region(image, y_start=0.3, y_end=0.5)
        if numero_region and 1000 <= numero_region <= 999999:
            return numero_region
        
        return None
    
    def extract_votos_candidatos(self, ocr_result, image: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
        """
        Extraer votos manuscritos de candidatos
        Mejorado para detectar números escritos a mano
        """
        votos_c1 = None
        votos_c2 = None
        
        for i, (bbox, text, confidence) in enumerate(ocr_result):
            text_upper = text.upper()
            
            # Buscar "JUNTOS POR EL PERU" o "JP"
            if "JUNTOS POR EL PERU" in text_upper or "JP" in text_upper:
                # Buscar número en las siguientes 5 líneas (ampliado)
                for j in range(i + 1, min(i + 6, len(ocr_result))):
                    next_bbox, next_text, next_conf = ocr_result[j]
                    
                    # Buscar números en el texto
                    match = self.votos_pattern.search(next_text)
                    if match:
                        votos_c1 = int(match.group(1))
                        if 0 <= votos_c1 <= 500:
                            break
                
                # Si no encontró en texto, buscar en región de imagen
                if votos_c1 is None:
                    votos_c1 = self._find_number_in_region(image, bbox, offset_x=100, offset_y=-20, range_y=60)
            
            # Buscar "FUERZA POPULAR"
            if "FUERZA POPULAR" in text_upper:
                for j in range(i + 1, min(i + 6, len(ocr_result))):
                    next_bbox, next_text, next_conf = ocr_result[j]
                    match = self.votos_pattern.search(next_text)
                    if match:
                        votos_c2 = int(match.group(1))
                        if 0 <= votos_c2 <= 500:
                            break
                
                if votos_c2 is None:
                    votos_c2 = self._find_number_in_region(image, bbox, offset_x=100, offset_y=-20, range_y=60)
        
        return votos_c1, votos_c2
    
    def _find_number_in_region(self, image: np.ndarray, ref_bbox=None, 
                                y_start=0, y_end=1, x_start=0, x_end=1,
                                offset_x=0, offset_y=0, range_y=50) -> Optional[int]:
        """
        Buscar número en una región específica de la imagen
        """
        h, w = image.shape
        
        if ref_bbox:
            # Calcular región basada en bbox de referencia
            xs = [p[0] for p in ref_bbox]
            ys = [p[1] for p in ref_bbox]
            center_x = sum(xs) / 4
            center_y = sum(ys) / 4
            
            x1 = int(max(0, center_x + offset_x))
            x2 = int(min(w, center_x + offset_x + 150))
            y1 = int(max(0, center_y + offset_y))
            y2 = int(min(h, center_y + offset_y + range_y))
        else:
            # Usar porcentajes de la imagen
            x1 = int(w * x_start)
            x2 = int(w * x_end)
            y1 = int(h * y_start)
            y2 = int(h * y_end)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Recortar región
        region = image[y1:y2, x1:x2]
        
        if region.size == 0:
            return None
        
        # OCR en la región
        result = self.reader.readtext(region, paragraph=False)
        
        for (bbox, text, confidence) in result:
            match = self.votos_pattern.search(text)
            if match:
                return int(match.group(1))
        
        return None
    
    def _get_bbox_center(self, bbox) -> Tuple[float, float]:
        """Calcular centro del bounding box"""
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        return (sum(xs) / 4, sum(ys) / 4)
    
    def extract_from_base64(self, image_base64: str) -> Dict:
        """Método principal mejorado"""
        try:
            # Decodificar imagen
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_bytes))
            image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Preprocesar
            processed = self.preprocess_image(image_np)
            
            # OCR con EasyOCR (parámetros mejorados)
            result = self.reader.readtext(
                processed,
                paragraph=False,
                width_ths=0.5,      # Más sensible
                height_ths=0.5,
                decoder='beamsearch',  # Mejor para números
                beamWidth=5
            )
            
            if not result:
                return self._empty_result("No se pudo leer la imagen")
            
            # Extraer datos con métodos mejorados
            numero_mesa = self.extract_numero_mesa(result, processed)
            votos_c1, votos_c2 = self.extract_votos_candidatos(result, processed)
            
            # Validar
            missing = []
            if numero_mesa is None:
                missing.append("número de mesa")
            else:
                # Validar que el número tenga sentido (5-6 dígitos)
                if len(str(numero_mesa)) < 5:
                    numero_mesa = None
                    missing.append("número de mesa (formato inválido)")
            
            if votos_c1 is None:
                missing.append("votos Juntos por el Perú")
            if votos_c2 is None:
                missing.append("votos Fuerza Popular")
            
            is_complete = len(missing) == 0
            
            # Calcular confianza
            confidence = 90 if is_complete else 50
            
            return {
                "success": is_complete,
                "device": "CPU",
                "data": {
                    "numero_mesa": numero_mesa,
                    "votos_candidato_1": votos_c1,
                    "votos_candidato_2": votos_c2
                },
                "confidence": confidence,
                "missing_fields": missing,
                "message": "Acta procesada correctamente" if is_complete else f"Faltan: {', '.join(missing)}"
            }
            
        except Exception as e:
            print(f"❌ Error CPU: {e}")
            import traceback
            traceback.print_exc()
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


# Preprocesamiento extra para manuscrito (lapicero)
def enhance_handwriting(image: np.ndarray) -> np.ndarray:
    """Mejora específica para números escritos con lapicero"""
    
    # 1. Aumentar saturación de tinta
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=1)
    
    # 2. Eliminar ruido
    denoised = cv2.fastNlMeansDenoising(dilated, h=10)
    
    # 3. Aplicar umbral adaptativo
    binary = cv2.adaptiveThreshold(
        denoised, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    return binary

# Instancia global
extractor = ActaExtractor()