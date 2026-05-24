# config.py
import os
import platform

class Config:
    """Configuración central del sistema"""
    
    # Detectar automáticamente si hay GPU NVIDIA
    @staticmethod
    def has_gpu():
        """Detectar si hay GPU NVIDIA disponible"""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi'], 
                capture_output=True, 
                text=True,
                shell=True
            )
            return result.returncode == 0
        except:
            return False
    
    @staticmethod
    def get_device():
        """Retorna 'gpu' o 'cpu' según disponibilidad"""
        return 'gpu' if Config.has_gpu() else 'cpu'
    
    # Configuración por defecto
    MAX_IMAGE_SIZE = 1200      # Tamaño máximo para CPU
    MAX_IMAGE_SIZE_GPU = 2000  # Tamaño máximo para GPU
    
    # Umbrales de confianza
    MIN_CONFIDENCE = 60
    MAX_VOTOS = 500
    MIN_VOTOS = 0
    
    # Patrones de búsqueda
    NUMERO_MESA_PATTERN = r'(\d{5,6})'
    VOTOS_PATTERN = r'\b(\d{1,3})\b'

# Instancia global
config = Config()