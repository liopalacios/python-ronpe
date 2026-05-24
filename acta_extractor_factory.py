# acta_extractor_factory.py
"""
Fábrica de extractores - Selecciona automáticamente CPU o GPU
"""

from config import config

# Variable global para el extractor
_extractor = None

def get_extractor():
    """
    Retorna el extractor adecuado según el hardware disponible.
    Singleton pattern - solo se inicializa una vez.
    """
    global _extractor
    
    if _extractor is None:
        device = config.get_device()
        
        if device == 'gpu':
            try:
                from acta_extractor_gpu import ActaExtractorGPU
                _extractor = ActaExtractorGPU()
                print("✅ Usando extractor GPU (NVIDIA CUDA)")
            except Exception as e:
                print(f"⚠️ Error cargando GPU: {e}")
                print("🔄 Fallback a CPU...")
                from acta_extractor_cpu import ActaExtractorCPU
                _extractor = ActaExtractorCPU()
        else:
            from acta_extractor_cpu import ActaExtractorCPU
            _extractor = ActaExtractorCPU()
            print("✅ Usando extractor CPU")
    
    return _extractor

# Para importar directamente
extractor = get_extractor()