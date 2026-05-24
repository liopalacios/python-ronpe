# acta_extractor_gpt.py
"""
Extractor de actas electorales usando GPT-4V
"""

import base64
import json
import re
from typing import Dict, Optional
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()




class ActaExtractorGPT:
    """
    Extractor usando GPT-4V - Versión simplificada
    """
    
    def __init__(self):
        print("🚀 Inicializando extractor GPT-4V...")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY no encontrada en .env")
        
        self.client = OpenAI(api_key=api_key)
        
        # Prompt especializado para actas electorales
        self.prompt = """
            Analiza esta imagen de un ACTA ELECTORAL de Perú (ONPE) y extrae EXACTAMENTE estos 3 valores:

            1. NÚMERO DE MESA: Busca junto a "Mesa de sufragio N°" - son 6 dígitos (ejemplo: 047291)
            2. VOTOS PERÚ LIBRE: Número manuscrito junto al texto "PERU LIBRE" o "PERÚ LIBRE"
            3. VOTOS FUERZA POPULAR: Número manuscrito junto al texto "FUERZA POPULAR"

            RESPONDE SOLO CON JSON en este formato exacto:
            {
                "numero_mesa": 47291,
                "votos_peru_libre": 150,
                "votos_fuerza_popular": 120
            }

            Si no puedes leer algún valor, ponlo como null.
            NO incluyas texto adicional, SOLO el JSON.
            """
        
        print("✅ Extractor GPT-4V listo")
    
    def extract_from_base64(self, image_base64: str) -> Dict:
        """
        Extraer datos de la imagen usando GPT-4V
        """
        try:
            # Limpiar base64 si tiene prefijo
            if "," in image_base64:
                image_base64 = image_base64.split(",")[1]
            
            # Llamar a GPT-4V
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Usa "gpt-4o" para mejor precisión
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.1
            )
            print(response)
            
            # Obtener respuesta
            respuesta_texto = response.choices[0].message.content
            
            # Limpiar y parsear JSON
            respuesta_texto = respuesta_texto.strip()
            if respuesta_texto.startswith("```json"):
                respuesta_texto = respuesta_texto[7:]
            if respuesta_texto.startswith("```"):
                respuesta_texto = respuesta_texto[3:]
            if respuesta_texto.endswith("```"):
                respuesta_texto = respuesta_texto[:-3]
            
            resultado = json.loads(respuesta_texto)
            
            # Normalizar nombres de campos
            numero_mesa = resultado.get("numero_mesa")
            votos_c1 = resultado.get("votos_peru_libre")
            votos_c2 = resultado.get("votos_fuerza_popular")
            
            # Validar rangos
            if numero_mesa and not (1000 <= numero_mesa <= 999999):
                numero_mesa = None
            if votos_c1 and not (0 <= votos_c1 <= 500):
                votos_c1 = None
            if votos_c2 and not (0 <= votos_c2 <= 500):
                votos_c2 = None
            
            # Determinar qué campos faltan
            missing = []
            if numero_mesa is None:
                missing.append("número de mesa")
            if votos_c1 is None:
                missing.append("votos Perú Libre")
            if votos_c2 is None:
                missing.append("votos Fuerza Popular")
            
            is_complete = len(missing) == 0
            
            return {
                "success": is_complete,
                "method": "GPT-4V",
                "data": {
                    "numero_mesa": numero_mesa,
                    "votos_candidato_1": votos_c1,
                    "votos_candidato_2": votos_c2
                },
                "missing_fields": missing,
                "raw_response": resultado
            }
            
        except json.JSONDecodeError as e:
            print(f"❌ Error parseando JSON: {e}")
            print(f"Respuesta cruda: {respuesta_texto}")
            return self._empty_result("Error en formato de respuesta")
            
        except Exception as e:
            print(f"❌ Error GPT-4V: {e}")
            return self._empty_result(str(e))
    
    def _empty_result(self, error_msg: str) -> Dict:
        return {
            "success": False,
            "method": "GPT-4V",
            "data": {
                "numero_mesa": None,
                "votos_candidato_1": None,
                "votos_candidato_2": None
            },
            "missing_fields": ["numero_mesa", "votos_candidato_1", "votos_candidato_2"],
            "error": error_msg
        }


# Instancia global
extractor = ActaExtractorGPT()
