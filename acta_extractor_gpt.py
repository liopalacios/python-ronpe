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
            Analiza esta imagen de un ACTA ELECTORAL de Perú (ONPE) y extrae EXACTAMENTE estos 6 valores:

            1. NÚMERO DE MESA: Busca junto a "Mesa de sufragio N°" - son 6 dígitos (ejemplo: 047291) en string porque hay mesas que empiezan con ceros, por ejemplo "047291" o "000123"
            2. VOTOS JP JUNTOS POR EL PERU: Número manuscrito junto al texto "JUNTOS POR EL PERU" o "JP"
            3. VOTOS FUERZA POPULAR: Número manuscrito junto al texto "FUERZA POPULAR" o "K"
            4. VOTOS EN BLANCO: Número manuscrito junto al texto "VOTOS EN BLANCO" alineado a su fila que le corresponda
            5. VOTOS NULOS: Número manuscrito junto al texto "VOTOS NULOS" alineado a su fila que le corresponda
            6. VOTOS INPUGNADOS: Número manuscrito junto al texto "VOTOS IMPUGNADOS" alineado a su fila que le corresponda
            7. TOTAL DE VOTOS EMITIDOS: Número manuscrito junto al texto "TOTAL DE VOTOS EMITIDOS" alineado a su fila que le corresponda

            RESPONDE SOLO CON JSON en este formato exacto  (sin texto adicional) como ejemplo el siguiente json:
            {
                "numero_mesa": "047291",
                "votos_jp": 150,
                "votos_k": 120,
                "votos_blanco": 5,
                "votos_nulos": 3,   
                "votos_inpugnados": 2,
                "total_votos_emitidos": 275
            }

            Si no puedes leer algún valor, ponlo como null, si la celda está vacía coloca valor cero.
            NO incluyas texto adicional, SOLO el JSON.
            - La imagen es de un acta electoral oficial, puede tener texto impreso y manuscrito, a veces tachaduras o manchas.
            - La imagen puede tener diferentes orientaciones, asegúrate de analizarla correctamente.
            - El número de mesa siempre tiene formato de 6 dígitos, los votos suelen ser números de 1 a 3 dígitos.
            
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
            print(f"✅ Datos extraídos: {resultado}")
            # Normalizar nombres de campos
            numero_mesa = resultado.get("numero_mesa")
            votos_c1 = resultado.get("votos_jp")
            votos_c2 = resultado.get("votos_k")
            votos_blanco = resultado.get("votos_blanco")
            votos_nulos = resultado.get("votos_nulos")  
            votos_inpugnados = resultado.get("votos_inpugnados")
            total_votos_emitidos = resultado.get("total_votos_emitidos")
            print(f"✅ Campos normalizados: numero_mesa={numero_mesa}, votos_c1={votos_c1}, votos_c2={votos_c2}, votos_blanco={votos_blanco}, votos_nulos={votos_nulos}, votos_inpugnados={votos_inpugnados}, total_votos_emitidos={total_votos_emitidos}")
            
           
            # Determinar qué campos faltan
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
                "method": "GPT-4V",
                "data": {
                    "numero_mesa": numero_mesa,
                    "votos_candidato_1": votos_c1,
                    "votos_candidato_2": votos_c2,
                    "votos_blanco": votos_blanco,
                    "votos_nulos": votos_nulos,
                    "votos_inpugnados": votos_inpugnados,
                    "total_votos_emitidos": total_votos_emitidos
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
