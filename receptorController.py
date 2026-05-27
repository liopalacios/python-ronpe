from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import os
import psycopg2
from psycopg2.extras import RealDictCursor
import redis
import boto3
from minio import Minio
import base64
from io import BytesIO
from PIL import Image
import json
import re
from typing import Optional, Dict, Any
import uuid
from datetime import datetime
load_dotenv()
from acta_extractor_gpt import extractor
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()

# ============ CONEXIONES A BASE DE DATOS ============

# PostgreSQL
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME", "whatsapp_evidencias"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "")
    )

class WhatsAppMessage(BaseModel):
    id: str
    account_id: int
    phone_number: str
    sender: str
    chat_id: str
    timestamp: int
    type: str  # "text" o "image"
    text: Optional[str] = None
    image_base64: Optional[str] = None
    caption: Optional[str] = None
    mimetype: Optional[str] = None
    size: Optional[int] = None

class TestImage(BaseModel):
    image_base64: str

@app.post("/test-gpt")
async def test_gpt_only(test: TestImage):
    """
    Endpoint de prueba - Solo para verificar que GPT funciona
    """
    print("🧪 Probando GPT-4V...")
    
    resultado = extractor.extract_from_base64(test.image_base64)
    
    return resultado

# Redis
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True
)

# MinIO
minio_client = Minio(
    os.getenv("MINIO_HOST", "localhost:9000"),
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY"),
    secure=False
)

# Crear bucket si no existe
bucket_name = os.getenv("MINIO_BUCKET", "whatsapp-evidencias")
if not minio_client.bucket_exists(bucket_name):
    minio_client.make_bucket(bucket_name)

# ============ MODELOS ============

class WhatsAppMessage(BaseModel):
    id: str
    account_id: int
    phone_number: str
    sender: str
    chat_id: str
    timestamp: int
    type: str  # "text" o "image"
    text: Optional[str] = None
    image_base64: Optional[str] = None
    caption: Optional[str] = None
    mimetype: Optional[str] = None
    size: Optional[int] = None

class ImagenData(BaseModel):
    nro: Optional[str] = None
    cantidad_candidato_1: Optional[int] = None
    cantidad_candidato_2: Optional[int] = None
    dni_presidente_mesa: Optional[str] = None
    completo: bool = False
    faltantes: list = []

# ============ FUNCIONES DE IA ============

async def extraer_datos_imagen(image_base64: str) -> ImagenData:
    """Envía imagen a OpenAI Vision para extraer datos"""
    
    prompt = """
    Analiza esta imagen de un acta electoral y extrae los siguientes datos:
    
    1. Número de acta (nro)
    2. Cantidad de votos candidato 1 (cantidad_candidato_1)
    3. Cantidad de votos candidato 2 (cantidad_candidato_2)
    4. DNI del presidente de mesa (dni_presidente_mesa)
    
    Responde SOLO con JSON en este formato:
    {
        "nro": "string o null",
        "cantidad_candidato_1": number o null,
        "cantidad_candidato_2": number o null,
        "dni_presidente_mesa": "string o null"
    }
    
    Si algún campo no es visible o legible, ponlo como null.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": [
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
        print("✅ Respuesta de IA recibida")
        print(response )
        # Parsear respuesta JSON
        result = json.loads(response.choices[0].message.content)
        
        # Validar datos extraídos
        datos = ImagenData(
            nro=result.get("nro"),
            cantidad_candidato_1=result.get("cantidad_candidato_1"),
            cantidad_candidato_2=result.get("cantidad_candidato_2"),
            dni_presidente_mesa=result.get("dni_presidente_mesa")
        )
        
        # Verificar qué campos están completos
        faltantes = []
        if not datos.nro:
            faltantes.append("número de acta")
        if datos.cantidad_candidato_1 is None:
            faltantes.append("votos candidato 1")
        if datos.cantidad_candidato_2 is None:
            faltantes.append("votos candidato 2")
        if not datos.dni_presidente_mesa:
            faltantes.append("DNI presidente de mesa")
        
        datos.completo = len(faltantes) == 0
        datos.faltantes = faltantes
        
        return datos
        
    except Exception as e:
        print(f"❌ Error en IA: {e}")
        return ImagenData(completo=False, faltantes=["error al procesar imagen"])

# ============ FUNCIONES DE ALMACENAMIENTO ============

async def guardar_imagen_minio(image_base64: str, message_id: str, tipo: str = "original") -> str:
    """Guarda imagen en MinIO y retorna la ruta"""
    try:
        # Decodificar base64 a bytes
        image_bytes = base64.b64decode(image_base64)
        
        # Determinar carpeta según tipo
        folder = "exitosos" if tipo == "original" else "reenviados"
        object_name = f"{folder}/{message_id}.jpg"
        
        # Subir a MinIO
        minio_client.put_object(
            bucket_name,
            object_name,
            BytesIO(image_bytes),
            len(image_bytes),
            content_type="image/jpeg"
        )
        
        return object_name
        
    except Exception as e:
        print(f"❌ Error guardando en MinIO: {e}")
        raise

async def guardar_registro_db(
    message_id: str, 
    sender: str, 
    nro_mesa: int,           # ← Añadir este parámetro
    votos_c1: int,           # ← Añadir este parámetro
    votos_c2: int,           # ← Añadir este parámetro
    votos_blanco: int,       # ← Añadir este parámetro
    votos_nulos: int,        # ← Añadir este parámetro
    votos_inpugnados: int,   # ← Añadir este parámetro
    minio_path: str,
    es_reenvio: bool = False,
    dni_presidente_mesa: Optional[str] = None
):
    """Guarda el registro en PostgreSQL"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        if es_reenvio:
            # Tabla de reenviados
            cur.execute("""
                INSERT INTO reenviados (
                    message_id, sender, nro, votos_candidato_1, votos_candidato_2,
                    votos_blanco, votos_nulos, votos_inpugnados, dni_presidente, minio_path, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                message_id, sender, nro_mesa, votos_c1,
                votos_c2, votos_blanco, votos_nulos, votos_inpugnados, dni_presidente_mesa,
                minio_path, datetime.now()
            ))
        else:
            # Tabla principal
            cur.execute("""
                INSERT INTO evidencias (
                    message_id, sender, nro, votos_candidato_1, votos_candidato_2,
                    votos_blancos, votos_nulos, votos_inpugnados, dni_presidente, minio_path, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                message_id, sender, nro_mesa, votos_c1,
                votos_c2, votos_blanco, votos_nulos, votos_inpugnados, dni_presidente_mesa,
                minio_path, datetime.now()
            ))
        
        conn.commit()
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Error guardando en DB: {e}")
        raise
    finally:
        cur.close()
        conn.close()

# ============ VERIFICACIÓN EN REDIS ============

async def verificar_sender_en_redis(sender: str) -> bool:
    """Verifica si el sender ya tiene un registro exitoso"""
    key = f"registro_exitoso:{sender}"
    return redis_client.exists(key) > 0

async def marcar_registro_exitoso(sender: str, nro: str):
    """Marca en Redis que este sender ya registró exitosamente"""
    key = f"registro_exitoso:{sender}"
    redis_client.setex(key, 86400 * 30, nro)  # Expira en 30 días

# ============ ENDPOINT PRINCIPAL ============

@app.post("/whatsapp/message")
async def process_whatsapp_message(msg: WhatsAppMessage):
    """
     Endpoint principal que recibe mensajes desde Node.js
    """
    print(f"📩 Mensaje recibido de {msg.sender} - Tipo: {msg.type}")
    print(f"📸 Imagen recibida: {bool(msg.image_base64)} - Texto: {msg.text is not None}")
    
    # 1. Verificar si es imagen
    if msg.type != "image" or not msg.image_base64:
        return {
            "reply": "📸 Por favor, envía una foto del acta electoral para procesar la evidencia."
        }
    
    # 2. Verificar si este sender ya registró exitosamente antes
    ##ya_registrado = await verificar_sender_en_redis(msg.sender)
    ya_registrado = False
    if ya_registrado:
        print(f"♻️ Sender {msg.sender} ya registrado - guardando como reenvío")
        
        try:
            minio_path = await guardar_imagen_minio(
                msg.image_base64, 
                msg.id, 
                tipo="reenviado"
            )
            
            #await guardar_registro_reenvio(
             #   message_id=msg.id,
              #  sender=msg.sender,
               # minio_path=minio_path
            #)
            
            return {
                "reply": "✅ Evidencia adicional recibida correctamente. Gracias."
            }
            
        except Exception as e:
            print(f"❌ Error guardando reenvío: {e}")
            return {
                "reply": "⚠️ Error al guardar la imagen. Por favor intenta nuevamente."
            }
    
    # 3. Primera vez - procesar con OCR
    print("🤖 Procesando imagen con GPT...")
    
    resultado = extractor.extract_from_base64(msg.image_base64)
    
    # DEBUG: Ver estructura exacta
    print(f"🔍 DEBUG - Resultado: {resultado}")
    
    # 4. Verificar si la extracción fue exitosa
    if not resultado.get("success", False):
        missing = resultado.get("missing_fields", ["datos no disponibles"])
        faltantes_texto = ", ".join(missing)
        
        try:
            await guardar_imagen_minio(
                msg.image_base64, 
                msg.id, 
                tipo="incompleta"
            )

             # Mostrar qué se detectó
            mensaje_detalle = []
            if datos.get("numero_mesa"):
                mensaje_detalle.append(f"✓ Mesa: {datos['numero_mesa']}")
            else:
                mensaje_detalle.append("✗ Mesa: no detectada")
            
            if datos.get("votos_candidato_1"):
                mensaje_detalle.append(f"✓ Juntos por el Perú: {datos['votos_candidato_1']}")
            else:
                mensaje_detalle.append("✗ Juntos por el Perú: no detectado")
            
            if datos.get("votos_candidato_2"):
                mensaje_detalle.append(f"✓ Fuerza Popular: {datos['votos_candidato_2']}")
            else:
                mensaje_detalle.append("✗ Fuerza Popular: no detectado")

            if datos.get("votos_blanco"):
                mensaje_detalle.append(f"✓ Votos en Blanco: {datos['votos_blanco']}")
            else:
                mensaje_detalle.append("✗ Votos en Blanco: no detectados")

            if datos.get("votos_nulos"):
                mensaje_detalle.append(f"✓ Votos Nulos: {datos['votos_nulos']}")
            else:
                mensaje_detalle.append("✗ Votos Nulos: no detectados")

            if datos.get("votos_inpugnados"):
                mensaje_detalle.append(f"✓ Votos Inpugnados: {datos['votos_inpugnados']}")
            else:
                mensaje_detalle.append("✗ Votos Inpugnados: no detectados")

            print(f"📊 Detección: {' | '.join(mensaje_detalle)}")
            
            return {
                "reply": f"📸 Datos detectados:\n" + "\n".join(mensaje_detalle) +
                        f"\n\nPor favor, asegúrate que la foto sea más clara y que se vean bien los números escritos a mano."
            }
        except:
            pass
        
        return {
            "reply": f"📸 No se pudo leer correctamente: {faltantes_texto}. Por favor, envía una foto más clara."
        }
    
    # 5. Obtener los datos (verificar que exista 'data')
    datos = resultado.get("data", {})
    
    if not datos:
        return {
            "reply": "📸 No se pudieron extraer los datos de la imagen. Por favor, envía una foto más clara."
        }
    
    # 6. Extraer valores con get() para evitar KeyError
    numero_mesa = datos.get("numero_mesa")
    votos_c1 = datos.get("votos_candidato_1")
    votos_c2 = datos.get("votos_candidato_2")
    votos_blanco = datos.get("votos_blanco")
    votos_nulos = datos.get("votos_nulos")
    votos_inpugnados = datos.get("votos_inpugnados")
    
    # También soportar nombres alternativos (por si acaso)
    if numero_mesa is None:
        numero_mesa = datos.get("nro") or datos.get("numero_acta")
    
    print(f"✅ Datos extraídos: nro_mesa={numero_mesa}, "
          f"votos_candidato_1={votos_c1}, votos_candidato_2={votos_c2}, "
          f"votos_blanco={votos_blanco}, votos_nulos={votos_nulos}, "
          f"votos_inpugnados={votos_inpugnados}")
    
    # 7. Validar que todos los datos están presentes
    missing_fields = []
    if numero_mesa is None:
        missing_fields.append("número de mesa")
    if votos_c1 is None:
        missing_fields.append("votos Juntos por el Perú")
    if votos_c2 is None:
        missing_fields.append("votos Fuerza Popular")
    if votos_blanco is None:
        missing_fields.append("votos en Blanco")
    if votos_nulos is None:
        missing_fields.append("votos Nulos")
    if votos_inpugnados is None:
        missing_fields.append("votos Inpugnados")
    
    if missing_fields:
        return {
            "reply": f"📸 No se pudieron leer: {', '.join(missing_fields)}. Por favor, envía una foto más clara."
        }
    
    # 8. Guardar todo
    try:
        # Guardar imagen en MinIO
        minio_path = await guardar_imagen_minio(
            msg.image_base64, 
            msg.id, 
            tipo="original"
        )
        
        # Guardar registro en PostgreSQL
        await guardar_registro_db(
            message_id=msg.id,
            sender=msg.sender,
            nro_mesa=numero_mesa,
            votos_c1=votos_c1,
            votos_c2=votos_c2,
            votos_blanco=votos_blanco,
            votos_nulos=votos_nulos,
            votos_inpugnados=votos_inpugnados,
            minio_path=minio_path
        )
        
        # Marcar en Redis como registrado exitosamente
        await marcar_registro_exitoso(msg.sender, numero_mesa)
        
        # Respuesta aleatoria de éxito
        respuestas_exito = [
            f"Por favor confirma la 📋 Acta N° {numero_mesa}  "+ "🖨️",
            
        ]
        
        import random
        reply = random.choice(respuestas_exito)
        
        return {"reply": reply}
        
    except Exception as e:
        print(f"❌ Error guardando registro: {e}")
        import traceback
        traceback.print_exc()
        return {
            "reply": "⚠️ Error interno al guardar la evidencia. Por favor intenta nuevamente."
        }

# ============ ENDPOINTS DE CONSULTA ============

@app.get("/stats/sender/{sender}")
async def consultar_sender(sender: str):
    """Consultar si un sender ya registró exitosamente"""
    registrado = await verificar_sender_en_redis(sender)
    
    if registrado:
        nro = redis_client.get(f"registro_exitoso:{sender}")
        return {"registrado": True, "nro_acta": nro}
    else:
        return {"registrado": False}

@app.get("/stats/totales")
async def stats_totales():
    """Estadísticas generales"""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Total de evidencias exitosas
        cur.execute("SELECT COUNT(*) as total FROM evidencias")
        exitosos = cur.fetchone()["total"]
        
        # Total de reenvíos
        cur.execute("SELECT COUNT(*) as total FROM reenviados")
        reenviados = cur.fetchone()["total"]
        
        return {
            "evidencias_exitosas": exitosos,
            "reenvios": reenviados,
            "total": exitosos + reenviados
        }
    finally:
        cur.close()
        conn.close()

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "method": "GPT-4V",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY"))
    }

# ============ SCRIPT PARA CREAR TABLAS ============
@app.on_event("startup")
async def startup():
    """Crear tablas si no existen"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Tabla de evidencias exitosas
        cur.execute("""
            CREATE TABLE IF NOT EXISTS evidencias (
                id SERIAL PRIMARY KEY,
                message_id VARCHAR(255) UNIQUE NOT NULL,
                sender VARCHAR(100) NOT NULL,
                nro VARCHAR(50),
                votos_candidato_1 INTEGER,
                votos_candidato_2 INTEGER,
                dni_presidente VARCHAR(20),
                minio_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Índice para búsqueda rápida por sender
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_evidencias_sender 
            ON evidencias(sender)
        """)
        
        # Tabla de reenviados
        cur.execute("""
            CREATE TABLE IF NOT EXISTS reenviados (
                id SERIAL PRIMARY KEY,
                message_id VARCHAR(255) UNIQUE NOT NULL,
                sender VARCHAR(100) NOT NULL,
                nro VARCHAR(50),
                votos_candidato_1 INTEGER,
                votos_candidato_2 INTEGER,
                dni_presidente VARCHAR(20),
                minio_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        print("✅ Tablas creadas/verificadas correctamente")
        
    except Exception as e:
        print(f"❌ Error creando tablas: {e}")
    finally:
        cur.close()
        conn.close()

@app.post("/whatsapp/register-dni")
async def register_dni(request: dict):
    """Registrar DNI de un sender en Redis solo si no existe"""
    sender = request.get("sender")
    dni = request.get("dni")
    
    if not sender or not dni:
        return {"success": False, "message": "Faltan datos requeridos"}
    
    # Validar DNI de 8 dígitos
    if not dni.isdigit() or len(dni) != 8:
        return {"success": False, "message": "DNI debe tener 8 dígitos numéricos"}
    
    # 🔥 VERIFICAR SI EL DNI YA EXISTE EN REDIS
    key = f"registro_exitoso:{sender}"
    existing_dni = redis_client.get(key)
    
    if existing_dni:
        # El sender ya tiene DNI registrado
        return {
            "success": True, 
            "exists": True,
            "message": "Cliente ya registrado"
        }
    
    # 🔥 VERIFICAR SI EL DNI YA ESTÁ REGISTRADO CON OTRO SENDER (opcional)
    # Buscar en PostgreSQL si el DNI ya existe con otro sender
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT sender FROM usuarios WHERE dni = %s", (dni,))
    existing = cur.fetchone()
    
    if existing:
        cur.close()
        conn.close()
        return {
            "success": False,
            "exists": True,
            "message": "Este DNI ya está registrado con otro número de WhatsApp"
        }
    
    try:
        # Guardar en Redis (expira en 30 días)
        redis_client.setex(key, 86400 * 30, dni)
        
        # Guardar en PostgreSQL
        cur.execute("""
            INSERT INTO usuarios (sender, dni, created_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (sender) DO UPDATE SET dni = EXCLUDED.dni
        """, (sender, dni, datetime.now()))
        conn.commit()
        cur.close()
        conn.close()
        
        return {
            "success": True,
            "exists": False,
            "message": "DNI registrado correctamente"
        }
        
    except Exception as e:
        print(f"❌ Error registrando DNI: {e}")
        return {"success": False, "message": "Error interno del servidor"}
    
@app.post("/whatsapp/check-sender")
async def check_sender_in_redis(request: dict):
    """Verificar si un sender existe en Redis con su DNI"""
    sender = request.get("sender")
    
    if not sender:
        return {"exists": False, "error": "No sender provided"}
    
    # Buscar en Redis
    key = f"registro_exitoso:{sender}"
    dni = redis_client.get(key)
    
    if dni:
        return {"exists": True, "dni": dni}
    else:
        return {"exists": False, "dni": None}

# ============ EJECUTAR ============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)