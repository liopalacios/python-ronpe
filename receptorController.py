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
import random
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
    phone_number: Optional[str] = None
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
    total_votos_emitidos: Optional[int] = None
    completo: bool = False
    faltantes: list = []



# ============ RESPUESTAS ALEATORIAS ============
RESPUESTAS_SOLICITUD_FOTO = [
    "📸 Por favor, envía la foto de tu acta electoral.",
    "📷 Estamos listos para recibir la foto de tu acta.",
    "🎯 Envía la imagen del acta para continuar.",
    "📸 Adjunta la foto del acta electoral, por favor.",
    "✅ Ya estás registrado. Envía la foto de tu acta."
]

RESPUESTAS_SOLICITUD_DNI = [
    "📝 Bienvenido al CNP. Por favor, escribe tu número de DNI (8 dígitos).",
    "🔑 Para comenzar, necesito tu número de DNI.",
    "📋 Por favor, regístrate enviando tu DNI de 8 dígitos.",
    "✅ Hola. Para continuar, escribe tu número de DNI.",
    "📝 Personero, por favor ingresa tu DNI (8 dígitos)."
]

RESPUESTAS_REGISTRO_EXITOSO = [
    "✅ Bienvenido {nombre}. Estamos listos para recibir la foto de tu acta.",
    "🙏 Gracias {nombre}. Ahora envía la foto de tu acta electoral.",
    "📸 {nombre}, ya estás registrado. Adjunta la foto de tu acta.",
    "✅ Registro exitoso {nombre}. Esperamos tu foto del acta.",
    "🎯 {nombre}, estamos listos. Envía la imagen del acta."
]

RESPUESTAS_ERROR_REGISTRO = [
    "❌ Error al registrar tu DNI. Por favor intenta nuevamente.",
    "⚠️ No se pudo procesar tu DNI. Revisa que sean 8 dígitos.",
    "❌ DNI inválido. Por favor verifica y envía nuevamente.",
    "⚠️ Ocurrió un error. Envía tu DNI de 8 dígitos otra vez."
]

RESPUESTAS_CONFIRMACION_SI = [
    "✅ ¡Excelente! Acta registrada correctamente.",
    "🙏 Gracias por confirmar. Todo está en orden.",
    "✅ Confirmación recibida. ¡Gracias por tu colaboración!",
    "📋 Acta confirmada correctamente. Gracias.",
    "✅ Perfecto. Acta registrada con éxito."
]

RESPUESTAS_CONFIRMACION_NO = [
    "📸 Por favor, envía NUEVAMENTE la foto del acta electoral con el número de mesa correcto.",
    "🔄 El número de mesa no coincide. Reenvía la foto del acta.",
    "📷 Envía otra foto del acta con el número de mesa correcto.",
    "❌ Número de mesa incorrecto. Por favor envía la foto nuevamente.",
    "📸 Corrige el número de mesa y reenvía la foto del acta."
]

RESPUESTAS_TEXTO_GENERICO = [
    "🙏 Estamos atentos a su valioso apoyo. \n Esperamos tu foto del acta.",
    "✅ Gracias por comunicarte. \n Si deseas ya puedes enviar la foto de tu acta electoral.📸",
    "📋 Eres parte de la solución. \n Esperamos tu foto del acta.",
    "🙌 Gracias por comunicarte. \n Estas habilitado para enviar tu foto del acta.",
    "✅ Continuamos coordinando. \n Por favor, envía la foto de tu acta electoral.   📸"
]
# ============ FUNCIONES DE IA ============
# ============ FUNCIONES DE COLA EN REDIS ============
async def guardar_respuesta_en_cola(chat_id: str, reply: str, delay_segundos: int = None):
    """Guarda una respuesta pendiente en Redis con delay aleatorio"""
    if delay_segundos is None:
        delay_segundos = random.randint(5, 12)
    
    key = f"respuesta_pendiente:{chat_id}"
    data = {
        "reply": reply,
        "scheduled_at": datetime.now().timestamp() + delay_segundos,
        "delay": delay_segundos
    }
    redis_client.setex(key, 60, json.dumps(data))  # Expira en 60 segundos
    print(f"📦 Respuesta guardada en cola para {chat_id}, delay: {delay_segundos}s")

async def obtener_respuesta_pendiente(chat_id: str) -> Optional[dict]:
    """Obtiene respuesta pendiente si está lista para enviar"""
    key = f"respuesta_pendiente:{chat_id}"
    data = redis_client.get(key)
    if data:
        data = json.loads(data)
        if data.get("scheduled_at", 0) <= datetime.now().timestamp():
            redis_client.delete(key)
            return data
    return None


@app.post("/whatsapp/confirmation-response")
async def process_confirmation_response(request: dict):
    """Procesa respuesta SI/NO y guarda en cola"""
    sender = request.get("sender")
    respuesta = request.get("respuesta", "").upper()
    
    confirmacion = await obtener_confirmacion_pendiente(sender)
    
    if not confirmacion:
        no_confirmacion_pendiente = [
            f"❓ No tengo una confirmación pendiente. Por favor, envía primero la foto de tu acta.",
            f"❓ No hay ninguna confirmación pendiente. Envía la foto de tu acta antes por favor.",
            f" No cuento con una confirmación pendiente. ❓ Por favor, primero envía la foto de tu acta.",
            f"❓ No poseo una confirmación pendiente. Por favor, envía la imagen de tu acta primero.",
            f" No tengo confirmaciones pendientes. ❓ Primero envía la foto de tu acta, por favor.",
            f"❓ No existe una confirmación pendiente para ti. Envía la foto de tu acta en primer lugar.",
            f" No hay confirmación pendiente registrada. ❓ Por favor, envía primero la foto de tu acta.",
            f"❓ No dispongo de ninguna confirmación pendiente. Por favor, primero envía la foto de tu acta.",
            f" No tengo pendiente ninguna confirmación. ❓ Por favor, envía tu acta en foto antes.",
            f"❓ Aún no tengo una confirmación pendiente. Por favor, envía primero la foto de tu acta."
        ]
        reply = random.choice(no_confirmacion_pendiente)

        await guardar_respuesta_en_cola(sender, reply)
        return {"success": True}
    
    if respuesta in ["SI", "SÍ", "YES", "Y", "1"]:
        numero_mesa = confirmacion.get("numero_mesa")
        await actualizar_confirmacion_registro(sender, numero_mesa)
        await marcar_registro_exitoso(sender, numero_mesa)
        await eliminar_confirmacion_pendiente(sender)
        
        reply = random.choice(RESPUESTAS_CONFIRMACION_SI)
        await guardar_respuesta_en_cola(sender, reply)
        
    elif respuesta in ["NO", "N", "0"]:
        await eliminar_confirmacion_pendiente(sender)
        reply = random.choice(RESPUESTAS_CONFIRMACION_NO)
        await guardar_respuesta_en_cola(sender, reply)
    
    return {"success": True}

@app.post("/whatsapp/text-message")
async def process_text_message(request: dict):
    """Procesa mensajes de texto y guarda respuesta en cola"""
    sender = request.get("sender")
    text = request.get("text", "")

    connfirt = get_db_connection()
    curfirt = connfirt.cursor()
    curfirt.execute("SELECT sender FROM usuarios WHERE sender != %s", (sender,))
    existe_usuario = curfirt.fetchone()
    
    if existe_usuario:
        # Usuario existe y no es SI/NO
        reply = random.choice(RESPUESTAS_TEXTO_GENERICO)
        await guardar_respuesta_en_cola(sender, reply)
    else:
        # Usuario NO existe
        reply = random.choice(RESPUESTAS_SOLICITUD_DNI)
        await guardar_respuesta_en_cola(sender, reply)
    
    return {"success": True}



@app.get("/whatsapp/respuestas-pendientes")
async def get_respuestas_pendientes():
    """Obtiene todas las respuestas pendientes listas para enviar"""
    keys = redis_client.keys("respuesta_pendiente:*")
    respuestas = []
    
    for key in keys:
        data = redis_client.get(key)
        if data:
            data = json.loads(data)
            if data.get("scheduled_at", 0) <= datetime.now().timestamp():
                chat_id = key.replace("respuesta_pendiente:", "")
                respuestas.append({
                    "chat_id": chat_id,
                    "reply": data["reply"]
                })
    
    return respuestas

@app.post("/whatsapp/respuesta-enviada")
async def respuesta_enviada(request: dict):
    """Elimina respuesta de la cola después de enviada"""
    chat_id = request.get("chat_id")
    key = f"respuesta_pendiente:{chat_id}"
    redis_client.delete(key)
    return {"success": True}

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
    total_votos_emitidos: int, # ← Añadir este parámetro
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
                    votos_blanco, votos_nulos, votos_impugnados, total_votos_emitidos, dni_presidente, minio_path, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                message_id, sender, nro_mesa, votos_c1,
                votos_c2, votos_blanco, votos_nulos, votos_inpugnados, total_votos_emitidos, dni_presidente_mesa,
                minio_path, datetime.now()
            ))
        else:
            # Tabla principal
            cur.execute("""
                INSERT INTO evidencias (
                    message_id, sender, nro, votos_candidato_1, votos_candidato_2,
                    votos_blancos, votos_nulos, votos_impugnados, total_votos_emitidos, dni_presidente, minio_path, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                message_id, sender, nro_mesa, votos_c1,
                votos_c2, votos_blanco, votos_nulos, votos_inpugnados, total_votos_emitidos, dni_presidente_mesa,
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


@app.get("/whatsapp/confirmacion/{sender}")
async def verificar_estado_confirmacion(sender: str):
    """Verificar si un sender tiene confirmación pendiente"""
    confirmacion = await obtener_confirmacion_pendiente(sender)
    
    if confirmacion:
        return {
            "pending": True,
            "numero_mesa": confirmacion.get("numero_mesa"),
            "expira_en": redis_client.ttl(f"pendiente_confirmacion:{sender}")
        }
    else:
        return {"pending": False}

# ============ ENDPOINT PRINCIPAL ============

@app.post("/whatsapp/message")
async def process_whatsapp_message(msg: WhatsAppMessage):
    """
     Endpoint principal que recibe mensajes desde Node.js
    """
    print(f"📩 Mensaje recibido de {msg.sender} - Tipo: {msg.type}")
    print(f"📸 Imagen recibida: {bool(msg.image_base64)} - Texto: {msg.text is not None}")
    
    #if msg.type == "text" and msg.text:
    #    texto = msg.text.strip().upper()
        
        # Verificar si hay confirmación pendiente
    #    confirmacion = await obtener_confirmacion_pendiente(msg.sender)
        
    #    if confirmacion:
    #        if texto in ["SI", "SÍ", "YES", "Y", "1"]:
                # ✅ Usuario confirmó el número de mesa
    #            numero_mesa = confirmacion.get("numero_mesa")
    #            datos_temp = confirmacion.get("datos", {})
                
                # Guardar registro en PostgreSQL
    #            await actualizar_confirmacion_registro(msg.sender, numero_mesa)
                
                # Marcar como registrado exitosamente
    #            await marcar_registro_exitoso(msg.sender, numero_mesa)
                
                # Limpiar estado de confirmación
    #            await eliminar_confirmacion_pendiente(msg.sender)
                
    #            respuestas = [
    #                "✅ Recepción conforme, muchas gracias por tu valioso apoyo."
    #            ]
    #            import random
    #            return {"reply": random.choice(respuestas)}
                
    #        elif texto in ["NO", "No", "N", "0"]:
                # ❌ Usuario dice que el número de mesa no es correcto
    #            await eliminar_confirmacion_pendiente(msg.sender)
    #            return {
    #                "reply": "📸 Por favor, envía nuevamente la foto del acta electoral para corregir el número de mesa."
    #            }
    #        else:
    #            # Respuesta no válida, seguir preguntando
    #            return {
    #                "reply": f"❓ No entendí tu respuesta. Por favor responde con *SI* o *NO* para confirmar si tu mesa es la {confirmacion.get('numero_mesa')}."
    #            }*/
    

    # 1. Verificar si es imagen
    if msg.type != "image" or not msg.image_base64:
        solicitud_foto_acta = [
            f"📸 Por favor, envía una foto del acta electoral para procesar la evidencia.",
            f"📸 Para procesar la evidencia, por favor envía una foto del acta electoral.",
            f" Envía una foto del acta electoral por favor 📸 para poder procesar la evidencia.",
            f"📸 Por favor, comparte una imagen del acta electoral y así procesar la evidencia.",
            f" Necesitamos que envíes una foto del acta electoral 📸 para procesar la evidencia.",
            f"📸 Para continuar con la evidencia, por favor envía una fotografía del acta electoral.",
            f" Por favor, envía una foto del acta electoral. 📸 Así podremos procesar la evidencia.",
            f"📸 Por favor, remite una foto del acta electoral con el fin de procesar la evidencia.",
            f" Envía una imagen clara del acta electoral 📸 para que podamos procesar la evidencia.",
            f"📸 Por favor, adjunta una foto del acta electoral y así procesaremos la evidencia correctamente."
        ]
        reply = random.choice(solicitud_foto_acta)
        await guardar_respuesta_en_cola(msg.sender, reply)
        return {"status": "queued"}

    
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
            
            exito_evidencia_adicional = [
                f"✅ Evidencia adicional recibida correctamente. Gracias.",
                f"✅ Hemos recibido la evidencia adicional sin problemas. Gracias.",
                f" Evidencia adicional recibida con éxito. ✅ Gracias por enviarla.",
                f"✅ La evidencia adicional fue recibida correctamente. Muchas gracias.",
                f" Recibimos correctamente tu evidencia adicional. ✅ Gracias.",
                f"✅ Evidencia adicional registrada exitosamente. Gracias por tu envío.",
                f" Se ha recibido la evidencia adicional de forma correcta. ✅ Gracias.",
                f"✅ Todo correcto con la evidencia adicional recibida. Gracias.",
                f" La evidencia adicional ha sido recibida satisfactoriamente. ✅ Gracias.",
                f"✅ Recibimos tu evidencia adicional sin errores. Muchas gracias."
            ]
            reply = random.choice(exito_evidencia_adicional)
            await guardar_respuesta_en_cola(msg.sender, reply)
            return {"status": "queued"}

            
        except Exception as e:
            print(f"❌ Error guardando reenvío: {e}")
            error_guardar_imagen = [
                f"⚠️ Error al guardar la imagen. Por favor, inténtalo de nuevo.",
                f"⚠️ Ocurrió un error al guardar la foto. Intenta nuevamente.",
                f" No se pudo guardar la imagen. ⚠️ Por favor, vuelve a intentarlo.",
                f"⚠️ Fallo al guardar la imagen. Inténtalo otra vez por favor.",
                f" Hubo un error guardando la imagen. ⚠️ Por favor, reintenta.",
                f"⚠️ No se ha podido guardar la imagen. Por favor, intenta de nuevo.",
                f" Error al almacenar la imagen. ⚠️ Por favor, vuelve a intentarlo.",
                f"⚠️ No se logró guardar la foto. Inténtalo nuevamente por favor.",
                f" Fallo al guardar la imagen. ⚠️ Por favor, realiza otro intento.",
                f"⚠️ La imagen no pudo ser guardada. Por favor, intenta nuevamente."
            ]
            reply = random.choice(error_guardar_imagen)
            await guardar_respuesta_en_cola(msg.sender, reply)
            return {"status": "queued"}
    
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
            
            datos_detectados_poco_claros = [
                f"📸 Datos detectados:\n" + "\n".join(mensaje_detalle) +
                f"\n\nPor favor, asegúrate de que la foto sea más clara y que los números escritos a mano se vean bien.",

                f"📸 Información detectada:\n" + "\n".join(mensaje_detalle) +
                f"\n\nPor favor, verifica que la imagen sea más nítida y que los números manuscritos se distingan correctamente.",

                f"📸 Se detectaron los siguientes datos:\n" + "\n".join(mensaje_detalle) +
                f"\n\nAsegúrate por favor de que la foto sea más clara y que los números escritos a mano se aprecien bien.",

                f"📸 Datos extraídos:\n" + "\n".join(mensaje_detalle) +
                f"\n\nPor favor, confirma que la fotografía sea más clara y que los números hechos a mano se vean correctamente.",

                f"📸 Esto es lo que se pudo leer:\n" + "\n".join(mensaje_detalle) +
                f"\n\nPor favor, asegúrate de enviar una foto más clara donde los números escritos a mano se aprecien bien.",

                f"📸 Contenido detectado:\n" + "\n".join(mensaje_detalle) +
                f"\n\nPor favor, cuida que la foto sea más nítida y que los números manuscritos se vean con claridad.",

                f"📸 Datos reconocidos:\n" + "\n".join(mensaje_detalle) +
                f"\n\nAsegúrate por favor de que la imagen sea más clara y que los números escritos manualmente se distingan bien.",

                f"📸 Información leída:\n" + "\n".join(mensaje_detalle) +
                f"\n\nPor favor, verifica que la foto tenga mejor claridad y que los números a mano se observen correctamente.",

                f"📸 Se ha detectado:\n" + "\n".join(mensaje_detalle) +
                f"\n\nPor favor, asegúrate de que la fotografía sea más clara y que los números escritos a mano se vean adecuadamente.",

                f"📸 Datos obtenidos:\n" + "\n".join(mensaje_detalle) +
                f"\n\nPor favor, procura que la imagen sea más clara y que los números hechos a mano se puedan leer bien."
            ]
            reply = random.choice(datos_detectados_poco_claros)
            await guardar_respuesta_en_cola(msg.sender, reply)
            return {"status": "queued"}
            
        except:
            pass

        respuestas_no_imagen = [
                f"📸 No se pudo leer correctamente: {faltantes_texto}. Por favor, envía una foto más clara.",
                f" No se pudo obtener correctamente: {faltantes_texto}.📸 Por favor, envía una foto más nitida.",
                f" No se pudo leer : {faltantes_texto}.📸 Por favor, envía una foto más clara. ",
                f"📸 No se pudo capturar : {faltantes_texto}. Por favor, envía una mejor fotografia.",
                f"  No se ha podido leer : {faltantes_texto}. Por favor, envía una nueva foto.📸",
                f"📸 No se pudo leer los valores : {faltantes_texto}. envíar una mejor foto. ",
                f" No se la logrado obtener los datos de la imagen📸. envía por favor, una foto más clara. ",
                f" No se logró obtener los datos de la foto.📸 Por favor, envía una nueva foto. ",
                f"📸 No se pudieron leer los datos de la fotografia. Por favor, envía una foto más clara.",
                f" No se ha podido leer los datos de la imagen. Por favor, envía una foto más clara.📸"
            ]            
            
        reply = random.choice(respuestas_no_imagen)
        await guardar_respuesta_en_cola(msg.sender, reply)
        return {"status": "queued"}
    
    # 5. Obtener los datos (verificar que exista 'data')
    datos = resultado.get("data", {})
    
    if not datos:
        
        respuestas_no_imagen = [
                f"📸 No se pudieron extraer los datos de la imagen. Por favor, envía una foto más clara.",
                f"📸 No se pudieron obtener los datos de la foto. envía una foto más clara por favor.",
                f" No se pudo extraer los datos de la fotografia. 📸envía nuevamente una foto más clara. ",
                f" No se pudo obtener los datos de la imagen. Por favor, vuelve a envia una foto más clara.📸",
                f" No se ha logrado extraer los datos de la foto📸. Por favor, toma una foto más clara.",
                f"📸 No se logró extraer los datos de la fotografia. envíanos una foto más nitida. ",
                f" No se la logrado obtener los datos de la imagen📸. envía por favor, una foto más clara. ",
                f" No se logró obtener los datos de la foto.📸 Por favor, envía una nueva foto. ",
                f"📸 No se pudieron leer los datos de la fotografia. Por favor, envía una foto más clara.",
                f" No se ha podido leer los datos de la imagen. Por favor, envía una foto más clara.📸"
            ]
            
        reply = random.choice(respuestas_no_imagen)
        await guardar_respuesta_en_cola(msg.sender, reply)
        return {"status": "queued"}

    
    # 6. Extraer valores con get() para evitar KeyError
    numero_mesa = datos.get("numero_mesa")
    votos_c1 = datos.get("votos_candidato_1")
    votos_c2 = datos.get("votos_candidato_2")
    votos_blanco = datos.get("votos_blanco")
    votos_nulos = datos.get("votos_nulos")
    votos_inpugnados = datos.get("votos_inpugnados")
    total_votos_emitidos = datos.get("total_votos_emitidos")
    
    # También soportar nombres alternativos (por si acaso)
    if numero_mesa is None:
        numero_mesa = datos.get("nro") or datos.get("numero_acta")
    
    print(f"✅ Datos extraídos: nro_mesa={numero_mesa}, "
          f"votos_candidato_1={votos_c1}, votos_candidato_2={votos_c2}, "
          f"votos_blanco={votos_blanco}, votos_nulos={votos_nulos}, "
          f"votos_inpugnados={votos_inpugnados}, total_votos_emitidos={total_votos_emitidos}")
    
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
        reply = f"📸 No se pudieron leer: {', '.join(missing_fields)}. Por favor, envía una foto más clara."
        await guardar_respuesta_en_cola(msg.sender, reply)
        return {"status": "queued"}

    
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
            total_votos_emitidos=total_votos_emitidos,
            minio_path=minio_path
        )
        
        # Marcar en Redis como registrado exitosamente
        await marcar_registro_exitoso(msg.sender, numero_mesa)
        



        if not missing_fields:
            # Guardar imagen en MinIO
            minio_path = await guardar_imagen_minio(msg.image_base64, msg.id, tipo="original")
            
            # Preparar datos temporales
            datos_temp = {
                "votos_c1": votos_c1,
                "votos_c2": votos_c2,
                "votos_blanco": votos_blanco,
                "votos_nulos": votos_nulos,
                "votos_inpugnados": votos_inpugnados,
                "minio_path": minio_path,
                "dni_presidente": datos.get("dni_presidente_mesa")
            }
            
            # Marcar confirmación pendiente en Redis
            await marcar_confirmacion_pendiente(msg.sender, numero_mesa, datos_temp)


            # Respuesta aleatoria de éxito
            respuestas_exito = [
                f"Por favor confirmanos que tu mesa es la  {numero_mesa} \n \n *(SI/NO)* ",
                f"confirmanos que la mesa es {numero_mesa} \n \n *(SI/NO)* ",
                f"Por favor confirmanos que esta mesa enviada es la  {numero_mesa} \n \n *(SI/NO)* ",
                f"confirmanos por favor si la mesa es la  {numero_mesa} \n \n *(SI/NO)* ",
                f"Por favor confirmanos numero de mesa  {numero_mesa} \n \n *(SI/NO)* ",
                f"Confirmanos que el numero la mesa es {numero_mesa} \n \n *(SI/NO)* ",
                f"Por favor responder si tu mesa es la  {numero_mesa} \n \n *(SI/NO)* ",
                f"confirmanos por favor que la mesa es el numero  {numero_mesa} \n \n *(SI/NO)* ",
                f"Por favor confirmanos {numero_mesa} es el numero de mesa\n \n *(SI/NO)* ",
                f"Por favor confirmanos que el {numero_mesa} es el numero de mesa enviado\n \n *(SI/NO)* "
            ]
            
            
            reply = random.choice(respuestas_exito)
            await guardar_respuesta_en_cola(msg.sender, reply)

        
        return {"reply": reply}
        
    except Exception as e:
        print(f"❌ Error guardando registro: {e}")
        import traceback
        traceback.print_exc()
        error_interno_evidencia = [
            f"⚠️ Error interno al guardar la evidencia. Por favor intenta nuevamente.",
            f"⚠️ Ocurrió un error interno guardando la evidencia. Inténtalo de nuevo.",
            f" Error interno al almacenar la evidencia. ⚠️ Por favor, vuelve a intentarlo.",
            f"⚠️ Fallo interno al guardar la evidencia. Por favor, realiza otro intento.",
            f" Hubo un error interno mientras se guardaba la evidencia. ⚠️ Intenta nuevamente.",
            f"⚠️ No se pudo guardar la evidencia por un error interno. Por favor, reintenta.",
            f" Error interno del sistema al guardar la evidencia. ⚠️ Por favor, inténtalo otra vez.",
            f"⚠️ Se produjo un error interno al guardar la evidencia. Vuelve a intentarlo por favor.",
            f" Error crítico interno al guardar la evidencia. ⚠️ Por favor, intenta de nuevo.",
            f"⚠️ La evidencia no pudo ser guardada por un error interno. Por favor, inténtalo nuevamente."
        ]
        reply = random.choice(error_interno_evidencia)
        await guardar_respuesta_en_cola(msg.sender, reply)



async def actualizar_confirmacion_registro(sender: str, numero_mesa: str):
    """Actualiza el campo confirmado = 1 para el registro del sender"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("""
            UPDATE evidencias 
            SET confirmado = 1 
            WHERE sender = %s 
            AND nro = %s
            AND (confirmado = 0 or confirmado IS NULL)
            
        """, (sender, numero_mesa))
        
        conn.commit()
        
        if cur.rowcount > 0:
            print(f"✅ Registro confirmado para sender {sender}, mesa {numero_mesa}")
        else:
            print(f"⚠️ No se encontró registro pendiente para sender {sender}")
            
    except Exception as e:
        conn.rollback()
        print(f"❌ Error actualizando confirmación: {e}")
        raise
    finally:
        cur.close()
        conn.close()

async def marcar_confirmacion_pendiente(sender: str, numero_mesa: str, datos_temp: dict):
    """Marca que el contacto está esperando confirmación del número de mesa"""
    key = f"pendiente_confirmacion:{sender}"
    data = {
        "numero_mesa": numero_mesa,
        "datos": datos_temp,
        "timestamp": datetime.now().isoformat()
    }
    redis_client.setex(key, 3600, json.dumps(data))  # Expira en 1 hora

async def obtener_confirmacion_pendiente(sender: str) -> Optional[dict]:
    """Obtiene datos de confirmación pendiente si existe"""
    key = f"pendiente_confirmacion:{sender}"
    data = redis_client.get(key)
    if data:
        return json.loads(data)
    return None

async def eliminar_confirmacion_pendiente(sender: str):
    """Elimina el estado de confirmación pendiente"""
    key = f"pendiente_confirmacion:{sender}"
    redis_client.delete(key)

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
    nombre = request.get("nombre")  # Opcional, por si quieres guardar el nombre también
    
    if not sender or not dni:
        faltan_datos_requeridos = [
            f"⚠️ Faltan datos requeridos.",
            f"❌ Hacen falta datos obligatorios.",
            f" No se completaron todos los datos necesarios. ⚠️",
            f"⚠️ Hay campos obligatorios sin completar.",
            f" Faltan información requerida para continuar. ❌",
            f"⚠️ No se han proporcionado todos los datos solicitados.",
            f" Es necesario completar todos los datos faltantes. ⚠️",
            f"❌ Algunos datos requeridos no fueron ingresados.",
            f"⚠️ La información está incompleta. Faltan datos obligatorios.",
            f" Debes llenar todos los campos requeridos. Faltan algunos. ⚠️"
        ]
        reply = random.choice(faltan_datos_requeridos)
        await guardar_respuesta_en_cola(request.get("sender"), reply)
        return {"status": "queued"}
    
    # Validar DNI de 8 dígitos
    if not dni.isdigit() or len(dni) != 8:
        dni_8_digitos = [
            f"⚠️ El DNI debe tener 8 dígitos numéricos.",
            f"❌ El DNI debe contener exactamente 8 números.",
            f" El número de DNI debe ser de 8 dígitos, solo números. ⚠️",
            f"⚠️ El DNI ingresado debe tener 8 caracteres numéricos.",
            f" Por favor, ingresa un DNI válido de 8 dígitos. ❌",
            f"⚠️ El DNI tiene que estar compuesto por 8 dígitos numéricos.",
            f" El formato correcto del DNI es 8 números. ⚠️",
            f"❌ El DNI debe ser numérico y tener una longitud de 8 dígitos.",
            f"⚠️ Verifica tu DNI: debe tener 8 dígitos y solo números.",
            f" El DNI requiere exactamente 8 dígitos, sin letras ni símbolos. ⚠️"
        ]

        reply = random.choice(dni_8_digitos)
        await guardar_respuesta_en_cola(request.get("sender"), reply)
        return {"status": "queued"}
    
    
    # 🔥 VERIFICAR SI EL DNI YA EXISTE EN REDIS
    key = f"registro_exitoso:{sender}"
    existing_dni = False
    
    connfirt = get_db_connection()
    curfirt = connfirt.cursor()
    curfirt.execute("SELECT sender FROM usuarios WHERE dni = %s and sender != %s", (dni, sender))
    existing_dni = curfirt.fetchone()
    

    if existing_dni:
        # El sender ya tiene DNI registrado
        cliente_ya_registrado = [
            f"⚠️ Cliente ya registrado anteriormente. \n Si el DNI es correcto, puedes enviar la foto del acta electoral para continuar.",
            f"❌ El cliente ya se encuentra registrado en el sistema previamente. \n Si el DNI es correcto, por favor envía la foto del acta electoral para continuar.",
            f" Este cliente ya había sido registrado con anterioridad. ⚠️ \n Si el DNI es correcto, puedes enviar la foto del acta electoral para continuar.",
            f"⚠️ El registro del cliente ya existe de antes. \n Si el DNI es correcto, por favor envía la foto del acta electoral para continuar.",
            f" El número de cliente ya está en nuestra base de datos. ❌  \n Si el DNI es correcto, puedes enviar la foto del acta electoral para continuar.",
            f"⚠️ Cliente previamente registrado. No es necesario volver a registrar. \n Si el DNI es correcto, por favor envía la foto del acta electoral para continuar.",
            f" Ya existe un registro previo para este cliente. ⚠️ \n Si el DNI es correcto, puedes enviar la foto del acta electoral para continuar.",
            f"❌ Este cliente ya fue dado de alta anteriormente. \n Si el DNI es correcto, por favor envía la foto del acta electoral para continuar.",
            f"⚠️ El cliente ya forma parte del sistema desde antes. \n Si el DNI es correcto, puedes enviar la foto del acta electoral para continuar.",
            f" Registro duplicado: este cliente ya había sido registrado. ⚠️ \n Si el DNI es correcto, por favor envía la foto del acta electoral para continuar."
        ]
        reply = random.choice(cliente_ya_registrado)
        await guardar_respuesta_en_cola(request.get("sender"), reply)
        return {"status": "queued"}
    
    # 🔥 VERIFICAR SI EL DNI YA ESTÁ REGISTRADO CON OTRO SENDER (opcional)
    # Buscar en PostgreSQL si el DNI ya existe con otro sender
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT sender FROM usuarios WHERE dni = %s", (dni,))
    existing = cur.fetchone()
    
    if existing:
        cur.close()
        conn.close()
        dni_ya_registrado = [
            f"Este DNI ya está registrado con otro número de WhatsApp. \n Por favor verifica el DNI ingresado o envia la foto del acta electoral para continuar.",
            f"⚠️ El DNI ingresado ya se encuentra registrado con un número de WhatsApp diferente. \n Por favor, verifica el DNI o envía la foto del acta electoral para continuar.",
            f" Este DNI ya ha sido registrado previamente con otro WhatsApp. Por favor, verifica. \n Si el DNI es correcto, puedes enviar la foto del acta electoral para continuar.",
            f"❌ El DNI ya está vinculado a otro número de WhatsApp. No se puede volver a registrar. \n Por favor, verifica el DNI o envía la foto del acta electoral para continuar.",
            f" El número de DNI ya existe en nuestros registros asociado a otro WhatsApp. 📱 \n Por favor, envía la foto del acta electoral para continuar." ,
            f"⚠️ Ya hay un registro activo con este DNI y otro número de WhatsApp. \n Por favor, verifica el DNI o envía la foto del acta electoral para continuar.",
            f" Este DNI pertenece a otra cuenta de WhatsApp. Por favor, contacta con soporte. 📞 \n Por favor, verifica el DNI o envía la foto del acta electoral para continuar.",
            f"❌ DNI ya registrado. El sistema muestra este documento vinculado a un WhatsApp distinto. \n Por favor, verifica el DNI o envía la foto del acta electoral para continuar.📸",
            f" El DNI que proporcionaste ya está asociado a otro número de WhatsApp en nuestra base de datos. \n Por favor, verifica el DNI o envía la foto del acta electoral para continuar.📄",
            f"⚠️ No es posible registrar este DNI nuevamente porque ya tiene un WhatsApp asignado. \n Por favor, verifica el DNI o envía la foto del acta electoral para continuar.👍"
        ]

        reply = random.choice(dni_ya_registrado)
        await guardar_respuesta_en_cola(request.get("sender"), reply)
        return {"status": "queued"}
    
    try:
        # Guardar en Redis (expira en 30 días)
        redis_client.setex(key, 86400 * 30, dni)
        
        # Guardar en PostgreSQL
        cur.execute("""
            INSERT INTO usuarios (sender, dni, nombre, created_at)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (sender) DO UPDATE SET dni = EXCLUDED.dni
        """, (sender, dni, nombre, datetime.now()))
        conn.commit()
        cur.close()
        conn.close()
        
        dni_registrado_correctamente = [
            f"{nombre} ✅ Su DNI se registró correctamente.\n Puede proceder a enviar la foto del acta electoral.",
            f"{nombre} ✅ Su DNI ha sido registrado exitosamente. \n Ahora puedes enviar la foto del acta electoral.",
            f"{nombre} ✅ Su DNI se ha guardado correctamente en el sistema. \n Por favor, envía la foto del acta electoral para continuar.",
            f"{nombre} ✅ Se ha registrado su DNI sin inconvenientes.\n Ahora, por favor, envía la foto del acta electoral.",
            f"{nombre} ✅ Su número de DNI fue registrado con éxito. ✅ \n Por favor, envía la foto del acta electoral para seguir adelante.",
            f"{nombre} ✅ Registro de DNI completado satisfactoriamente. \n Ahora puedes enviar la foto del acta electoral.",
            f"{nombre} ✅ Su DNI se almacenó correctamente. \n Por favor, envía la foto del acta electoral para continuar con el proceso.",
            f"{nombre} ✅ Su DNI se ha registrado de forma exitosa. \n Ahora, por favor, envía la foto del acta electoral para avanzar.",
            f"{nombre} Proceso completado: Su DNI se ha registrado correctamente. ✅ \n Por favor, envía la foto del acta electoral para seguir con el registro.",
            f"{nombre} ✅ Tu DNI ha quedado registrado sin errores. \n Ahora puedes enviar la foto del acta electoral para continuar con el proceso."
        ]

        reply = random.choice(dni_registrado_correctamente)
        await guardar_respuesta_en_cola(request.get("sender"), reply)
        return {"status": "queued"}

    except Exception as e:
        print(f"❌ Error registrando DNI: {e}")
        error_interno_registrar_personero = [
            f"⚠️ Error interno del servidor al registrar el personero.",
            f"❌ Ocurrió un error interno en el servidor mientras se registraba al personero.",
            f" Error interno del sistema al intentar registrar al personero. ⚠️",
            f"⚠️ Fallo interno del servidor durante el registro del personero.",
            f" Hubo un error interno al registrar al personero. Por favor, intenta nuevamente. ❌",
            f"⚠️ El servidor presentó un error interno al procesar el registro del personero.",
            f" No se pudo completar el registro del personero por un error interno del servidor. ⚠️",
            f"❌ Error interno en el servidor. No se pudo registrar al personero.",
            f"⚠️ Se produjo un fallo interno al guardar los datos del personero.",
            f" Error crítico del servidor al registrar al personero. ⚠️ Por favor, reintenta más tarde."
        ]

        reply = random.choice(error_interno_registrar_personero)
        await guardar_respuesta_en_cola(request.get("sender"), reply)
        return {"status": "queued"}

    
@app.post("/whatsapp/check-sender")
async def check_sender_in_redis(request: dict):
    """Verificar si un sender existe en Redis con su DNI"""
    sender = request.get("sender")
    
    if not sender:
        return {"exists": False, "error": "No sender provided"}
    
    # 🔥 BUSCAR EN POSTGRESQL (no en Redis)
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        cur.execute("""
            SELECT sender, dni, nombre 
            FROM usuarios 
            WHERE sender = %s
        """, (sender,))
        
        result = cur.fetchone()
        
        if result:
            return {
                "exists": True, 
                "dni": result["dni"],
                "nombre": result.get("nombre")
            }
        else:
            return {"exists": False, "dni": None}
            
    except Exception as e:
        print(f"❌ Error consultando PostgreSQL: {e}")
        return {"exists": False, "error": "Database error"}
    finally:
        cur.close()
        conn.close()

# ============ EJECUTAR ============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)