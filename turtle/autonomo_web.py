#!/usr/bin/env python3
"""
autonomo_web.py - PERSEGUIDOR AGRESIVO DE TURTLEBOT4 CON INTERFAZ WEB

Combina el sistema de persecución YOLO con visualización web en tiempo real:
- Recibe imágenes del robot por UDP
- Detecta el robot perrito con YOLO entrenado
- Envía comandos de persecución
- Transmite datos por WebSocket (cámara YOLO + LiDAR + métricas)
- Interfaz Gradio para monitoreo remoto

Modelo: YOLOv8 Medium entrenado con ~494 imágenes
Métricas: mAP50=83.78%, Precision=98.34%, Recall=100%
"""
import socket
import base64
import struct
import time
import asyncio
import threading
import json
from typing import Optional, Dict, Any

import numpy as np
import cv2

# ====== IMPORTS OPCIONALES (Web Server) ======
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print("[WARN] websockets no disponible. Instalar: pip install websockets")

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("[WARN] gradio no disponible. Instalar: pip install gradio")

# ====== MODELO YOLO ======
MODEL_PATH = "other_models/best11s.pt"

# ====== Configuración del Robot ======
ROBOT_IP = "10.182.184.106"
IMG_PORT = 6000
CTRL_PORT = 5007

DESIRED_DOMAIN_ID = 1
PAIRING_CODE = "ROBOT_A_11"
EXPECTED_ROBOT_NAME = "turtlebot4_lite_11"

# ====== Parámetros de Control - COMPENSACIÓN DE DELAY ======
MAX_LIN = 0.5
MAX_ANG = 0.7
K_ANG = 0.5
K_LIN = 1.0
TARGET_AREA = 0.30

ANGULAR_THRESHOLD = 0.15
AREA_THRESHOLD = TARGET_AREA * 0.8

CAMERA_DELAY = 3.0
DETECTION_TIMEOUT = 2.0
MIN_DETECTIONS_TO_MOVE = 1
CONFIDENCE_THRESHOLD = 0.25

ENABLE_PREDICTION = True
PREDICTION_FACTOR = 0.8
MIN_VELOCITY_THRESHOLD = 0.01

ENABLE_FIRST_DETECTION_COMPENSATION = True
FIRST_DETECTION_REVERSE_ANGLE = 45.0
FIRST_DETECTION_REVERSE_TIME = 1.5

COMMAND_RATE = 0.1

# ====== Configuración WebSocket Server ======
WS_HOST = "0.0.0.0"
WS_PORT = 8765
GRADIO_PORT = 7860

# ====== Variables Globales para Web ======
latest_web_image = None
latest_web_metrics = {
    'fps': 0.0,
    'detections': 0,
    'velocity_linear': 0.0,
    'velocity_angular': 0.0,
    'status': 'INICIANDO',
    'confidence': 0.0,
    'area_fraction': 0.0,
    'prediction_enabled': ENABLE_PREDICTION
}
latest_lidar_data = None  # Para datos LiDAR (simulado o real)
connected_ws_clients = set()
web_lock = threading.Lock()


def clamp(x, mn, mx):
    """Limita un valor entre mínimo y máximo"""
    return max(mn, min(mx, x))


def do_handshake(sock, robot_addr):
    """Realiza handshake con el robot"""
    sock.settimeout(1.0)
    print(f"[HANDSHAKE] Conectando con {robot_addr}...")
    
    while True:
        msg = f"HELLO {DESIRED_DOMAIN_ID} {PAIRING_CODE}".encode("utf-8")
        sock.sendto(msg, robot_addr)

        try:
            data, addr = sock.recvfrom(4096)
            text = data.decode("utf-8").strip()
            parts = text.split()

            if len(parts) >= 3 and parts[0] == "ACK":
                domain_id = int(parts[1])
                robot_name = " ".join(parts[2:])
                
                if domain_id == DESIRED_DOMAIN_ID and robot_name == EXPECTED_ROBOT_NAME:
                    print(f"[HANDSHAKE] ✅ Conectado con '{robot_name}'")
                    sock.settimeout(None)
                    return
                    
        except socket.timeout:
            print("[HANDSHAKE] Reintentando...")
        except KeyboardInterrupt:
            raise


def load_model(path):
    """Carga el modelo YOLO entrenado"""
    print(f"[MODEL] Cargando modelo desde {path}...")
    
    try:
        import torch
    except ImportError:
        print("[MODEL] ❌ PyTorch no instalado. Instala: pip install torch ultralytics")
        return None, None

    try:
        from ultralytics import YOLO
        model = YOLO(path)
        model.conf = 0.5
        model.iou = 0.45
        print("[MODEL] ✅ Modelo YOLOv8 cargado exitosamente")
        print(f"[MODEL] 📊 Entrenado con: mAP50=83.78%, Precision=98.34%, Recall=100%")
        return model, 'ultralytics'
    except Exception as e:
        print(f"[MODEL] ❌ Error cargando Ultralytics: {e}")

    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path, verbose=False)
        print("[MODEL] ✅ Modelo YOLOv5 cargado (fallback)")
        return model, 'yolov5'
    except Exception as e:
        print(f"[MODEL] ❌ YOLOv5 también falló: {e}")

    print("[MODEL] ❌ No se pudo cargar el modelo")
    return None, None


def run_inference(model, backend, img):
    """Ejecuta inferencia en la imagen"""
    if model is None:
        return []

    if backend == 'yolov5':
        results = model(img)
        try:
            return results.xyxy[0].cpu().numpy()
        except:
            try:
                return results.pred[0].cpu().numpy()
            except:
                return []

    if backend == 'ultralytics':
        res = model(img)[0]
        dets = []
        for b in res.boxes:
            xyxy = b.xyxy.cpu().numpy().ravel()
            conf = float(b.conf.cpu().numpy())
            cls = int(b.cls.cpu().numpy())
            dets.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, cls])
        return np.array(dets) if dets else []

    return []


def calculate_control(dets, img_shape, detection_history, last_detection_time, command_history):
    """
    Calcula comandos con PREDICCIÓN de posición futura del objetivo
    Retorna: (v, w, detection_info)
    """
    h, w = img_shape[:2]
    current_time = time.time()
    
    if dets is None or len(dets) == 0:
        time_since_detection = current_time - last_detection_time if last_detection_time > 0 else 999
        
        if time_since_detection < 0.5:
            return 0.0, 0.0, None
        elif time_since_detection < 2.0:
            return 0.0, MAX_ANG * 0.4, None
        else:
            return 0.0, MAX_ANG * 0.6, None
    
    best = None
    best_score = 0
    for d in dets:
        x1, y1, x2, y2 = d[0], d[1], d[2], d[3]
        conf = float(d[4]) if len(d) > 4 else 1.0
        
        if conf < CONFIDENCE_THRESHOLD:
            continue
        
        area = (x2 - x1) * (y2 - y1)
        score = (area * 2.0) + (conf * area * 0.5)
        
        if score > best_score:
            best_score = score
            best = d
    
    if best is None:
        return 0.0, 0.0, None
    
    x1, y1, x2, y2 = best[0], best[1], best[2], best[3]
    conf = float(best[4]) if len(best) > 4 else 1.0
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bbox_area = (x2 - x1) * (y2 - y1)
    area_frac = bbox_area / (w * h)
    
    err_x = (cx - w/2.0) / (w/2.0)
    
    detection_info = {
        'cx': cx,
        'cy': cy,
        'err_x': err_x,
        'area_frac': area_frac,
        'conf': conf,
        'time': current_time
    }
    
    detection_history.append(detection_info)
    if len(detection_history) > 5:
        detection_history.pop(0)
    
    predicted_err_x = err_x
    predicted_area = area_frac
    
    if ENABLE_PREDICTION and len(detection_history) >= 2:
        prev_det = detection_history[-2]
        dt = current_time - prev_det['time']
        
        if dt > 0.001:
            velocity_x = (cx - prev_det['cx']) / dt
            velocity_area = (area_frac - prev_det['area_frac']) / dt
            
            delay_to_compensate = CAMERA_DELAY * PREDICTION_FACTOR
            
            predicted_cx = cx + (velocity_x * delay_to_compensate)
            predicted_cx = max(0, min(w, predicted_cx))
            
            predicted_err_x = (predicted_cx - w/2.0) / (w/2.0)
            predicted_area = area_frac + (velocity_area * delay_to_compensate)
            predicted_area = max(0.0, min(1.0, predicted_area))
    
    if len(detection_history) >= 3:
        recent_errors = [predicted_err_x] + [d.get('predicted_err_x', d['err_x']) for d in detection_history[-2:]]
        avg_err_x = sum(recent_errors) / len(recent_errors)
        
        recent_areas = [predicted_area] + [d.get('predicted_area', d['area_frac']) for d in detection_history[-2:]]
        avg_area = sum(recent_areas) / len(recent_areas)
    else:
        avg_err_x = predicted_err_x
        avg_area = predicted_area
    
    detection_info['predicted_err_x'] = predicted_err_x
    detection_info['predicted_area'] = predicted_area
    
    angular_compensation = 0.0
    if len(command_history) > 0:
        recent_time = current_time - 2.0
        recent_angular = [cmd['w'] for cmd in command_history if cmd['time'] > recent_time]
        if recent_angular:
            angular_compensation = sum(recent_angular) / len(recent_angular) * 0.3
    
    v = 0.0
    w = 0.0
    
    w = -K_ANG * avg_err_x * MAX_ANG
    w = w - angular_compensation
    w = clamp(w, -MAX_ANG, MAX_ANG)
    
    if avg_area >= AREA_THRESHOLD:
        v = 0.0
        w = 0.0
    elif abs(avg_err_x) > 0.25:
        if abs(avg_err_x) > 0.5:
            v = K_LIN * MAX_LIN * 0.2
        else:
            v = K_LIN * MAX_LIN * 0.4
    else:
        if avg_area < 0.05:
            v = MAX_LIN
        elif avg_area < 0.15:
            v = MAX_LIN * 0.85
        else:
            speed_factor = 1.0 - (avg_area / AREA_THRESHOLD)
            v = K_LIN * MAX_LIN * max(0.45, speed_factor)
    
    v = clamp(v, 0.0, MAX_LIN)
    
    return v, w, detection_info


def send_command(sock, addr, v, w):
    """Envía comando de velocidad al robot"""
    payload = struct.pack('ff', float(v), float(w))
    try:
        sock.sendto(payload, addr)
        return True
    except Exception as e:
        print(f"[SEND] ❌ Error: {e}")
        return False


def draw_lidar_visualization(ranges=None, width=600, height=600):
    """
    Genera visualización del LiDAR (simulado o real)
    Si ranges=None, genera datos simulados para demostración
    """
    import math
    
    img = np.zeros((height, width, 3), dtype=np.uint8)
    center_x = width // 2
    center_y = height // 2
    max_range = 3.0
    scale = min(width, height) // 2 / max_range

    # Centro del robot
    cv2.circle(img, (center_x, center_y), 5, (0, 255, 0), -1)

    # Si no hay datos, simular un entorno básico
    if ranges is None:
        num_points = 360
        ranges = []
        for i in range(num_points):
            angle_deg = i
            # Simular paredes lejanas con algunos obstáculos
            if 80 < angle_deg < 100 or 260 < angle_deg < 280:
                ranges.append(1.5)  # Obstáculo cercano
            else:
                ranges.append(2.5)  # Pared lejana

    num_points = len(ranges)
    angle_increment = 2 * math.pi / num_points

    # Dibujar puntos del LiDAR
    for i, r in enumerate(ranges):
        if r > 0.1 and r < max_range:
            angle = i * angle_increment
            x = int(center_x + r * scale * math.cos(angle))
            y = int(center_y - r * scale * math.sin(angle))
            
            # Color según distancia
            if r < 0.5:
                color = (0, 0, 255)  # Rojo - muy cerca
            elif r < 1.5:
                color = (0, 165, 255)  # Naranja - cerca
            else:
                color = (0, 255, 255)  # Amarillo - lejos
            
            cv2.circle(img, (x, y), 2, color, -1)

    # Círculos de referencia
    cv2.circle(img, (center_x, center_y), int(1.0 * scale), (100, 100, 100), 1)
    cv2.circle(img, (center_x, center_y), int(2.0 * scale), (100, 100, 100), 1)

    # Etiquetas
    cv2.putText(img, "1m", (center_x + int(1.0 * scale) + 10, center_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, "2m", (center_x + int(2.0 * scale) + 10, center_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def update_web_data(vis_img, metrics_dict, lidar_ranges=None):
    """Actualiza datos globales para el servidor web"""
    global latest_web_image, latest_web_metrics, latest_lidar_data
    
    with web_lock:
        # Convertir imagen para web (JPEG base64)
        if vis_img is not None:
            _, jpeg = cv2.imencode('.jpg', vis_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            latest_web_image = base64.b64encode(jpeg.tobytes()).decode('ascii')
        
        # Actualizar métricas
        latest_web_metrics.update(metrics_dict)
        
        # Generar visualización LiDAR
        lidar_img = draw_lidar_visualization(lidar_ranges)
        _, jpeg_lidar = cv2.imencode('.jpg', lidar_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        latest_lidar_data = base64.b64encode(jpeg_lidar.tobytes()).decode('ascii')


# ====== WEBSOCKET SERVER ======
async def websocket_handler(websocket):
    """Handler para clientes WebSocket"""
    connected_ws_clients.add(websocket)
    client_addr = websocket.remote_address
    print(f"[WS] Cliente conectado: {client_addr}")
    
    try:
        while True:
            # Enviar datos actualizados a todos los clientes
            with web_lock:
                if latest_web_image:
                    await websocket.send(json.dumps({
                        'type': 'image',
                        'data': latest_web_image
                    }))
                
                if latest_lidar_data:
                    await websocket.send(json.dumps({
                        'type': 'lidar',
                        'data': latest_lidar_data
                    }))
                
                await websocket.send(json.dumps({
                    'type': 'metrics',
                    'data': latest_web_metrics
                }))
            
            await asyncio.sleep(0.033)  # ~30 FPS
            
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_ws_clients.remove(websocket)
        print(f"[WS] Cliente desconectado: {client_addr}")


async def start_websocket_server():
    """Inicia el servidor WebSocket"""
    if not WEBSOCKETS_AVAILABLE:
        print("[WS] WebSocket no disponible - servidor deshabilitado")
        return
    
    async with websockets.serve(websocket_handler, WS_HOST, WS_PORT):
        print(f"[WS] 🌐 Servidor WebSocket activo en ws://{WS_HOST}:{WS_PORT}")
        await asyncio.Future()  # Run forever


def run_websocket_server_thread():
    """Thread para ejecutar servidor WebSocket"""
    asyncio.run(start_websocket_server())


# ====== GRADIO INTERFACE ======
def create_gradio_interface():
    """Crea interfaz Gradio para visualización web"""
    if not GRADIO_AVAILABLE:
        print("[GRADIO] Gradio no disponible - interfaz deshabilitada")
        return None
    
    def get_camera_image():
        """Obtiene última imagen de cámara con detecciones"""
        with web_lock:
            if latest_web_image:
                img_bytes = base64.b64decode(latest_web_image)
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if frame is not None:
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return np.zeros((240, 320, 3), dtype=np.uint8)
    
    def get_lidar_image():
        """Obtiene visualización del LiDAR"""
        with web_lock:
            if latest_lidar_data:
                img_bytes = base64.b64decode(latest_lidar_data)
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if frame is not None:
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return np.zeros((600, 600, 3), dtype=np.uint8)
    
    def get_metrics_text():
        """Obtiene texto con métricas actuales"""
        with web_lock:
            m = latest_web_metrics
            text = f"""
### 📊 Métricas en Tiempo Real

- **Estado**: {m.get('status', 'N/A')}
- **FPS**: {m.get('fps', 0):.1f}
- **Detecciones**: {m.get('detections', 0)}
- **Confianza**: {m.get('confidence', 0):.2f}
- **Área objetivo**: {m.get('area_fraction', 0)*100:.1f}%

### 🚀 Comandos de Control

- **Velocidad Lineal**: {m.get('velocity_linear', 0):.2f} m/s ({m.get('velocity_linear', 0)/MAX_LIN*100:.0f}%)
- **Velocidad Angular**: {m.get('velocity_angular', 0):.2f} rad/s ({m.get('velocity_angular', 0)/MAX_ANG*100 if MAX_ANG > 0 else 0:.0f}%)

### ⚙️ Configuración

- **Predicción**: {'✅ Activa' if m.get('prediction_enabled', False) else '❌ Inactiva'}
- **Robot IP**: {ROBOT_IP}
- **Modelo**: best11s.pt
"""
            return text
    
    with gr.Blocks(title="🤖 TurtleBot4 Pursuit System", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🐕 PUPPYBOT HUNTER - Sistema de Persecución Autónomo")
        gr.Markdown("**Modelo**: YOLOv8 | **Métricas**: mAP50=83.78%, Precision=98.34%, Recall=100%")
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 📷 Cámara con Detecciones YOLO")
                camera_output = gr.Image(
                    label="Vista en vivo",
                    every=0.1,
                    value=get_camera_image
                )
            
            with gr.Column(scale=1):
                metrics_text = gr.Markdown(
                    value=get_metrics_text,
                    every=0.5
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📡 Visualización LiDAR")
                lidar_output = gr.Image(
                    label="Mapa de entorno",
                    every=0.1,
                    value=get_lidar_image
                )
        
        gr.Markdown("---")
        gr.Markdown("🌐 **WebSocket API**: `ws://localhost:8765` | 🖥️ **Panel Web**: `http://localhost:7860`")
    
    return demo


def run_gradio_interface():
    """Thread para ejecutar interfaz Gradio"""
    demo = create_gradio_interface()
    if demo:
        print(f"[GRADIO] 🌐 Lanzando interfaz en http://{WS_HOST}:{GRADIO_PORT}")
        demo.launch(
            server_name=WS_HOST,
            server_port=GRADIO_PORT,
            share=False,
            quiet=True
        )


def main():
    """Loop principal del sistema de persecución con web server"""
    
    # Iniciar servidores web en threads separados
    if WEBSOCKETS_AVAILABLE:
        ws_thread = threading.Thread(target=run_websocket_server_thread, daemon=True)
        ws_thread.start()
        print("[MAIN] ✅ Servidor WebSocket iniciado")
    
    if GRADIO_AVAILABLE:
        gradio_thread = threading.Thread(target=run_gradio_interface, daemon=True)
        gradio_thread.start()
        print("[MAIN] ✅ Interfaz Gradio iniciada")
        time.sleep(2)  # Dar tiempo a Gradio para iniciar
    
    # Socket para recibir imágenes
    img_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    robot_addr = (ROBOT_IP, IMG_PORT)
    
    # Socket para enviar comandos
    ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ctrl_addr = (ROBOT_IP, CTRL_PORT)
    
    print("=" * 70)
    print("🤖 PERSEGUIDOR AGRESIVO DE TURTLEBOT4 - PUPPYBOT HUNTER 🐕")
    print("=" * 70)
    print(f"🎯 Robot objetivo: {ROBOT_IP}")
    print(f"📥 Recibiendo imágenes: Puerto {IMG_PORT}")
    print(f"📤 Enviando comandos: Puerto {CTRL_PORT}")
    print(f"⚡ Velocidad máxima: {MAX_LIN} m/s")
    print(f"🔄 Giro máximo: {MAX_ANG:.1f} rad/s (~{int(MAX_ANG*57)}°/s)")
    print(f"🎯 Target área: {TARGET_AREA*100:.0f}% del frame")
    print(f"⚙️  Frecuencia comandos: {1/COMMAND_RATE:.0f} Hz")
    if WEBSOCKETS_AVAILABLE:
        print(f"🌐 WebSocket: ws://{WS_HOST}:{WS_PORT}")
    if GRADIO_AVAILABLE:
        print(f"🖥️  Interfaz Web: http://{WS_HOST}:{GRADIO_PORT}")
    print("=" * 70)
    
    # Handshake
    do_handshake(img_sock, robot_addr)
    
    # Cargar modelo
    model, backend = load_model(MODEL_PATH)
    if model is None:
        print("[WARN] Continuando sin modelo - solo mostrará imágenes")
    
    # Variables de control
    last_command_time = 0.0
    last_v, last_w = 0.0, 0.0
    frame_count = 0
    fps_time = time.time()
    fps = 0.0
    
    detection_history = []
    command_history = []
    last_detection_time = 0.0
    consecutive_detections = 0
    
    first_detection_done = False
    compensating_first_detection = False
    compensation_start_time = 0.0
    last_angular_command = 0.0
    
    print("\n[MAIN] 🚀 Iniciando loop con PREDICCIÓN DE MOVIMIENTO...\n")
    print(f"🔮 Predicción: {'ACTIVADA' if ENABLE_PREDICTION else 'DESACTIVADA'} (factor={PREDICTION_FACTOR})")
    print(f"⏱️  Compensando delay: {CAMERA_DELAY}s")
    print(f"⚡ Confianza mínima: {CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"🎯 Reacción inmediata: {MIN_DETECTIONS_TO_MOVE} detección")
    print(f"🔄 Compensación 1ra detección: {'ACTIVADA' if ENABLE_FIRST_DETECTION_COMPENSATION else 'DESACTIVADA'}")
    print(f"Esperando primera imagen...\n")
    
    try:
        while True:
            # Recibir imagen
            data, addr = img_sock.recvfrom(200000)
            
            try:
                text = data.decode("utf-8", errors="ignore").strip()
            except:
                continue
            
            if not text:
                continue
            
            parts = text.split()
            
            if parts[0] == "IMG":
                if len(parts) < 6:
                    continue
                
                # Decodificar imagen
                b64 = " ".join(parts[5:])
                try:
                    jpeg_bytes = base64.b64decode(b64)
                    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    
                    if img is None:
                        continue
                    
                except Exception as e:
                    continue
                
                # Calcular FPS
                frame_count += 1
                if time.time() - fps_time >= 1.0:
                    fps = frame_count / (time.time() - fps_time)
                    frame_count = 0
                    fps_time = time.time()
                
                # Inferencia
                dets = run_inference(model, backend, img)
                
                current_time = time.time()
                
                # ====== COMPENSACIÓN DE PRIMERA DETECCIÓN ======
                if (ENABLE_FIRST_DETECTION_COMPENSATION and 
                    not first_detection_done and 
                    not compensating_first_detection and
                    dets is not None and len(dets) > 0):
                    
                    best = max(dets, key=lambda d: float(d[4]) if len(d) > 4 else 0.0)
                    conf = float(best[4]) if len(best) > 4 else 0.0
                    
                    if conf > 0.6:
                        print("\n" + "="*70)
                        print("🎯 PRIMERA DETECCIÓN EN VERDE - INICIANDO COMPENSACIÓN DE DELAY")
                        print("="*70)
                        
                        compensating_first_detection = True
                        compensation_start_time = current_time
                        
                        reverse_angle_rad = np.radians(FIRST_DETECTION_REVERSE_ANGLE)
                        if last_angular_command < 0:
                            reverse_angle_rad = abs(reverse_angle_rad)
                        else:
                            reverse_angle_rad = -abs(reverse_angle_rad)
                
                if compensating_first_detection:
                    elapsed = current_time - compensation_start_time
                    
                    if elapsed < FIRST_DETECTION_REVERSE_TIME:
                        v = 0.0
                        w = reverse_angle_rad
                        
                        if current_time - last_command_time >= COMMAND_RATE:
                            send_command(ctrl_sock, ctrl_addr, v, w)
                            last_v, last_w = v, w
                            last_command_time = current_time
                        
                        continue
                    else:
                        print(f"[COMPENSATE] ✅ COMPENSACIÓN COMPLETADA")
                        compensating_first_detection = False
                        first_detection_done = True
                        detection_history.clear()
                        command_history.clear()
                
                # ====== CONTROL NORMAL ======
                if dets is not None and len(dets) > 0:
                    last_detection_time = current_time
                    consecutive_detections += 1
                else:
                    consecutive_detections = 0
                    if current_time - last_detection_time > 2.0:
                        detection_history.clear()
                
                # Enviar comando
                if (current_time - last_command_time) >= COMMAND_RATE:
                    v, w, detection_info = calculate_control(
                        dets, 
                        img.shape, 
                        detection_history, 
                        last_detection_time,
                        command_history
                    )
                    
                    if (dets is None or len(dets) == 0) and abs(w) > 0.01:
                        last_angular_command = w
                    
                    should_send = True
                    if abs(v) < 0.01 and abs(w) < 0.01 and abs(last_v) < 0.01 and abs(last_w) < 0.01:
                        should_send = False
                    
                    if should_send:
                        if send_command(ctrl_sock, ctrl_addr, v, w):
                            last_v, last_w = v, w
                            command_history.append({'v': v, 'w': w, 'time': current_time})
                            if len(command_history) > 30:
                                command_history.pop(0)
                    
                    last_command_time = current_time
                
                # Visualización
                vis = img.copy()
                h_img, w_img = img.shape[:2]
                
                best_conf = 0.0
                best_area = 0.0
                
                if dets is not None and len(dets) > 0:
                    for d in dets:
                        x1, y1, x2, y2 = int(d[0]), int(d[1]), int(d[2]), int(d[3])
                        conf = float(d[4])
                        
                        bbox_width = x2 - x1
                        bbox_height = y2 - y1
                        bbox_area = bbox_width * bbox_height
                        area_frac = bbox_area / (w_img * h_img)
                        
                        if conf > best_conf:
                            best_conf = conf
                            best_area = area_frac
                        
                        if conf > 0.6:
                            color = (0, 255, 0)
                            thickness = 4
                        elif conf > 0.4:
                            color = (0, 255, 255)
                            thickness = 3
                        else:
                            color = (0, 165, 255)
                            thickness = 2
                        
                        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
                        
                        label = f"{conf:.2f} ({area_frac*100:.1f}%)"
                        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(vis, (x1, y1-label_h-10), (x1+label_w+10, y1), color, -1)
                        cv2.putText(vis, label, (x1+5, y1-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        cv2.circle(vis, (cx, cy), 5, color, -1)
                        cv2.line(vis, (w_img//2, h_img//2), (cx, cy), color, 2)
                
                # Crosshair
                cv2.line(vis, (w_img//2-20, h_img//2), (w_img//2+20, h_img//2), (0, 255, 255), 2)
                cv2.line(vis, (w_img//2, h_img//2-20), (w_img//2, h_img//2+20), (0, 255, 255), 2)
                
                # Panel de información
                panel_h = 120
                overlay = vis.copy()
                cv2.rectangle(overlay, (0, 0), (w_img, panel_h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
                
                y_offset = 25
                time_since_detection = time.time() - last_detection_time if last_detection_time > 0 else 0
                detection_status = f"✅ ACTIVO" if consecutive_detections >= MIN_DETECTIONS_TO_MOVE else f"⏳ Conf..."
                det_color = (0, 255, 0) if consecutive_detections >= MIN_DETECTIONS_TO_MOVE else (0, 255, 255)
                
                cv2.putText(vis, f"FPS: {fps:.1f} | Det: {len(dets) if dets is not None else 0} | {detection_status}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, det_color, 2)
                y_offset += 28
                
                v_color = (0, 255, 0) if abs(last_v) > 0.01 else (128, 128, 128)
                w_color = (0, 255, 0) if abs(last_w) > 0.01 else (128, 128, 128)
                cv2.putText(vis, f"V: {last_v:.2f} m/s | W: {last_w:.2f} rad/s", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 28
                
                if WEBSOCKETS_AVAILABLE or GRADIO_AVAILABLE:
                    cv2.putText(vis, f"WEB: ws://{WS_HOST}:{WS_PORT} | http://{WS_HOST}:{GRADIO_PORT}", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                
                # Determinar estado
                if consecutive_detections >= MIN_DETECTIONS_TO_MOVE:
                    if abs(last_v) > 0.1:
                        status_text = "🚀 PERSIGUIENDO"
                    else:
                        status_text = "🎯 ALINEADO"
                elif dets is not None and len(dets) > 0:
                    status_text = "🔍 DETECTANDO"
                elif time_since_detection < 2.0:
                    status_text = "⏸️ PERDIDO"
                else:
                    status_text = "🔄 BUSCANDO"
                
                # Actualizar datos para web
                metrics = {
                    'fps': fps,
                    'detections': len(dets) if dets is not None else 0,
                    'velocity_linear': last_v,
                    'velocity_angular': last_w,
                    'status': status_text,
                    'confidence': best_conf,
                    'area_fraction': best_area,
                    'prediction_enabled': ENABLE_PREDICTION
                }
                update_web_data(vis, metrics, lidar_ranges=None)  # LiDAR simulado
                
                # Mostrar ventana local
                cv2.imshow("🐕 PUPPYBOT HUNTER - Presiona Q para salir", vis)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            elif parts[0] == "HELLO":
                domain = parts[1] if len(parts) > 1 else "1"
                ack = f"ACK {domain} {EXPECTED_ROBOT_NAME}".encode("utf-8")
                img_sock.sendto(ack, addr)
    
    except KeyboardInterrupt:
        print("\n[MAIN] Interrumpido por usuario")
    
    finally:
        print("\n[MAIN] 🛑 Deteniendo robot...")
        for _ in range(5):
            send_command(ctrl_sock, ctrl_addr, 0.0, 0.0)
            time.sleep(0.05)
        
        img_sock.close()
        ctrl_sock.close()
        cv2.destroyAllWindows()
        print("[MAIN] ✅ Cerrado correctamente. Robot detenido.")
        print("=" * 70)


if __name__ == '__main__':
    main()
