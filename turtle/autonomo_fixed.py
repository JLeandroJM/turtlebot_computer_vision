#!/usr/bin/env python3
"""
autonomo_fixed.py - PERSEGUIDOR AGRESIVO DE TURTLEBOT4

Recibe imágenes del robot, detecta el robot perrito con YOLO entrenado
y lo persigue a máxima velocidad.

Modelo: YOLOv8 Medium entrenado con ~494 imágenes
Métricas: mAP50=83.78%, Precision=98.34%, Recall=100%
"""
import socket
import base64
import struct
import time

import numpy as np
import cv2

# ====== MODELO YOLO ======
MODEL_PATH = "other_models/best11s.pt"  # ✅ Modelo entrenado en Colab

# ====== Configuración del Robot ======
ROBOT_IP = "10.182.184.106"
IMG_PORT = 6000
CTRL_PORT = 5007

DESIRED_DOMAIN_ID = 1
PAIRING_CODE = "ROBOT_A_11"
EXPECTED_ROBOT_NAME = "turtlebot4_lite_11"

# ====== Parámetros de Control - COMPENSACIÓN DE DELAY ======
MAX_LIN = 0.5     # m/s - Velocidad reducida para compensar delay (antes 0.8)
MAX_ANG = 0.7     # rad/s - Giro moderado (~40°/s) para evitar sobregiro
K_ANG = 0.5       # Ganancia angular - giros suaves y controlados
K_LIN = 1.0       # Ganancia lineal - avance completo
TARGET_AREA = 0.30  # Distancia objetivo (30% del frame)

# Umbral para decidir acciones
ANGULAR_THRESHOLD = 0.15  # Umbral para considerar centrado
AREA_THRESHOLD = TARGET_AREA * 0.8  # Acepta hasta 80% del target

# Sistema de compensación de delay - PREDICCIÓN AVANZADA
CAMERA_DELAY = 3.0  # Delay estimado de la cámara en segundos
DETECTION_TIMEOUT = 2.0  # Tiempo sin detección antes de buscar
MIN_DETECTIONS_TO_MOVE = 1  # Reacción inmediata
CONFIDENCE_THRESHOLD = 0.25  # Umbral bajo - acepta casi todo

# Parámetros de predicción
ENABLE_PREDICTION = True  # Activar predicción de movimiento
PREDICTION_FACTOR = 0.8  # Factor de predicción (0-1, qué tanto predecir)
MIN_VELOCITY_THRESHOLD = 0.01  # Velocidad mínima para considerar movimiento

# Compensación de primera detección (combatir delay inicial)
ENABLE_FIRST_DETECTION_COMPENSATION = True  # Activar compensación en primera detección
FIRST_DETECTION_REVERSE_ANGLE = 45.0  # Grados a retroceder en el giro (~0.79 rad)
FIRST_DETECTION_REVERSE_TIME = 1.5  # Tiempo para ejecutar el giro inverso (segundos)

# Tasa de envío de comandos
COMMAND_RATE = 0.1  # Comandos cada 0.1s (10 Hz)


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

    # Priorizar Ultralytics (YOLOv8) - tu modelo entrenado
    try:
        from ultralytics import YOLO
        model = YOLO(path)
        model.conf = 0.5  # Umbral de confianza 50% (ajustado para tu modelo)
        model.iou = 0.45  # IoU para NMS
        print("[MODEL] ✅ Modelo YOLOv8 cargado exitosamente")
        print(f"[MODEL] 📊 Entrenado con: mAP50=83.78%, Precision=98.34%, Recall=100%")
        return model, 'ultralytics'
    except Exception as e:
        print(f"[MODEL] ❌ Error cargando Ultralytics: {e}")

    # Fallback: Intentar YOLOv5 (torch.hub)
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path, verbose=False)
        print("[MODEL] ✅ Modelo YOLOv5 cargado (fallback)")
        return model, 'yolov5'
    except Exception as e:
        print(f"[MODEL] ❌ YOLOv5 también falló: {e}")

    print("[MODEL] ❌ No se pudo cargar el modelo")
    print("[MODEL] 💡 Verifica que exista: models_trained/best.pt")
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
    
    Estrategia ANTI-DELAY:
    1. Estima velocidad del objetivo entre frames
    2. Predice dónde ESTARÁ en CAMERA_DELAY segundos
    3. Apunta hacia la posición PREDICHA (no actual)
    4. Considera nuestros comandos pasados
    """
    h, w = img_shape[:2]
    current_time = time.time()
    
    # Sin detecciones: verificar timeout
    if dets is None or len(dets) == 0:
        time_since_detection = current_time - last_detection_time if last_detection_time > 0 else 999
        
        # Si hace MUY poco que perdió la detección (< 0.5s), PARAR
        if time_since_detection < 0.5:
            return 0.0, 0.0, None
        # Si hace poco (0.5-2s), buscar LENTO
        elif time_since_detection < 2.0:
            return 0.0, MAX_ANG * 0.4, None  # Giro lento
        # Si pasó más tiempo, buscar NORMAL
        else:
            return 0.0, MAX_ANG * 0.6, None  # Giro más rápido
    
    # Encontrar mejor detección - PRIORIZAR ÁREA sobre confianza
    best = None
    best_score = 0
    for d in dets:
        x1, y1, x2, y2 = d[0], d[1], d[2], d[3]
        conf = float(d[4]) if len(d) > 4 else 1.0
        
        # Filtrar SOLO detecciones extremadamente bajas
        if conf < CONFIDENCE_THRESHOLD:
            continue
        
        area = (x2 - x1) * (y2 - y1)
        # Score: ÁREA tiene más peso que confianza
        # Si confianza baja pero área grande = probablemente es el robot
        score = (area * 2.0) + (conf * area * 0.5)
        
        if score > best_score:
            best_score = score
            best = d
    
    if best is None:
        return 0.0, 0.0, None
    
    # Calcular centro y área del bbox
    x1, y1, x2, y2 = best[0], best[1], best[2], best[3]
    conf = float(best[4]) if len(best) > 4 else 1.0
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bbox_area = (x2 - x1) * (y2 - y1)
    area_frac = bbox_area / (w * h)
    
    # Error angular: qué tan descentrado está (-1 a 1)
    err_x = (cx - w/2.0) / (w/2.0)
    
    # Guardar detección en historial
    detection_info = {
        'cx': cx,
        'cy': cy,
        'err_x': err_x,
        'area_frac': area_frac,
        'conf': conf,
        'time': current_time
    }
    
    # Agregar a historial
    detection_history.append(detection_info)
    # Mantener solo últimos 5 frames (reducido de 10)
    if len(detection_history) > 5:
        detection_history.pop(0)
    
    # ¡YA NO ESPERA! Reacciona inmediatamente si MIN_DETECTIONS_TO_MOVE = 1
    
    # ====== PREDICCIÓN DE MOVIMIENTO ======
    predicted_err_x = err_x
    predicted_area = area_frac
    
    if ENABLE_PREDICTION and len(detection_history) >= 2:
        # Calcular velocidad del objetivo
        prev_det = detection_history[-2]
        dt = current_time - prev_det['time']
        
        if dt > 0.001:  # Evitar división por cero
            # Velocidad horizontal (movimiento lateral del objetivo)
            velocity_x = (cx - prev_det['cx']) / dt  # pixels/segundo
            
            # Velocidad de cambio de área (acercándose/alejándose)
            velocity_area = (area_frac - prev_det['area_frac']) / dt
            
            # Predecir posición futura (compensar delay)
            delay_to_compensate = CAMERA_DELAY * PREDICTION_FACTOR
            
            predicted_cx = cx + (velocity_x * delay_to_compensate)
            predicted_cx = max(0, min(w, predicted_cx))  # Limitar a imagen
            
            predicted_err_x = (predicted_cx - w/2.0) / (w/2.0)
            
            # Predecir área futura
            predicted_area = area_frac + (velocity_area * delay_to_compensate)
            predicted_area = max(0.0, min(1.0, predicted_area))
            
            print(f"[PREDICT] vel_x={velocity_x:.1f}px/s, delay={delay_to_compensate:.1f}s → err={err_x:.2f}→{predicted_err_x:.2f}")
    
    # Suavizado adicional con historial
    if len(detection_history) >= 3:
        # Promediar últimas 2 predicciones
        recent_errors = [predicted_err_x] + [d.get('predicted_err_x', d['err_x']) for d in detection_history[-2:]]
        avg_err_x = sum(recent_errors) / len(recent_errors)
        
        recent_areas = [predicted_area] + [d.get('predicted_area', d['area_frac']) for d in detection_history[-2:]]
        avg_area = sum(recent_areas) / len(recent_areas)
    else:
        avg_err_x = predicted_err_x
        avg_area = predicted_area
    
    # Guardar predicción en detección actual
    detection_info['predicted_err_x'] = predicted_err_x
    detection_info['predicted_area'] = predicted_area
    
    # ====== COMPENSACIÓN POR COMANDOS ENVIADOS ======
    # Si enviamos comandos de giro hace poco, el robot ya está girando
    # Reducir comando angular para evitar sobregiro
    angular_compensation = 0.0
    if len(command_history) > 0:
        # Sumar giros enviados en los últimos 2 segundos
        recent_time = current_time - 2.0
        recent_angular = [cmd['w'] for cmd in command_history if cmd['time'] > recent_time]
        if recent_angular:
            angular_compensation = sum(recent_angular) / len(recent_angular) * 0.3  # 30% de compensación
    
    # Decisión de control con PREDICCIÓN
    v = 0.0
    w = 0.0
    
    # Calcular velocidad angular hacia posición PREDICHA
    w = -K_ANG * avg_err_x * MAX_ANG
    w = w - angular_compensation  # Compensar inercia
    w = clamp(w, -MAX_ANG, MAX_ANG)
    
    # ====== ESTRATEGIA DE BÚSQUEDA DE CUERPO COMPLETO ======
    
    # Si está MUY cerca (área grande), PARAR completamente
    if avg_area >= AREA_THRESHOLD:
        v = 0.0
        w = 0.0
        print(f"[CONTROL] 🎯 OBJETIVO ALCANZADO: area={avg_area:.3f} - PARADO")
    
    # Descentrado significativamente: GIRAR mientras avanza POCO
    elif abs(avg_err_x) > 0.25:
        if abs(avg_err_x) > 0.5:  # MUY descentrado
            v = K_LIN * MAX_LIN * 0.2  # Avance MUY lento mientras alinea
            print(f"[CONTROL] 🔄 ALINEANDO (cuerpo completo): err_pred={avg_err_x:.2f}, v={v:.2f}, w={w:.2f}")
        else:  # Moderadamente descentrado
            v = K_LIN * MAX_LIN * 0.4  # Avance lento
            print(f"[CONTROL] ↗️  AJUSTANDO (cuerpo completo): err_pred={avg_err_x:.2f}, v={v:.2f}, w={w:.2f}")
    
    # ✅ CUERPO COMPLETO Y CENTRADO: ¡PERSEGUIR!
    else:
        # Velocidad proporcional a distancia predicha
        if avg_area < 0.05:  # MUY lejos
            v = MAX_LIN  # 100% velocidad - PERSECUCIÓN MÁXIMA
            print(f"[CONTROL] 🚀 PERSECUCIÓN MÁXIMA (cuerpo completo): area={avg_area:.3f}, v={v:.2f}")
        elif avg_area < 0.15:  # Distancia media
            v = MAX_LIN * 0.85  # 85% velocidad
            print(f"[CONTROL] ➡️  PERSIGUIENDO (cuerpo completo): area={avg_area:.3f}, v={v:.2f}")
        else:  # Cerca
            speed_factor = 1.0 - (avg_area / AREA_THRESHOLD)
            v = K_LIN * MAX_LIN * max(0.45, speed_factor)  # mínimo 45% velocidad
            print(f"[CONTROL] 🐢 ACERCÁNDOSE (cuerpo completo): area={avg_area:.3f}, v={v:.2f}")
    
    v = clamp(v, 0.0, MAX_LIN)
    
    return v, w, detection_info


def send_command(sock, addr, v, w):
    """Envía comando de velocidad al robot"""
    payload = struct.pack('ff', float(v), float(w))
    try:
        sock.sendto(payload, addr)
        # Solo printar si es un comando importante (no spam)
        if v > 0.01 or abs(w) > 0.01:
            print(f"[SEND] ✅ v={v:.2f} m/s, w={w:.2f} rad/s → {addr[0]}:{addr[1]}")
        return True
    except Exception as e:
        print(f"[SEND] ❌ Error: {e}")
        return False


def main():
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
    
    # Sistema de compensación de delay con PREDICCIÓN
    detection_history = []  # Historial de detecciones
    command_history = []  # Historial de comandos enviados (para compensación)
    last_detection_time = 0.0  # Última vez que detectó
    consecutive_detections = 0  # Contador de detecciones consecutivas
    
    # Sistema de compensación de primera detección
    first_detection_done = False  # Flag para saber si ya compensamos la primera detección
    compensating_first_detection = False  # Flag para saber si estamos en proceso de compensación
    compensation_start_time = 0.0  # Tiempo de inicio de compensación
    last_angular_command = 0.0  # Último comando angular enviado antes de detectar
    
    print("\n[MAIN] 🚀 Iniciando loop con PREDICCIÓN DE MOVIMIENTO...\n")
    print(f"🔮 Predicción: {'ACTIVADA' if ENABLE_PREDICTION else 'DESACTIVADA'} (factor={PREDICTION_FACTOR})")
    print(f"⏱️  Compensando delay: {CAMERA_DELAY}s")
    print(f"⚡ Confianza mínima: {CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"🎯 Reacción inmediata: {MIN_DETECTIONS_TO_MOVE} detección")
    print(f"🔄 Compensación 1ra detección: {'ACTIVADA' if ENABLE_FIRST_DETECTION_COMPENSATION else 'DESACTIVADA'} ({FIRST_DETECTION_REVERSE_ANGLE}° por {FIRST_DETECTION_REVERSE_TIME}s)")
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
            
            # Procesar solo mensajes IMG
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
                    print(f"[IMG] Error decodificando: {e}")
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
                # Si detectamos por primera vez (bbox verde = conf > 0.6) y tenemos compensación activa
                if (ENABLE_FIRST_DETECTION_COMPENSATION and 
                    not first_detection_done and 
                    not compensating_first_detection and
                    dets is not None and len(dets) > 0):
                    
                    # Verificar si hay detección con alta confianza (verde)
                    best = max(dets, key=lambda d: float(d[4]) if len(d) > 4 else 0.0)
                    conf = float(best[4]) if len(best) > 4 else 0.0
                    
                    if conf > 0.6:  # Detección verde (alta confianza)
                        print("\n" + "="*70)
                        print("🎯 PRIMERA DETECCIÓN EN VERDE - INICIANDO COMPENSACIÓN DE DELAY")
                        print("="*70)
                        print(f"[COMPENSATE] Retrocediendo giro {FIRST_DETECTION_REVERSE_ANGLE}° por {FIRST_DETECTION_REVERSE_TIME}s...")
                        
                        compensating_first_detection = True
                        compensation_start_time = current_time
                        
                        # Calcular giro inverso en radianes
                        reverse_angle_rad = np.radians(FIRST_DETECTION_REVERSE_ANGLE)
                        # Si el último giro fue negativo (derecha), retroceder positivo (izquierda)
                        if last_angular_command < 0:
                            reverse_angle_rad = abs(reverse_angle_rad)
                        else:
                            reverse_angle_rad = -abs(reverse_angle_rad)
                        
                        print(f"[COMPENSATE] Último giro: {last_angular_command:.2f} rad/s")
                        print(f"[COMPENSATE] Giro inverso: {reverse_angle_rad:.2f} rad ({FIRST_DETECTION_REVERSE_ANGLE}°)")
                
                # Si estamos en proceso de compensación
                if compensating_first_detection:
                    elapsed = current_time - compensation_start_time
                    
                    if elapsed < FIRST_DETECTION_REVERSE_TIME:
                        # Ejecutar giro inverso
                        v = 0.0  # NO avanzar durante compensación
                        w = reverse_angle_rad  # Giro inverso constante
                        
                        if current_time - last_command_time >= COMMAND_RATE:
                            send_command(ctrl_sock, ctrl_addr, v, w)
                            last_v, last_w = v, w
                            last_command_time = current_time
                            print(f"[COMPENSATE] ⏪ Retrocediendo giro... {elapsed:.1f}/{FIRST_DETECTION_REVERSE_TIME}s (w={w:.2f})")
                        
                        # Pasar al siguiente frame
                        continue
                    else:
                        # Compensación completada
                        print(f"[COMPENSATE] ✅ COMPENSACIÓN COMPLETADA - Buscando target centrado...")
                        print("="*70 + "\n")
                        compensating_first_detection = False
                        first_detection_done = True
                        
                        # Limpiar detecciones antiguas para empezar fresco
                        detection_history.clear()
                        command_history.clear()
                
                # ====== CONTROL NORMAL ======
                # Actualizar última detección
                if dets is not None and len(dets) > 0:
                    last_detection_time = current_time
                    consecutive_detections += 1
                else:
                    consecutive_detections = 0
                    # Limpiar historial si no hay detecciones
                    if current_time - last_detection_time > 2.0:
                        detection_history.clear()
                
                # Enviar comando a tasa controlada
                if (current_time - last_command_time) >= COMMAND_RATE:
                    v, w, detection_info = calculate_control(
                        dets, 
                        img.shape, 
                        detection_history, 
                        last_detection_time,
                        command_history
                    )
                    
                    # Guardar último comando angular para compensación futura (cuando no hay detección)
                    if (dets is None or len(dets) == 0) and abs(w) > 0.01:
                        last_angular_command = w
                    
                    # Enviar comando si cambió o si está en movimiento
                    should_send = True
                    if abs(v) < 0.01 and abs(w) < 0.01 and abs(last_v) < 0.01 and abs(last_w) < 0.01:
                        should_send = False  # Ya está parado
                    
                    if should_send:
                        if send_command(ctrl_sock, ctrl_addr, v, w):
                            last_v, last_w = v, w
                            # Guardar comando en historial
                            command_history.append({
                                'v': v,
                                'w': w,
                                'time': current_time
                            })
                            # Mantener solo últimos 30 comandos (~3 segundos)
                            if len(command_history) > 30:
                                command_history.pop(0)
                    
                    last_command_time = current_time
                
                # Visualización mejorada
                vis = img.copy()
                h_img, w_img = img.shape[:2]
                
                # Dibujar detecciones con info detallada
                if dets is not None and len(dets) > 0:
                    for i, d in enumerate(dets):
                        x1, y1, x2, y2 = int(d[0]), int(d[1]), int(d[2]), int(d[3])
                        conf = float(d[4])
                        
                        # Calcular dimensiones
                        bbox_width = x2 - x1
                        bbox_height = y2 - y1
                        bbox_area = bbox_width * bbox_height
                        area_frac = bbox_area / (w_img * h_img)
                        
                        # Color según confianza
                        if conf > 0.6:
                            color = (0, 255, 0)  # Verde brillante - ALTA CONFIANZA
                            thickness = 4
                        elif conf > 0.4:
                            color = (0, 255, 255)  # Amarillo - confianza media
                            thickness = 3
                        else:
                            color = (0, 165, 255)  # Naranja - baja confianza
                            thickness = 2
                        
                        # Bbox
                        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
                        
                        # Etiqueta simple con fondo
                        label = f"{conf:.2f} ({area_frac*100:.1f}%)"
                        
                        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(vis, (x1, y1-label_h-10), (x1+label_w+10, y1), color, -1)
                        cv2.putText(vis, label, (x1+5, y1-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        
                        # Centro del objeto
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        cv2.circle(vis, (cx, cy), 5, color, -1)
                        
                        # Línea desde centro de imagen al objeto
                        cv2.line(vis, (w_img//2, h_img//2), (cx, cy), color, 2)
                
                # Crosshair en el centro
                cv2.line(vis, (w_img//2-20, h_img//2), (w_img//2+20, h_img//2), (0, 255, 255), 2)
                cv2.line(vis, (w_img//2, h_img//2-20), (w_img//2, h_img//2+20), (0, 255, 255), 2)
                
                # Panel de información superior (más grande para info de delay)
                panel_h = 150
                overlay = vis.copy()
                cv2.rectangle(overlay, (0, 0), (w_img, panel_h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
                
                # Info en pantalla - más completa
                y_offset = 25
                time_since_detection = time.time() - last_detection_time if last_detection_time > 0 else 0
                detection_status = f"✅ ACTIVO" if consecutive_detections >= MIN_DETECTIONS_TO_MOVE else f"⏳ Conf..."
                det_color = (0, 255, 0) if consecutive_detections >= MIN_DETECTIONS_TO_MOVE else (0, 255, 255)
                cv2.putText(vis, f"FPS: {fps:.1f} | Det: {len(dets) if dets is not None else 0} | {detection_status} | Delay: {time_since_detection:.1f}s", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, det_color, 2)
                y_offset += 28
                
                # Predicción y historial
                pred_status = "🔮 PRED ON" if ENABLE_PREDICTION and len(detection_history) >= 2 else "📊 NORMAL"
                history_color = (0, 255, 0) if len(detection_history) >= MIN_DETECTIONS_TO_MOVE else (0, 165, 255)
                cv2.putText(vis, f"{pred_status} | Hist: {len(detection_history)} | Cmds: {len(command_history)}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, history_color, 2)
                y_offset += 28
                
                # Estado de velocidad con colores
                v_color = (0, 255, 0) if abs(last_v) > 0.01 else (128, 128, 128)
                w_color = (0, 255, 0) if abs(last_w) > 0.01 else (128, 128, 128)
                cv2.putText(vis, f"Vel Linear: {last_v:.2f} m/s ({last_v/MAX_LIN*100:.0f}%)", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, v_color, 2)
                y_offset += 25
                cv2.putText(vis, f"Vel Angular: {last_w:.2f} rad/s ({last_w/MAX_ANG*100 if MAX_ANG > 0 else 0:.0f}%)", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, w_color, 2)
                y_offset += 25
                cv2.putText(vis, f"Target: {ROBOT_IP}:{CTRL_PORT}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Estado de persecución (esquina superior derecha)
                if len(detection_history) > 0 and detection_history[-1].get('is_body_complete', False):
                    if abs(last_v) > 0.1:
                        status = "🚀 PERSIGUIENDO"
                        status_color = (0, 255, 0)
                    else:
                        status = "🎯 ALINEADO"
                        status_color = (0, 255, 200)
                elif dets is not None and len(dets) > 0:
                    body_stat = detection_history[-1].get('body_status', 'PARCIAL') if detection_history else 'PARCIAL'
                    status = f"🔍 {body_stat}"
                    status_color = (0, 255, 255)
                elif time_since_detection < 2.0:
                    status = "⏸️  PERDIDO"
                    status_color = (255, 165, 0)
                else:
                    status = "� BUSCANDO"
                    status_color = (0, 165, 255)
                
                (status_w, status_h), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.putText(vis, status, (w_img-status_w-10, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                
                cv2.imshow("🐕 PUPPYBOT HUNTER - Presiona Q para salir", vis)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Responder handshake si llega HELLO
            elif parts[0] == "HELLO":
                domain = parts[1] if len(parts) > 1 else "1"
                ack = f"ACK {domain} {EXPECTED_ROBOT_NAME}".encode("utf-8")
                img_sock.sendto(ack, addr)
    
    except KeyboardInterrupt:
        print("\n[MAIN] Interrumpido por usuario")
    
    finally:
        # Enviar STOP al salir (múltiples veces para asegurar)
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
