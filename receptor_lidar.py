#!/usr/bin/env python3
"""
receptor_lidar.py - RECEPTOR DE LIDAR PARA LA PC

Recibe datos SCAN del robot por UDP puerto 6001
Compatible con autonomo_async.py para fusión de sensores

USO STANDALONE (prueba):
    python3 receptor_lidar.py

USO CON AUTONOMO_ASYNC:
    # Importar en autonomo_async.py:
    from receptor_lidar import LidarReceiverUDP, process_lidar_data
"""
import socket
import asyncio
import time
import math
from typing import List, Tuple, Optional


class LidarReceiverUDP:
    """
    Receptor de datos LiDAR por UDP desde el robot
    """
    def __init__(self, robot_ip: str = "10.182.184.101", 
                 lidar_port: int = 6001,
                 pairing_code: str = "ROBOT_A_11",
                 desired_domain_id: int = 1):
        """
        Args:
            robot_ip: IP del robot TurtleBot4
            lidar_port: Puerto UDP para LiDAR (6001 recomendado)
            pairing_code: Código de emparejamiento
            desired_domain_id: ROS_DOMAIN_ID
        """
        self.robot_ip = robot_ip
        self.lidar_port = lidar_port
        self.robot_addr = (robot_ip, lidar_port)
        self.pairing_code = pairing_code
        self.desired_domain_id = desired_domain_id
        
        # Estado
        self.connected = False
        self.robot_name = None
        self.sock = None
        
        # Datos más recientes
        self.latest_scan = None
        self.scan_count = 0
        self.last_scan_time = 0.0
    
    async def connect_async(self):
        """Establece conexión con el robot (handshake)"""
        print(f"[LIDAR-RX] Conectando a {self.robot_addr}...")
        
        # Crear socket UDP
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        
        loop = asyncio.get_event_loop()
        
        # Enviar HELLO hasta recibir ACK
        max_attempts = 10
        for attempt in range(max_attempts):
            msg = f"HELLO {self.desired_domain_id} {self.pairing_code}".encode("utf-8")
            await loop.sock_sendto(self.sock, msg, self.robot_addr)
            
            try:
                data = await asyncio.wait_for(
                    loop.sock_recv(self.sock, 4096),
                    timeout=1.0
                )
                
                text = data.decode("utf-8").strip()
                parts = text.split()
                
                if len(parts) >= 3 and parts[0] == "ACK":
                    domain_id = int(parts[1])
                    robot_name = " ".join(parts[2:])
                    
                    if domain_id == self.desired_domain_id:
                        self.connected = True
                        self.robot_name = robot_name
                        print(f"[LIDAR-RX] ✅ Conectado a '{robot_name}'")
                        return True
            
            except asyncio.TimeoutError:
                if attempt < max_attempts - 1:
                    print(f"[LIDAR-RX] Reintentando ({attempt+1}/{max_attempts})...")
                    await asyncio.sleep(0.5)
        
        print("[LIDAR-RX] ❌ No se pudo conectar")
        return False
    
    async def receive_loop_async(self):
        """
        Loop asíncrono para recibir scans del LiDAR
        Actualiza self.latest_scan con cada nuevo scan
        """
        if not self.connected:
            print("[LIDAR-RX] ❌ No conectado. Llama a connect_async() primero.")
            return
        
        print("[LIDAR-RX] 📡 Iniciando recepción...")
        loop = asyncio.get_event_loop()
        
        fps_count = 0
        fps_time = time.time()
        
        try:
            while True:
                # Recibir datos
                data = await loop.sock_recv(self.sock, 200000)  # Buffer grande
                
                try:
                    text = data.decode("utf-8", errors="ignore").strip()
                except:
                    continue
                
                if not text:
                    continue
                
                parts = text.split()
                
                if parts[0] == "SCAN" and len(parts) >= 8:
                    # Parsear: SCAN <domain> <name> <sec> <nsec> <angle_min> <angle_inc> <n> r1 r2 ...
                    try:
                        domain_id = int(parts[1])
                        # robot_name podría tener espacios, pero simplificamos
                        stamp_sec = int(parts[3])
                        stamp_nsec = int(parts[4])
                        angle_min = float(parts[5])
                        angle_increment = float(parts[6])
                        n = int(parts[7])
                        
                        if len(parts) >= 8 + n:
                            ranges = [float(parts[8 + i]) for i in range(n)]
                            
                            # Guardar scan más reciente
                            self.latest_scan = {
                                'ranges': ranges,
                                'angle_min': angle_min,
                                'angle_increment': angle_increment,
                                'timestamp': time.time(),
                                'num_points': n
                            }
                            
                            self.scan_count += 1
                            self.last_scan_time = time.time()
                            
                            # FPS
                            fps_count += 1
                            if time.time() - fps_time >= 2.0:
                                fps = fps_count / (time.time() - fps_time)
                                print(f"[LIDAR-RX] Scans: {self.scan_count} | "
                                      f"FPS: {fps:.1f} | Puntos: {n}")
                                fps_count = 0
                                fps_time = time.time()
                    
                    except (ValueError, IndexError) as e:
                        print(f"[LIDAR-RX] Error parseando SCAN: {e}")
                        continue
                
                elif parts[0] == "HELLO":
                    # Responder ACK para mantener conexión
                    ack = f"ACK {self.desired_domain_id} PC_LIDAR".encode("utf-8")
                    await loop.sock_sendto(self.sock, ack, self.robot_addr)
                
                # Pequeña pausa para no saturar CPU
                await asyncio.sleep(0.001)
        
        except asyncio.CancelledError:
            print("[LIDAR-RX] 🛑 Recepción detenida")
        finally:
            if self.sock:
                self.sock.close()


# ====== FUNCIONES DE PROCESAMIENTO ======

def process_lidar_data(scan_data: dict) -> dict:
    """
    Procesa datos crudos del LiDAR y extrae información útil
    
    Args:
        scan_data: Dict con 'ranges', 'angle_min', 'angle_increment'
    
    Returns:
        Dict con información procesada:
        - 'objects': Lista de objetos detectados
        - 'closest': Objeto más cercano
        - 'front_clear': Bool, si el frente está despejado
    """
    if not scan_data:
        return {'objects': [], 'closest': None, 'front_clear': True}
    
    ranges = scan_data['ranges']
    angle_min = scan_data['angle_min']
    angle_inc = scan_data['angle_increment']
    
    # Encontrar objetos en sectores
    objects = []
    
    # Sector frontal: ±30° (aproximadamente)
    front_angles = []
    front_distances = []
    
    for i, r in enumerate(ranges):
        angle = angle_min + i * angle_inc
        
        # Solo puntos válidos (0.1m - 10m)
        if not (0.1 < r < 10.0 and math.isfinite(r)):
            continue
        
        # Sector frontal: -30° a +30° (≈ -0.52 a +0.52 rad)
        if -0.52 < angle < 0.52:
            front_angles.append(angle)
            front_distances.append(r)
            
            # Objeto detectado
            objects.append({
                'angle': angle,
                'distance': r,
                'angle_deg': math.degrees(angle)
            })
    
    # Objeto más cercano en el frente
    closest = None
    if front_distances:
        min_dist = min(front_distances)
        min_idx = front_distances.index(min_dist)
        closest = {
            'angle': front_angles[min_idx],
            'distance': min_dist,
            'angle_deg': math.degrees(front_angles[min_idx])
        }
    
    # ¿Frente despejado? (>1m de distancia mínima)
    front_clear = (closest is None) or (closest['distance'] > 1.0)
    
    return {
        'objects': objects,
        'closest': closest,
        'front_clear': front_clear,
        'num_objects': len(objects)
    }


def get_search_hint_from_lidar(scan_data: dict) -> Optional[float]:
    """
    Obtiene sugerencia de dirección de búsqueda basada en LiDAR
    Útil cuando se pierde el objetivo visual
    
    Args:
        scan_data: Datos del LiDAR
    
    Returns:
        Ángulo sugerido (rad) o None si no hay pista
    """
    if not scan_data:
        return None
    
    processed = process_lidar_data(scan_data)
    
    if processed['closest']:
        # Hay algo cerca - podría ser el objetivo
        return processed['closest']['angle']
    
    return None


# ====== PRUEBA STANDALONE ======

async def test_standalone():
    """Prueba del receptor en modo standalone"""
    print("=" * 70)
    print("🔵 RECEPTOR DE LIDAR - PRUEBA STANDALONE")
    print("=" * 70)
    print()
    
    # Configuración
    ROBOT_IP = "10.182.184.101"
    LIDAR_PORT = 6001  # Puerto DIFERENTE al de imágenes (6000)
    PAIRING_CODE = "ROBOT_A_11"
    DOMAIN_ID = 1
    
    receiver = LidarReceiverUDP(ROBOT_IP, LIDAR_PORT, PAIRING_CODE, DOMAIN_ID)
    
    # Conectar
    if not await receiver.connect_async():
        print("[ERROR] No se pudo conectar al robot")
        print("\n💡 ASEGÚRATE DE QUE EL ROBOT ESTÉ EJECUTANDO:")
        print("   python3 enviador_lidar.py")
        return
    
    print("\n✅ Conexión establecida")
    print("📊 Procesando scans en tiempo real...\n")
    
    # Crear tarea de recepción
    receive_task = asyncio.create_task(receiver.receive_loop_async())
    
    # Tarea de procesamiento (cada 1 segundo muestra info)
    try:
        while True:
            await asyncio.sleep(1.0)
            
            if receiver.latest_scan:
                processed = process_lidar_data(receiver.latest_scan)
                
                if processed['closest']:
                    obj = processed['closest']
                    print(f"🎯 Objeto cercano: {obj['distance']:.2f}m @ {obj['angle_deg']:.1f}°")
                
                if not processed['front_clear']:
                    print("⚠️  FRENTE BLOQUEADO")
                else:
                    print("✅ Frente despejado")
    
    except KeyboardInterrupt:
        print("\n\n[MAIN] Deteniendo...")
        receive_task.cancel()
        try:
            await receive_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    print("\n🧪 Modo de prueba standalone")
    print("Para usar con autonomo_async.py, importa las clases\n")
    
    try:
        asyncio.run(test_standalone())
    except KeyboardInterrupt:
        print("\n[MAIN] Sistema detenido")
