#!/usr/bin/env python3
"""
lidar_receiver.py - Ejemplo de receptor de datos LiDAR para integrar con autonomo_web.py

Este script muestra cómo recibir datos del LiDAR del TurtleBot4 de dos formas:
1. Desde ROS2 (si está disponible)
2. Desde UDP (si el robot transmite por socket)

Para integrar con autonomo_web.py, copiar la función correspondiente.
"""

import socket
import struct
import time
import numpy as np

# ====== Configuración ======
ROBOT_IP = "10.182.184.104"
LIDAR_PORT = 6001  # Puerto para datos LiDAR (ajustar según configuración)


# ====== MÉTODO 1: Recibir LiDAR por UDP ======
def receive_lidar_udp():
    """
    Recibe datos del LiDAR por UDP
    Formato esperado: array de floats (distancias en metros)
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', LIDAR_PORT))
    sock.settimeout(1.0)
    
    print(f"[LIDAR UDP] Escuchando en puerto {LIDAR_PORT}...")
    
    try:
        while True:
            try:
                data, addr = sock.recvfrom(65536)  # Buffer grande para LiDAR
                
                # Decodificar datos
                # Formato: array de floats (4 bytes cada uno)
                num_points = len(data) // 4
                ranges = struct.unpack(f'{num_points}f', data)
                
                print(f"[LIDAR] Recibidos {len(ranges)} puntos desde {addr}")
                print(f"[LIDAR] Min: {min(ranges):.2f}m, Max: {max(ranges):.2f}m")
                
                # Retornar para usar en autonomo_web.py
                return ranges
                
            except socket.timeout:
                print("[LIDAR UDP] Timeout esperando datos...")
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\n[LIDAR UDP] Detenido")
    finally:
        sock.close()


# ====== MÉTODO 2: Recibir LiDAR por ROS2 ======
def receive_lidar_ros2():
    """
    Recibe datos del LiDAR mediante ROS2
    Requiere: rclpy, sensor_msgs
    
    Instalar:
    pip install rclpy sensor_msgs
    """
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import LaserScan
    except ImportError:
        print("[ERROR] ROS2 no disponible. Instalar: pip install rclpy sensor_msgs")
        return None
    
    class LidarReceiver(Node):
        def __init__(self):
            super().__init__('lidar_receiver')
            self.subscription = self.create_subscription(
                LaserScan,
                '/scan',
                self.lidar_callback,
                10
            )
            self.latest_ranges = None
            print("[LIDAR ROS2] Suscrito a /scan")
        
        def lidar_callback(self, msg):
            # Reducir resolución para eficiencia (tomar 1 de cada 10 puntos)
            self.latest_ranges = [r for i, r in enumerate(msg.ranges) if i % 10 == 0]
            
            print(f"[LIDAR ROS2] Recibidos {len(msg.ranges)} puntos (guardados {len(self.latest_ranges)})")
            print(f"[LIDAR ROS2] Min: {msg.range_min:.2f}m, Max: {msg.range_max:.2f}m")
    
    rclpy.init()
    receiver = LidarReceiver()
    
    try:
        print("[LIDAR ROS2] Esperando mensajes...")
        rclpy.spin(receiver)
    except KeyboardInterrupt:
        print("\n[LIDAR ROS2] Detenido")
    finally:
        receiver.destroy_node()
        rclpy.shutdown()


# ====== MÉTODO 3: Integración con autonomo_web.py ======
def integrate_with_autonomo_web():
    """
    Código de ejemplo para integrar en autonomo_web.py
    """
    
    example_code = """
# ========== AGREGAR AL INICIO DE autonomo_web.py ==========

# Variables globales para LiDAR
lidar_ranges = None
lidar_sock = None

def setup_lidar_udp():
    '''Configura socket para recibir LiDAR por UDP'''
    global lidar_sock
    lidar_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    lidar_sock.bind(('', 6001))  # Puerto LiDAR
    lidar_sock.settimeout(0.01)  # Non-blocking
    print("[LIDAR] Socket UDP configurado en puerto 6001")

def receive_lidar_data():
    '''Recibe datos del LiDAR (llamar en el loop principal)'''
    global lidar_ranges, lidar_sock
    
    if lidar_sock is None:
        return None
    
    try:
        data, _ = lidar_sock.recvfrom(65536)
        num_points = len(data) // 4
        lidar_ranges = struct.unpack(f'{num_points}f', data)
        return lidar_ranges
    except socket.timeout:
        pass
    except Exception as e:
        print(f"[LIDAR] Error: {e}")
    
    return lidar_ranges


# ========== MODIFICAR EN main() de autonomo_web.py ==========

def main():
    # ... código existente ...
    
    # Configurar LiDAR (AGREGAR después de crear sockets)
    setup_lidar_udp()
    
    # ... en el loop principal, ANTES de update_web_data() ...
    
    # Recibir datos del LiDAR
    current_lidar_ranges = receive_lidar_data()
    
    # Actualizar web con datos reales del LiDAR
    update_web_data(vis, metrics, lidar_ranges=current_lidar_ranges)
    
    # ... resto del código ...

# Al finalizar, cerrar socket LiDAR
finally:
    # ... otros close() ...
    if lidar_sock:
        lidar_sock.close()
"""
    
    print("="*70)
    print("📋 CÓDIGO PARA INTEGRAR EN autonomo_web.py")
    print("="*70)
    print(example_code)
    print("="*70)


# ====== MÉTODO 4: Simulador de datos LiDAR para pruebas ======
def simulate_lidar_data():
    """
    Genera datos simulados del LiDAR para pruebas
    Útil cuando no hay robot real disponible
    """
    import math
    import random
    
    num_points = 360  # 360 grados
    ranges = []
    
    for i in range(num_points):
        angle_deg = i
        
        # Simular entorno con paredes y obstáculos
        if 45 < angle_deg < 135:  # Pared frontal derecha
            distance = 2.0 + random.uniform(-0.1, 0.1)
        elif 225 < angle_deg < 315:  # Pared trasera
            distance = 3.0 + random.uniform(-0.1, 0.1)
        elif 80 < angle_deg < 100:  # Obstáculo cercano (robot perro?)
            distance = 1.0 + random.uniform(-0.2, 0.2)
        else:  # Espacio libre
            distance = 4.0 + random.uniform(-0.5, 0.5)
        
        # Limitar rango válido (LiDAR típico: 0.12m - 3.5m)
        distance = max(0.12, min(3.5, distance))
        ranges.append(distance)
    
    return ranges


# ====== MÉTODO 5: Transmitir LiDAR simulado por UDP (para pruebas) ======
def transmit_simulated_lidar():
    """
    Transmite datos LiDAR simulados por UDP
    Útil para probar autonomo_web.py sin robot real
    
    Ejecutar en una terminal:
    python3 lidar_receiver.py transmit
    
    Y en otra terminal:
    python3 autonomo_web.py
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target_addr = ('localhost', 6001)
    
    print(f"[LIDAR SIMULATOR] Transmitiendo datos simulados a {target_addr}")
    print("[LIDAR SIMULATOR] Presiona Ctrl+C para detener")
    
    try:
        while True:
            # Generar datos simulados
            ranges = simulate_lidar_data()
            
            # Empaquetar como floats
            data = struct.pack(f'{len(ranges)}f', *ranges)
            
            # Enviar
            sock.sendto(data, target_addr)
            
            print(f"[LIDAR SIMULATOR] Enviados {len(ranges)} puntos", end='\r')
            time.sleep(0.1)  # 10 Hz
            
    except KeyboardInterrupt:
        print("\n[LIDAR SIMULATOR] Detenido")
    finally:
        sock.close()


# ====== MAIN ======
def main():
    import sys
    
    print("="*70)
    print("🔴 RECEPTOR DE DATOS LIDAR - TurtleBot4")
    print("="*70)
    print("\nModos disponibles:")
    print("1. python3 lidar_receiver.py udp       - Recibir por UDP")
    print("2. python3 lidar_receiver.py ros2      - Recibir por ROS2")
    print("3. python3 lidar_receiver.py integrate - Mostrar código de integración")
    print("4. python3 lidar_receiver.py simulate  - Generar datos simulados")
    print("5. python3 lidar_receiver.py transmit  - Transmitir datos simulados por UDP")
    print("="*70)
    
    if len(sys.argv) < 2:
        print("\n❌ Especifica un modo. Ejemplo: python3 lidar_receiver.py udp")
        return
    
    mode = sys.argv[1].lower()
    
    if mode == 'udp':
        receive_lidar_udp()
    elif mode == 'ros2':
        receive_lidar_ros2()
    elif mode == 'integrate':
        integrate_with_autonomo_web()
    elif mode == 'simulate':
        ranges = simulate_lidar_data()
        print(f"\n✅ Generados {len(ranges)} puntos simulados")
        print(f"📊 Min: {min(ranges):.2f}m, Max: {max(ranges):.2f}m, Avg: {np.mean(ranges):.2f}m")
    elif mode == 'transmit':
        transmit_simulated_lidar()
    else:
        print(f"\n❌ Modo '{mode}' no reconocido")


if __name__ == '__main__':
    main()
