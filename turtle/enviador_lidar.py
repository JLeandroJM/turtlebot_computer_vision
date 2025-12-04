#!/usr/bin/env python3
"""
enviador_lidar.py - ENVÍA SOLO DATOS LIDAR DEL TURTLEBOT4

Nodo ROS2 que envía datos del sensor LiDAR por UDP a la PC
Puerto separado del que envía imágenes para evitar congestión

PUERTO RECOMENDADO: 6001 (imágenes en 6000, lidar en 6001)

FORMATO DE MENSAJE:
SCAN <domain_id> <robot_name> <stamp_sec> <stamp_nsec> <angle_min> <angle_increment> <n> r1 r2 ... rn

USO:
    ros2 run <tu_paquete> enviador_lidar.py
    
O directamente:
    python3 enviador_lidar.py
"""
import os
import socket
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class UdpLidarSenderNode(Node):
    def __init__(self):
        super().__init__("udp_lidar_sender")

        # ========= Parámetros =========
        self.declare_parameter("port", 6001)  # Puerto DIFERENTE al de imágenes
        self.declare_parameter("robot_name", "turtlebot4_lite_11")
        self.declare_parameter("pairing_code", "ROBOT_A_11")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("send_rate", 10)  # Hz - limitar rate para no saturar red

        port = self.get_parameter("port").get_parameter_value().integer_value
        self.robot_name = self.get_parameter("robot_name").get_parameter_value().string_value
        self.pairing_code = self.get_parameter("pairing_code").get_parameter_value().string_value
        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        self.send_rate = self.get_parameter("send_rate").get_parameter_value().integer_value

        # ========= ROS_DOMAIN_ID =========
        self.ros_domain_id = int(os.environ.get("ROS_DOMAIN_ID", "1"))
        self.get_logger().info(f"ROS_DOMAIN_ID detectado: {self.ros_domain_id}")

        # ========= Socket UDP =========
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        self.get_logger().info(f"LiDAR UDP escuchando en 0.0.0.0:{port}")

        # ========= Estado de emparejamiento =========
        self.authorized_addr = None  # (ip, puerto) de la PC emparejada
        self.get_logger().info("Esperando HELLO para emparejar PC...")

        # ========= Subscripción a LiDAR =========
        self.sub_scan = self.create_subscription(
            LaserScan, 
            scan_topic, 
            self.scan_callback, 
            10
        )

        # ========= Control de rate =========
        self.last_send_time = 0.0
        self.min_interval = 1.0 / self.send_rate if self.send_rate > 0 else 0.0
        
        # Estadísticas
        self.scan_count = 0
        self.sent_count = 0

        # ========= Hilo UDP (para HELLO / ACK) =========
        self.running = True
        self.udp_thread = threading.Thread(target=self.udp_loop, daemon=True)
        self.udp_thread.start()

        self.get_logger().info(f"✅ Nodo LiDAR iniciado - Rate: {self.send_rate} Hz")

    # ================== Hilo UDP (HELLO/ACK) ==================
    def udp_loop(self):
        """Maneja handshake con la PC"""
        self.get_logger().info("Hilo UDP para handshake iniciado.")
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)
                text = data.decode("utf-8").strip()
                parts = text.split()

                if not parts:
                    continue

                cmd_type = parts[0]

                if cmd_type == "HELLO":
                    self.handle_hello(parts, addr)
                else:
                    # Ignorar otros mensajes
                    pass

            except Exception as e:
                self.get_logger().error(f"Error en udp_loop: {e}")
                break

        self.get_logger().info("Hilo UDP finalizado.")

    def handle_hello(self, parts, addr):
        """
        Maneja mensaje HELLO de la PC
        Formato: HELLO <desired_domain_id> <pairing_code>
        """
        if len(parts) < 3:
            self.get_logger().warn(f"HELLO inválido desde {addr}: {parts}")
            return

        desired_domain_str = parts[1]
        pairing_code = parts[2]

        try:
            desired_domain = int(desired_domain_str)
        except ValueError:
            self.get_logger().warn(f"HELLO con domain_id inválido: '{desired_domain_str}'")
            return

        # Verificar pairing code
        if pairing_code != self.pairing_code:
            self.get_logger().warn(f"HELLO con pairing_code incorrecto desde {addr}")
            return

        # Verificar domain_id
        if desired_domain != self.ros_domain_id:
            self.get_logger().warn(
                f"HELLO con domain_id {desired_domain} pero robot tiene {self.ros_domain_id}"
            )
            return

        # Aceptar emparejamiento (una sola PC)
        if self.authorized_addr is None:
            self.authorized_addr = addr
            self.get_logger().info(f"✅ PC emparejada: {addr}")
        else:
            if addr != self.authorized_addr:
                self.get_logger().warn(
                    f"HELLO desde {addr} pero ya hay PC: {self.authorized_addr}"
                )
                return

        # Responder ACK <domain_id> <robot_name>
        ack_msg = f"ACK {self.ros_domain_id} {self.robot_name}".encode("utf-8")
        self.sock.sendto(ack_msg, addr)

    # ================== Callback de LiDAR ==================
    def scan_callback(self, msg: LaserScan):
        """
        Callback que recibe datos del LiDAR y los envía por UDP
        """
        self.scan_count += 1

        # Sin PC emparejada, no enviar
        if self.authorized_addr is None:
            return

        # Control de rate (limitar FPS de envío)
        import time
        current_time = time.time()
        if current_time - self.last_send_time < self.min_interval:
            return  # Saltar este scan para no saturar red

        self.last_send_time = current_time

        try:
            # Extraer datos del scan
            ranges = list(msg.ranges)
            n = len(ranges)

            # Formato del mensaje:
            # SCAN <domain_id> <robot_name> <stamp_sec> <stamp_nsec> <angle_min> <angle_increment> <n> r1 r2 ... rn
            header = (
                f"SCAN {self.ros_domain_id} {self.robot_name} "
                f"{msg.header.stamp.sec} {msg.header.stamp.nanosec} "
                f"{msg.angle_min} {msg.angle_increment} {n}"
            )

            # Convertir rangos a string (con 3 decimales para reducir tamaño)
            ranges_str = " ".join(f"{r:.3f}" for r in ranges)

            # Mensaje completo
            text = f"{header} {ranges_str}"
            data = text.encode("utf-8")

            # Enviar por UDP
            self.sock.sendto(data, self.authorized_addr)
            self.sent_count += 1

            # Log cada 50 scans enviados
            if self.sent_count % 50 == 0:
                self.get_logger().info(
                    f"📡 Scans enviados: {self.sent_count} | "
                    f"Puntos: {n} | "
                    f"Rango: [{msg.range_min:.2f}, {msg.range_max:.2f}]m"
                )

        except Exception as e:
            self.get_logger().error(f"Error enviando SCAN: {e}")

    # ================== Cleanup ==================
    def destroy_node(self):
        """Limpieza al cerrar el nodo"""
        self.running = False
        try:
            self.sock.close()
        except Exception:
            pass
        
        self.get_logger().info(
            f"🛑 Nodo finalizado - Total scans enviados: {self.sent_count}/{self.scan_count}"
        )
        super().destroy_node()


def main(args=None):
    """Punto de entrada principal"""
    rclpy.init(args=args)
    node = UdpLidarSenderNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[LIDAR] Detenido por usuario")
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
