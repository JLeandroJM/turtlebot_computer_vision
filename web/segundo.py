#!/usr/bin/env python3

import asyncio
import websockets
import json
import base64
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge
import cv2


class WsSender(Node):
    def __init__(self):
        super().__init__('ws_sender')
        self.bridge = CvBridge()
        self.latest_scan = None
        self.latest_image = None

        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(
            Image, '/oakd/rgb/preview/image_raw', self.image_callback, 10)

        self.get_logger().info('WsSender iniciado')

    def scan_callback(self, msg):
        self.latest_scan = {
            'type': 'scan',
            'ranges': [round(r, 2) for r in msg.ranges[::10]]
        }

    def image_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            cv_img = cv2.resize(cv_img, (320, 240))
            _, jpeg = cv2.imencode(
                '.jpg', cv_img, [cv2.IMWRITE_JPEG_QUALITY, 70])
            self.latest_image = {
                'type': 'image',
                'data': base64.b64encode(jpeg.tobytes()).decode('ascii')
            }
        except Exception as e:
            self.get_logger().error(f'Error en imagen: {e}')


ws_sender = None
connected_clients = set()


async def handler(websocket):
    connected_clients.add(websocket)
    print(f"Cliente conectado: {websocket.remote_address}")
    try:
        while True:
            if ws_sender.latest_image:
                await websocket.send(json.dumps(ws_sender.latest_image))
            if ws_sender.latest_scan:
                await websocket.send(json.dumps(ws_sender.latest_scan))
            await asyncio.sleep(0.033)
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.remove(websocket)
        print(f"Cliente desconectado")


async def main():
    global ws_sender
    rclpy.init()
    ws_sender = WsSender()

    async def spin_ros():
        while rclpy.ok():
            rclpy.spin_once(ws_sender, timeout_sec=0.01)
            await asyncio.sleep(0.01)

    async def start_server():
        async with websockets.serve(handler, "0.0.0.0", 8765):
            print("WebSocket servidor escuchando en puerto 8765")
            await asyncio.Future()

    await asyncio.gather(spin_ros(), start_server())

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Detenido")