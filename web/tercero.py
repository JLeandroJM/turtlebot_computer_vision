#!/usr/bin/env python3

import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
import math
import gradio as gr
from threading import Thread
import time

latest_image = None
latest_lidar = None
connected = False


def draw_lidar(ranges, width=600, height=600):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    center_x = width // 2
    center_y = height // 2
    max_range = 3.0
    scale = min(width, height) // 2 / max_range

    cv2.circle(img, (center_x, center_y), 5, (0, 255, 0), -1)

    num_points = len(ranges)
    angle_increment = 2 * math.pi / num_points

    for i, r in enumerate(ranges):
        if r > 0.1 and r < max_range:
            angle = i * angle_increment
            x = int(center_x + r * scale * math.cos(angle))
            y = int(center_y - r * scale * math.sin(angle))
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

    cv2.circle(img, (center_x, center_y), int(1.0 * scale), (100, 100, 100), 2)
    cv2.circle(img, (center_x, center_y), int(2.0 * scale), (100, 100, 100), 2)

    cv2.putText(img, "1m", (center_x + int(1.0 * scale) + 10, center_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, "2m", (center_x + int(2.0 * scale) + 10, center_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


async def receive_from_robot():
    global latest_image, latest_lidar, connected
    uri = "ws://172.20.10.7:8765"

    while True:
        try:
            async with websockets.connect(uri) as websocket:
                connected = True
                print(f"✓ Conectado al TurtleBot4")
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)

                    if data['type'] == 'image':
                        img_bytes = base64.b64decode(data['data'])
                        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        if frame is not None:
                            latest_image = cv2.cvtColor(
                                frame, cv2.COLOR_BGR2RGB)

                    elif data['type'] == 'scan':
                        latest_lidar = draw_lidar(data['ranges'])

        except Exception as e:
            connected = False
            print(f"Error: {e}. Reconectando en 2s...")
            await asyncio.sleep(2)


def start_websocket():
    asyncio.run(receive_from_robot())


def update_camera():
    if latest_image is not None:
        return latest_image
    return np.zeros((240, 320, 3), dtype=np.uint8)


def update_lidar():
    if latest_lidar is not None:
        return latest_lidar
    return np.zeros((600, 600, 3), dtype=np.uint8)


def get_status():
    if connected and latest_image is not None and latest_lidar is not None:
        return "🟢 Conectado - Recibiendo datos"
    elif connected:
        return "🟡 Conectado - Esperando datos..."
    else:
        return "🔴 Desconectado"


ws_thread = Thread(target=start_websocket, daemon=True)
ws_thread.start()

with gr.Blocks(title="TurtleBot4 Monitor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 TurtleBot4 Live Monitor")

    status = gr.Textbox(label="Estado", value=get_status, every=1)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📷 Cámara")
            camera_output = gr.Image(
                label="Vista en vivo", every=0.05, value=update_camera)

        with gr.Column():
            gr.Markdown("### 📡 LIDAR")
            lidar_output = gr.Image(
                label="Vista en vivo", every=0.05, value=update_lidar)

if __name__ == "__main__":
    print("🌐 Abriendo interfaz Gradio...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)