# 🚀 Guía de Instalación Rápida

Esta guía te llevará desde cero hasta tener el sistema funcionando en **15 minutos**.

---

## ⚡ Instalación Express (PC)

```bash
# 1. Clonar repositorio
git clone <tu-repo-url>
cd turtlebot4-pursuit

# 2. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Verificar instalación
python3 -c "import cv2, numpy, ultralytics; print('✅ Todo listo')"
```

---

## 🤖 Configuración del Robot

### Opción A: Copiar scripts por SSH

```bash
# Desde tu PC
scp turtle/enviador.py ubuntu@<ROBOT_IP>:~/
scp turtle/enviador_lidar.py ubuntu@<ROBOT_IP>:~/
```

### Opción B: Clonar en el robot

```bash
# SSH al robot
ssh ubuntu@<ROBOT_IP>

# Clonar
git clone <tu-repo-url>
cd turtlebot4-pursuit/turtle

# Las dependencias ya están en el robot (ROS2 incluye todo)
```

---

## 🎮 Primera Ejecución

### Terminal 1: Robot - Imágenes
```bash
ssh ubuntu@<ROBOT_IP>
python3 enviador.py
```

### Terminal 2: Robot - LiDAR
```bash
ssh ubuntu@<ROBOT_IP>
python3 enviador_lidar.py
```

### Terminal 3: PC - Control
```bash
python3 autonomo_async.py
```

**¡Listo!** Deberías ver 2 ventanas (cámara + LiDAR) y el robot persiguiendo.

---

## 🔧 Configuración Mínima

Solo necesitas cambiar **1 línea** en `autonomo_async.py`:

```python
# Línea 85
ROBOT_IP = "10.182.184.101"  # ← Cambiar por la IP de tu robot
```

Para encontrar la IP del robot:
```bash
ssh ubuntu@turtlebot4  # o como lo tengas configurado
hostname -I
```

---

## ✅ Verificación

Si todo funciona correctamente verás:

### En la consola:
```
[HANDSHAKE] ✅ Conectado con 'turtlebot4_lite_11'
[MODEL] ✅ Modelo YOLOv8 cargado exitosamente
[MAIN] ✅ 6 tareas iniciadas en paralelo (CON LiDAR)
[CTRL] 🎯 SIGUIENDO ⚡ PERSIGUIENDO | v=0.45 m/s, w=-0.15 rad/s
```

### Ventana 1 - Cámara:
- Bounding box verde alrededor del objetivo
- FPS mostrando ~30 RX, ~15 YOLO, ~30 CMD
- Latencia < 100ms

### Ventana 2 - LiDAR:
- Puntos verdes/rojos actualizándose
- Robot (triángulo amarillo) en el centro
- Sectores frontales resaltados

---

## 🐛 Problemas Comunes

### "No module named 'ultralytics'"
```bash
pip install ultralytics torch
```

### "Connection refused" o "Timeout"
```bash
# Verificar conexión
ping <ROBOT_IP>

# Verificar que los scripts del robot estén corriendo
ssh ubuntu@<ROBOT_IP>
ps aux | grep enviador
```

### "CUDA not available" (normal)
El sistema funciona en CPU. Para usar GPU (opcional):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Robot no se mueve
Verificar que el robot no esté en modo "pausa" (botón central del Create3).

---

## 📚 Siguiente Paso

Lee el [README.md](README.md) completo para entender todos los estados y configuraciones avanzadas.

---

## 💡 Tips Rápidos

- **Presiona Q** en cualquier ventana para detener
- **Ctrl+C** en la consola para salida de emergencia
- Logs en tiempo real muestran el estado actual
- Verde en cámara = detectando objetivo
- Cyan = navegando con memoria
- Rojo = obstáculo cerca

---

## 🎯 Escenario de Prueba

1. Coloca el robot a ~2m del objetivo
2. Ejecuta el sistema
3. Mueve el objetivo lentamente
4. Observa cómo el robot persigue
5. Coloca una caja entre ellos
6. Observa cómo rodea la caja

**¡Disfruta!** 🚀
