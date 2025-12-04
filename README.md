# 🤖 TurtleBot4 Intelligent Pursuit System

Sistema autónomo de persecución visual con navegación inteligente para TurtleBot4 Lite. El robot persigue un objetivo móvil (perro robot) esquivando obstáculos dinámicos mediante fusión de sensores (cámara + LiDAR).

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Tabla de Contenidos

- [Características](#-características)
- [Arquitectura del Sistema](#-arquitectura-del-sistema)
- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Entrenamiento del Modelo](#-entrenamiento-del-modelo)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Configuración](#-configuración)
- [Resultados](#-resultados)
- [Troubleshooting](#-troubleshooting)
- [Contribuciones](#-contribuciones)
- [Licencia](#-licencia)

---

## 🌟 Características

### Detección Visual
- ✅ **YOLOv11s** entrenado custom (mAP50: 83.78%, Precision: 98.34%, Recall: 100%)
- ✅ Inferencia en tiempo real (~15 FPS)
- ✅ Detección robusta con alta confianza

### Navegación Inteligente
- 🧭 **Memoria espacial**: Estima distancia y ángulo del objetivo
- 🔄 **Wall-following**: Rodea obstáculos persistentemente
- 🎯 **5 estados**: TRACKING → EVADING → NAVIGATING → SEARCHING → LOST
- 📡 **Fusión de sensores**: Cámara + LiDAR para decisiones robustas

### Arquitectura Asíncrona
- ⚡ **Latencia ultra-baja**: 50-100ms (30x mejora vs secuencial)
- 🔀 **6 tareas paralelas**: RX Cámara, RX LiDAR, YOLO, Control, 2x Visualización
- 🚀 **30 Hz de comandos**: Control fluido sin lag
- 📊 **Visualización dual**: Ventana de cámara + ventana de LiDAR

### Robustez
- 🛡️ **Recuperación inteligente**: Encuentra objetivo tras oclusiones (2-4s)
- 🚧 **Evasión activa**: No se detiene, esquiva mientras mantiene vista
- 🌐 **Búsqueda dirigida**: Usa última posición conocida para navegar

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                      TURTLEBOT4 ROBOT                       │
├─────────────────────────────────────────────────────────────┤
│  ROS2 Nodes:                                                │
│  ├─ enviador.py         → UDP 6000 (Imágenes JPEG)         │
│  └─ enviador_lidar.py   → UDP 6001 (Scans LiDAR)          │
└────────────────────────┬────────────────────────────────────┘
                         │ WiFi / Ethernet
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    PC CONTROL (autonomo_async.py)           │
├─────────────────────────────────────────────────────────────┤
│  Tareas Asíncronas:                                         │
│                                                              │
│  1. RX Cámara (30 FPS)    ┐                                 │
│     └→ Queue(2) ──────────┼→ 2. YOLO (15 FPS)              │
│                           │     └→ Queue(3) ─────┐          │
│  3. RX LiDAR (10 FPS)    ─┤                      │          │
│     └→ Global var         │                      ↓          │
│                           └──────────→ 4. Control (30 Hz)   │
│                                          ├→ Navegación      │
│                                          └→ UDP 5007 (Twist)│
│                                                              │
│  5. Visualización Cámara  ←── Global state                  │
│  6. Visualización LiDAR   ←── Global LiDAR data            │
└─────────────────────────────────────────────────────────────┘
```

### Pipeline de Procesamiento

```
Imagen → Decodificación → YOLO → BBox → Control → Velocidades
  ↓                                 ↓        ↓
LiDAR → Sectores → Obstáculos → Memoria → Navegación
```

---

## 📦 Requisitos

### Hardware
- **Robot**: TurtleBot4 Lite (Create3 + Raspberry Pi 4 + RPLIDAR A1)
- **PC**: Laptop/Desktop con WiFi (macOS/Linux/Windows)
- **Conexión**: Misma red WiFi que el robot

### Software

#### En el Robot (TurtleBot4)
- Ubuntu 22.04
- ROS2 Humble
- Python 3.10+

#### En el PC
- Python 3.8+
- OpenCV 4.x
- NumPy
- Ultralytics (YOLOv11)
- asyncio (built-in)

---

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/turtlebot4-pursuit.git
cd turtlebot4-pursuit
```

### 2. Crear entorno virtual

```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install opencv-python numpy ultralytics torch
```

### 4. Verificar instalación

```bash
python3 -c "import cv2, numpy, ultralytics; print('✅ Instalación correcta')"
```

### 5. Configurar robot

Copiar scripts al robot:

```bash
# Desde tu PC
scp turtle/enviador.py ubuntu@<ROBOT_IP>:~/
scp turtle/enviador_lidar.py ubuntu@<ROBOT_IP>:~/
```

---

## 🎮 Uso

### Paso 1: Iniciar transmisores en el robot

```bash
# SSH al robot
ssh ubuntu@<ROBOT_IP>

# Terminal 1: Enviar imágenes
python3 enviador.py

# Terminal 2: Enviar LiDAR
python3 enviador_lidar.py
```

### Paso 2: Ejecutar sistema de control en PC

```bash
# En tu PC
python3 autonomo_async.py
```

### Paso 3: Observar el comportamiento

**Ventana 1 - Cámara**:
- Bounding boxes verdes en objetivo detectado
- Panel superior con métricas (FPS, latencia, estado)
- Modo de operación en tiempo real

**Ventana 2 - LiDAR**:
- Vista polar del entorno
- Puntos rojos = obstáculos cerca
- Puntos verdes = espacio libre
- X amarilla = objetivo estimado (cuando no visible)

**Consola**:
```
[CTRL] 🎯 SIGUIENDO ⚡ PERSIGUIENDO | v=0.45 m/s, w=-0.15 rad/s
[CTRL] 🧭 HACIA OBJETIVO 30° (1.8m) | v=0.30 m/s, w=-0.35 rad/s
[CTRL] 🔄 RODEANDO ⬅️ IZQUIERDA (obj 0.45m) | v=0.13 m/s, w=0.42 rad/s
```

### Controles

- **Q**: Cerrar visualización y detener robot
- **Ctrl+C**: Salida de emergencia

---

## 🎓 Entrenamiento del Modelo

### Dataset

1. **Captura**: Grabar video desde cámara del TurtleBot4
   ```bash
   # En el robot
   ros2 run image_view video_recorder image:=/oakd/rgb/preview/image_raw
   ```

2. **Procesamiento**: Extraer frames
   ```bash
   ffmpeg -i video.mp4 -vf fps=10 frames/frame_%04d.jpg
   ```

3. **Etiquetado**: Usar [Roboflow](https://roboflow.com)
   - Crear proyecto tipo "Object Detection"
   - Subir frames (~500 imágenes)
   - Anotar bounding boxes manualmente
   - Aplicar augmentation (flip, rotate, brightness)
   - Exportar en formato YOLOv11

### Entrenamiento

Usar el notebook proporcionado:

```bash
# En Google Colab con GPU
jupyter notebook train_yolo_colab.ipynb
```

O entrenar localmente:

```python
from ultralytics import YOLO

# Cargar modelo pre-entrenado
model = YOLO('yolo11s.pt')

# Entrenar
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='turtlebot_pursuit'
)

# Guardar mejor modelo
model.save('models_trained/best11s.pt')
```

### Validación

```bash
# Evaluar modelo
yolo val model=models_trained/best11s.pt data=data.yaml

# Inferencia en imagen de prueba
yolo predict model=models_trained/best11s.pt source=test/images/
```

---

## 📁 Estructura del Proyecto

```
turtlebot4-pursuit/
├── autonomo_async.py          # 🎯 Sistema principal de control
├── receptor_lidar.py           # 📡 Receptor standalone LiDAR (testing)
├── compare_performance.py      # 📊 Benchmark latencia
│
├── turtle/                     # 🤖 Scripts del robot
│   ├── enviador.py            # Transmisor de imágenes
│   └── enviador_lidar.py      # Transmisor de LiDAR
│
├── models_trained/             # 🧠 Modelos entrenados
│   ├── best11s.pt             # Modelo YOLOv11s final
│   └── training_results/      # Curvas de entrenamiento
│
├── other_models/               # 📦 Modelos auxiliares
│   ├── best11s.pt             # Backup del modelo
│   └── yolo11n.pt             # Modelo nano (más rápido)
│
├── data.yaml                   # ⚙️ Configuración del dataset
│
├── train/                      # 📚 Dataset entrenamiento (gitignore)
│   ├── images/
│   └── labels/
│
├── valid/                      # ✅ Dataset validación (gitignore)
│   ├── images/
│   └── labels/
│
├── test/                       # 🧪 Dataset prueba (gitignore)
│   ├── images/
│   └── labels/
│
├── backups/                    # 💾 Versiones anteriores
├── web/                        # 🌐 Experimentos web (deprecated)
│
├── NAVEGACION_INTELIGENTE.md  # 📖 Documentación navegación
├── README.md                   # 📄 Este archivo
├── .gitignore                  # 🚫 Archivos excluidos
└── requirements.txt            # 📋 Dependencias Python
```

---

## ⚙️ Configuración

### Parámetros del Robot

En `autonomo_async.py`:

```python
# Dirección IP del robot
ROBOT_IP = "10.182.184.101"  # Cambiar según tu robot

# Puertos de comunicación
IMG_PORT = 6000   # Imágenes
LIDAR_PORT = 6001 # LiDAR
CTRL_PORT = 5007  # Comandos

# Parámetros de control
MAX_LIN = 0.5     # Velocidad lineal máxima (m/s)
MAX_ANG = 0.7     # Velocidad angular máxima (rad/s)

# Umbrales de detección
CONFIDENCE_THRESHOLD = 0.25  # Confianza mínima YOLO
OBSTACLE_DISTANCE = 0.5      # Distancia de seguridad (m)
```

### Parámetros de Navegación

```python
# Tiempo de memoria espacial
MEMORY_TIMEOUT = 3.0  # segundos

# Velocidades de navegación
NAV_SPEED = 0.6        # 60% en navegación normal
WALL_FOLLOW_SPEED = 0.25  # 25% rodeando obstáculos

# Frecuencias de procesamiento
YOLO_RATE = 15    # FPS de inferencia
COMMAND_RATE = 30 # Hz de comandos
```

---

## 📊 Resultados

### Métricas del Modelo

| Métrica | Valor |
|---------|-------|
| mAP50 | 83.78% |
| Precision | 98.34% |
| Recall | 100% |
| Clases | 1 (perro robot) |
| FPS Inferencia | ~15 FPS |

### Rendimiento del Sistema

| Aspecto | Antes (Secuencial) | Ahora (Asíncrono) | Mejora |
|---------|-------------------|-------------------|--------|
| Latencia total | ~3000 ms | 50-100 ms | **30x** |
| Tasa de comandos | Variable | 30 Hz constante | ✅ |
| Recuperación tras pérdida | 8+ seg | 2-4 seg | **3x** |
| Éxito en recuperación | ~50% | ~90% | **+40%** |

### Comportamientos Validados

- ✅ Persecución fluida en línea recta
- ✅ Giro suave para centrar objetivo
- ✅ Evasión de obstáculos frontales
- ✅ Rodeo de cajas manteniendo vista
- ✅ Navegación hacia última posición tras oclusión
- ✅ Recuperación automática del objetivo
- ✅ Manejo de múltiples obstáculos

---

## 🐛 Troubleshooting

### Problema: Robot no se conecta

**Síntomas**: "Timeout" en handshake

**Soluciones**:
1. Verificar IP del robot: `ping <ROBOT_IP>`
2. Verificar misma red WiFi
3. Revisar firewall (permitir puertos 6000, 6001, 5007)
4. Reiniciar scripts del robot

### Problema: Latencia alta

**Síntomas**: FPS bajos, comandos lentos

**Soluciones**:
1. Cerrar otros programas pesados
2. Reducir `YOLO_RATE` (línea 157)
3. Usar modelo más ligero: `yolo11n.pt`
4. Verificar ancho de banda WiFi

### Problema: No detecta al objetivo

**Síntomas**: Bounding boxes no aparecen

**Soluciones**:
1. Verificar iluminación del entorno
2. Reducir `CONFIDENCE_THRESHOLD` (línea 163)
3. Acercarse más al objetivo
4. Re-entrenar con más imágenes

### Problema: Robot choca con obstáculos

**Síntomas**: No esquiva correctamente

**Soluciones**:
1. Aumentar `OBSTACLE_DISTANCE` a 0.7m (línea 166)
2. Verificar LiDAR funcionando: ventana "LiDAR Scan"
3. Reducir velocidades de navegación
4. Calibrar ángulos de sectores

### Problema: Pierde objetivo frecuentemente

**Síntomas**: Pasa a LOST rápidamente

**Soluciones**:
1. Aumentar tiempo de memoria a 5s (línea ~166 en `update_state()`)
2. Mejorar iluminación
3. Re-entrenar con más variedad de ángulos
4. Reducir velocidad de persecución

---

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

### Ideas para contribuir

- 🎯 Detección multi-objetivo
- 🗺️ SLAM para mapeo del entorno
- 🧠 Planificación global de rutas
- 📱 App móvil para control remoto
- 🎥 Grabación automática de datasets
- 🚀 Optimización con TensorRT

---

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

---

## 👥 Autores

- **Tu Nombre** - *Desarrollo inicial* - [tu-usuario](https://github.com/tu-usuario)

---

## 🙏 Agradecimientos

- [Ultralytics](https://github.com/ultralytics/ultralytics) por YOLOv11
- [TurtleBot4](https://turtlebot.github.io/turtlebot4-user-manual/) por la documentación
- [ROS2 Community](https://docs.ros.org/) por las herramientas
- [Roboflow](https://roboflow.com) por la plataforma de etiquetado

---

## 📚 Referencias

- [YOLO: Real-Time Object Detection](https://docs.ultralytics.com/)
- [ROS2 Navigation Stack](https://navigation.ros.org/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Python asyncio](https://docs.python.org/3/library/asyncio.html)

---

## 📧 Contacto

Para preguntas o colaboraciones:
- Email: tu-email@ejemplo.com
- GitHub Issues: [Crear issue](https://github.com/tu-usuario/turtlebot4-pursuit/issues)

---

<div align="center">
  
**⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub ⭐**

Hecho con ❤️ y 🤖

</div>
