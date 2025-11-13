import cv2
import numpy as np
import time
from typing import Optional

# Globals to allow safe start/stop from external modules (main_CONSOLA, main.py, etc.)
_is_running = False
cap = None  # cámara global, para asegurar liberación desde cualquier contexto


def clasificar_color(r: int, g: int, b: int) -> str:
    """Clasifica un color RGB en una de las categorías predefinidas.

    Mantiene la implementación original pero con anotaciones de tipo mínimas.
    """
    color_bgr = np.uint8([[[b, g, r]]])
    color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = color_hsv[0][0]

    min_saturacion_para_color = 50
    if s < min_saturacion_para_color:
        return "otro color"

    if 100 <= h <= 140:
        return "azul"
    if (0 <= h <= 10) or (165 <= h <= 179):
        return "rojo"
    if 20 <= h <= 40:
        return "amarillo"
    if 140 <= h <= 165:
        return "morado"
    if 10 <= h <= 20:
        return "naranja"
    if 80 <= h <= 100:
        return "celeste"
    if 40 <= h <= 80:
        return "verde"

    # fallback RGB
    colores_referencia_rgb = {
        "rojo": (255, 0, 0),
        "azul": (0, 0, 255),
        "amarillo": (255, 255, 0),
        "morado": (128, 0, 128),
        "naranja": (255, 165, 0),
        "celeste": (0, 191, 255),
        "verde": (0, 255, 0),
    }

    min_distancia = float('inf')
    nombre_color_cercano = "otro color"
    for nombre, (cr, cg, cb) in colores_referencia_rgb.items():
        distancia = np.sqrt((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2)
        if distancia < min_distancia:
            min_distancia = distancia
            nombre_color_cercano = nombre

    umbral_distancia = 60
    if min_distancia < umbral_distancia:
        return nombre_color_cercano
    else:
        return "otro color"


def run(stop_event: Optional[object] = None, socket_conn: Optional[object] = None) -> None:
    """Inicia el detector de color usando la cámara por defecto.

    Compatibilidades:
    - `stop_event`: objeto con `is_set()` para permitir parada desde hilos externos.
    - `socket_conn`: socket-like con `sendall(bytes)` para enviar datos (opcional).

    El diseño copia el patrón de `numbers_repaso.py`: evita múltiples instancias simultáneas,
    gestiona recursos globales y es seguro para importar en otros módulos.
    """
    global _is_running, cap

    if _is_running:
        print("⚠️ El detector de color ya está en ejecución. Ignorando nueva llamada a run().")
        return

    _is_running = True
    print("▶ Iniciando detector de color...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara. Asegúrate de que no esté en uso y de que los drivers estén instalados.")
        _is_running = False
        return

    try:
        try:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            print("Intentando desactivar auto-exposición y establecer brillo neutral en la cámara...")
        except Exception as e:
            print(f"No se pudo configurar la cámara (propiedades no soportadas o error): {e}")

        ancho_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        alto_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        print(f"Resolución de la cámara: {ancho_frame}x{alto_frame}")

        tamaño_cuadrado = 50
        grosor_linea = 2

        x_centro = ancho_frame // 2
        y_centro = alto_frame // 2

        x1_cuadrado = x_centro - (tamaño_cuadrado // 2)
        y1_cuadrado = y_centro - (tamaño_cuadrado // 2)
        x2_cuadrado = x_centro + (tamaño_cuadrado // 2)
        y2_cuadrado = y_centro + (tamaño_cuadrado // 2)

        print("Detector activo. Presiona 'q' en la ventana o establece stop_event desde otro hilo para detenerlo.")

        while _is_running:
            if stop_event is not None and hasattr(stop_event, 'is_set') and stop_event.is_set():
                print("🛑 Stop event recibido. Saliendo de color.run().")
                break

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            roi = frame[y1_cuadrado:y2_cuadrado, x1_cuadrado:x2_cuadrado]

            if roi.size == 0:
                # No terminamos la ejecución solamente porque la ROI esté vacía; esperamos y seguimos
                time.sleep(0.1)
                continue

            color_bgr_promedio = cv2.mean(roi)[:3]
            b, g, r = int(color_bgr_promedio[0]), int(color_bgr_promedio[1]), int(color_bgr_promedio[2])

            color_clasificado = clasificar_color(r, g, b)

            cv2.rectangle(frame, (x1_cuadrado, y1_cuadrado), (x2_cuadrado, y2_cuadrado), (255, 0, 0), grosor_linea)

            texto_color_rgb = f"RGB: ({r}, {g}, {b})"
            texto_color_clasificado = f"Color: {color_clasificado}"

            cv2.putText(frame, texto_color_rgb, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, texto_color_clasificado, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Envío opcional del color por socket en formato simple
            if socket_conn is not None:
                try:
                    message = f"{color_clasificado}:{r},{g},{b}\n"
                    socket_conn.sendall(message.encode())
                except Exception as e:
                    print(f"❌ Error al enviar por socket: {e}")

            cv2.imshow('Detector de Color en Camara', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Tecla 'q' pulsada. Saliendo del detector de color.")
                break

    except Exception as e:
        print("Error en color.run():", e)

    finally:
        print("⏹ Liberando recursos del detector de color...")
        _is_running = False
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        cv2.destroyAllWindows()
        time.sleep(0.1)
        print("✅ Detector de color detenido correctamente.")


if __name__ == '__main__':
    run()