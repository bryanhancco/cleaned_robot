import cv2
import mediapipe as mp
import numpy as np
import time
import random
from typing import Optional

# Globals
_is_running = False
cap = None

# MediaPipe helpers (module-level is fine; no heavy allocation)
mp_mano = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # Configuraciones para el funcionamiento de mp_hands


def _draw_debug_overlays(frame, hand_landmarks):
    """Dibuja sobre el mismo frame los marcadores y la línea usada para calcular dirección.

    Esto evita crear múltiples ventanas y mantiene una sola ventana de visualización.
    """
    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_mano.HAND_CONNECTIONS)
    hand_landmarks = hand_landmarks.landmark
    height, width, _ = frame.shape

    # Dibujar círculos para 4,0,12 y línea entre 0 y 8
    for landmark_id in (4, 0, 12):
        lm = hand_landmarks[landmark_id]
        cx, cy = int(lm.x * width), int(lm.y * height)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    lm0 = hand_landmarks[0]
    lm8 = hand_landmarks[8]
    cx0, cy0 = int(lm0.x * width), int(lm0.y * height)
    cx8, cy8 = int(lm8.x * width), int(lm8.y * height)
    cv2.line(frame, (cx0, cy0), (cx8, cy8), (0, 255, 0), 2)


def jugar_direcciones(rounds=4, timeout=15, hold_time=1.0, stop_event: Optional[object] = None, socket_conn: Optional[object] = None):
    """Juego interactivo: pedir mano izquierda/derecha en varias rondas.

    - stop_event: objeto con is_set() para detener desde fuera.
    - socket_conn: socket-like opcional para enviar resultados.
    """
    global _is_running, cap

    if _is_running:
        print("⚠️ El juego de direcciones ya está en ejecución. Ignorando nueva llamada.")
        return

    _is_running = True

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara. Asegúrate de que esté disponible.")
        _is_running = False
        return

    cv2.namedWindow("Seguimiento", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Seguimiento", 1280, 720)

    manos = mp_mano.Hands()

    try:
        print("Vamos a jugar. Te pediré que levantes la mano derecha o izquierda.")
        time.sleep(0.6)

        for r in range(rounds):
            objetivo = random.choice(["Derecha", "Izquierda"])
            print(f"Ronda {r+1}. Levanta la mano {objetivo.lower()}. Tienes {timeout} segundos.")
            start = time.time()
            hold_start = None
            success = False
            direction = None

            while time.time() - start < timeout and cap.isOpened() and _is_running:
                if stop_event is not None and hasattr(stop_event, 'is_set') and stop_event.is_set():
                    print("🛑 Stop event recibido. Saliendo del juego.")
                    return

                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resultado = manos.process(frame_rgb)

                direction = None
                if resultado.multi_hand_landmarks:
                    # Tomamos la primera mano detectada para decidir dirección
                    first_landmarks = resultado.multi_hand_landmarks[0]
                    _draw_debug_overlays(frame, first_landmarks)
                    lm0 = first_landmarks.landmark[0]
                    lm8 = first_landmarks.landmark[8]
                    if lm8.x > lm0.x:
                        direction = "Derecha"
                    else:
                        direction = "Izquierda"
                    cv2.putText(frame, f'Direccion: {direction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # lógica de hold: aceptar cuando la dirección pedida se mantiene continuamente
                if direction == objetivo:
                    if hold_start is None:
                        hold_start = time.time()
                    elapsed = time.time() - hold_start
                    cv2.putText(frame, f"Holding: {elapsed:.1f}/{hold_time}s", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,0), 2)
                    if elapsed >= hold_time:
                        success = True
                        print("¡Bien hecho! Correcto.")
                        # enviar por socket si corresponde
                        if socket_conn is not None:
                            try:
                                socket_conn.sendall(f"Ronda {r+1}: OK\n".encode())
                            except Exception as e:
                                print(f"❌ Error al enviar por socket: {e}")
                        break
                else:
                    hold_start = None

                secs_left = int(timeout - (time.time() - start))
                cv2.putText(frame, f"Tiempo: {secs_left}s  Objetivo: {objetivo}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                cv2.imshow('Seguimiento', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Saliendo del juego por tecla 'q'.")
                    return

            if not success:
                if direction is None:
                    print("No detecté tu mano. Intenta acercarte o mejorar la iluminación.")
                else:
                    print(f"Casi. Yo vi la mano en {direction.lower()}. ¡Sigue intentándolo!")
            time.sleep(0.8)

        print("Hemos terminado las rondas. ¡Buen trabajo!")

    except Exception as e:
        print("Error en jugar_direcciones():", e)

    finally:
        _is_running = False
        try:
            manos.close()
        except Exception:
            pass
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        cv2.destroyAllWindows()


def run(stop_event: Optional[object] = None, socket_conn: Optional[object] = None, rounds: int = 4, timeout: int = 15, hold_time: float = 1.0):
    """Interfaz pública similar a numbers_repaso.run()

    Llama a `jugar_direcciones` con los parámetros provistos.
    """
    return jugar_direcciones(rounds=rounds, timeout=timeout, hold_time=hold_time, stop_event=stop_event, socket_conn=socket_conn)


if __name__ == '__main__':
    try:
        run()
    finally:
        # asegurar liberación si el usuario ejecuta directamente
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
