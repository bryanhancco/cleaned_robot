import cv2
import mediapipe as mp
import lgpio
import time

# === CONFIGURACIÓN SERVO ===
CHIP = 4
PIN_SERVO1 = 14     # Vertical (arriba-abajo)
PIN_SERVO2 = 15     # Horizontal (izquierda-derecha)
FREQ = 50

# Límites de movimiento
SERVO1_MIN = 50
SERVO1_MAX = 165
SERVO2_MIN = 0
SERVO2_MAX = 179

# Posiciones por defecto (rostro centrado)
DEFAULT_SERVO1 = 135
DEFAULT_SERVO2 = 90

# === VARIABLES DE ESTADO ===
servo1_angle = DEFAULT_SERVO1
servo2_angle = DEFAULT_SERVO2
last_servo1 = DEFAULT_SERVO1
last_servo2 = DEFAULT_SERVO2
last_move_time = 0
MOVE_INTERVAL = 1.0   # segundos entre actualizaciones de ángulo

# === FUNCIONES ===
def set_angle(chip, pin, angle):
    """Convierte un ángulo (0–180°) en un pulso PWM."""
    pulse = 0.0005 + (angle / 180.0) * 0.002
    duty = (pulse / 0.02) * 100
    lgpio.tx_pwm(chip, pin, FREQ, duty)
    time.sleep(0.03)
    lgpio.tx_pwm(chip, pin, 0, 0)

def smooth_move(chip, pin, current_angle, target_angle, steps=10, delay=0.05):
    """Mueve el servo gradualmente hacia el ángulo objetivo."""
    step_size = (target_angle - current_angle) / steps
    for i in range(1, steps + 1):
        intermediate_angle = current_angle + step_size * i
        set_angle(chip, pin, intermediate_angle)
        time.sleep(delay)
    return target_angle

def limit(val, min_val, max_val):
    return max(min_val, min(max_val, val))

# === INICIALIZACIÓN GPIO ===
h = lgpio.gpiochip_open(CHIP)
lgpio.gpio_claim_output(h, PIN_SERVO1)
lgpio.gpio_claim_output(h, PIN_SERVO2)
print("GPIO inicializados correctamente.")

# Posición inicial
set_angle(h, PIN_SERVO1, servo1_angle)
set_angle(h, PIN_SERVO2, servo2_angle)
print(f"Posición inicial: Servo1={servo1_angle}°, Servo2={servo2_angle}°")

# === INICIALIZACIÓN MEDIAPIPE ===
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# === CAPTURA DE VIDEO ===
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

print("\nIniciando seguimiento facial... Presiona 'q' para salir.\n")

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("⚠️ Error: no se pudo acceder a la cámara.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb)

        frame_height, frame_width, _ = frame.shape
        now = time.time()

        # --- Si se detecta un rostro ---
        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            x_center = bbox.xmin + bbox.width / 2
            y_center = bbox.ymin + bbox.height / 2
            cx = int(x_center * frame_width)
            cy = int(y_center * frame_height)

            # Dibujar caja en pantalla
            cv2.rectangle(frame,
                          (int(bbox.xmin * frame_width), int(bbox.ymin * frame_height)),
                          (int((bbox.xmin + bbox.width) * frame_width),
                           int((bbox.ymin + bbox.height) * frame_height)),
                          (0, 255, 0), 2)

            # --- Control de movimiento ---
            error_x = cx - frame_width // 2
            error_y = cy - frame_height // 2

            # Si el rostro está casi centrado → posición por defecto
            if abs(error_x) < 50 and abs(error_y) < 50:
                target_servo1 = DEFAULT_SERVO1
                target_servo2 = DEFAULT_SERVO2
            else:
                kx = 0.05
                ky = 0.05
                target_servo2 = servo2_angle - error_x * kx
                target_servo1 = servo1_angle + error_y * ky

            # Limitar ángulos
            target_servo1 = limit(target_servo1, SERVO1_MIN, SERVO1_MAX)
            target_servo2 = limit(target_servo2, SERVO2_MIN, SERVO2_MAX)
            servo1_angle = target_servo1
            servo2_angle = target_servo2

            # Solo mover cada MOVE_INTERVAL segundos y si la diferencia > 10°
            if (now - last_move_time > MOVE_INTERVAL):

                # Movimiento suave
                set_angle(h, PIN_SERVO1, target_servo1)
                set_angle(h, PIN_SERVO2, target_servo2)
                # servo1_angle = set_angle(h, PIN_SERVO1, servo1_angle, target_servo1, steps=15, delay=0.03)
                # servo2_angle = set_angle(h, PIN_SERVO2, servo2_angle, target_servo2, steps=15, delay=0.03)

                print(f"[{time.strftime('%H:%M:%S')}] Movimiento suave aplicado → Servo1={int(target_servo1)}°, Servo2={int(target_servo2)}°")

                last_move_time = now
                last_servo1 = target_servo1
                last_servo2 = target_servo2

        else:
            # Si no hay rostro, volver a posición por defecto cada MOVE_INTERVAL segundos
            if now - last_move_time > MOVE_INTERVAL and (servo1_angle != DEFAULT_SERVO1 or servo2_angle != DEFAULT_SERVO2):
                set_angle(h, PIN_SERVO1, target_servo1)
                set_angle(h, PIN_SERVO2, target_servo2)
                #servo1_angle = set_angle(h, PIN_SERVO1, servo1_angle, DEFAULT_SERVO1, steps=20, delay=0.03)
                #servo2_angle = set_angle(h, PIN_SERVO2, servo2_angle, DEFAULT_SERVO2, steps=20, delay=0.03)
                print(f"[{time.strftime('%H:%M:%S')}] Rostro perdido → Volviendo suavemente a posición central ({DEFAULT_SERVO1}, {DEFAULT_SERVO2})")
                last_servo1, last_servo2 = DEFAULT_SERVO1, DEFAULT_SERVO2
                last_move_time = now

        # Mostrar video
        cv2.imshow("Face Tracking - MediaPipe + Servos", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n🛑 Interrumpido por el usuario.")

finally:
    print("\nLiberando recursos...")
    lgpio.gpio_free(h, PIN_SERVO1)
    lgpio.gpio_free(h, PIN_SERVO2)
    lgpio.gpiochip_close(h)
    cap.release()
    cv2.destroyAllWindows()
    print("✅ GPIO y cámara liberados.")