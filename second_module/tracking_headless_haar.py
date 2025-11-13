import cv2
import time
import os
import argparse
import logging
import threading
from pathlib import Path

# Servo / GPIO: usamos lgpio si no estamos en modo simulación
try:
    import lgpio
except Exception:
    lgpio = None

# ===================== CONFIGURACIÓN SERVO =====================
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

# Intervalo entre actualizaciones de ángulo
MOVE_INTERVAL = 1.0   # segundos


def limit(val, min_val, max_val):
    return max(min_val, min(max_val, val))


def parse_args():
    p = argparse.ArgumentParser(description='Face tracking headless con Haar/LBP para Raspberry Pi')
    p.add_argument('--camera', '-c', type=int, default=0, help='Índice del dispositivo de cámara')
    p.add_argument('--width', type=int, default=640, help='Ancho del frame')
    p.add_argument('--height', type=int, default=480, help='Alto del frame')
    p.add_argument('--save-frames', action='store_true', help='Guardar frames anotados en ./frames para depuración')
    p.add_argument('--debug', action='store_true', help='Modo debug (logs más verbosos)')
    p.add_argument('--reopen-threshold', type=int, default=8, help='Lecturas fallidas antes de reabrir la cámara')
    p.add_argument('--reopen-delay', type=float, default=1.0, help='Segundos a esperar antes de reintentar abrir la cámara')
    p.add_argument('--use-v4l2', action='store_true', help='Forzar backend V4L2 al abrir la cámara')
    p.add_argument('--threaded', action='store_true', help='Usar lectura de cámara en hilo')
    p.add_argument('--process-every', type=int, default=2, help='Procesar detección cada N frames')
    p.add_argument('--max-fps', type=float, default=10.0, help='Limitar FPS de procesamiento (0 = sin límite)')
    p.add_argument('--detector', choices=['haar', 'lbp'], default='haar', help='Detector a usar: haar (default) o lbp')
    p.add_argument('--simulate-gpio', action='store_true', help='No usar lgpio, solo simular movimientos (útil para PC)')
    return p.parse_args()


def set_angle_lgpio(chip, pin, angle):
    """Envía PWM usando lgpio (si está disponible)."""
    angle = max(0, min(180, float(angle)))
    pulse = 0.0005 + (angle / 180.0) * 0.002
    duty = (pulse / 0.02) * 100
    lgpio.tx_pwm(chip, pin, FREQ, duty)
    time.sleep(0.03)
    lgpio.tx_pwm(chip, pin, 0, 0)


class CameraReader:
    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.latest = (False, None)

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            try:
                ok, frame = self.cap.read()
            except Exception:
                ok, frame = False, None
            with self.lock:
                self.latest = (ok, frame)
            time.sleep(0.005)

    def read(self):
        with self.lock:
            return self.latest

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format='[%(asctime)s] %(levelname)s: %(message)s')

    if args.save_frames:
        os.makedirs('frames', exist_ok=True)

    # Cargar cascada
    if args.detector == 'lbp':
        cascade_name = 'lbpcascade_frontalface.xml'
    else:
        cascade_name = 'haarcascade_frontalface_default.xml'
    cascade_path = os.path.join(cv2.data.haarcascades, cascade_name)
    if not os.path.exists(cascade_path):
        logging.error('No se encontró el archivo de cascada: %s', cascade_path)
        return
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Estado de servos
    servo1_angle = DEFAULT_SERVO1
    servo2_angle = DEFAULT_SERVO2
    last_move_time = 0

    # Inicializar GPIO (si no simulamos)
    simulate = args.simulate_gpio
    if not simulate:
        if lgpio is None:
            logging.error('lgpio no está disponible. Usa --simulate-gpio para ejecutar sin hardware')
            return
        try:
            h = lgpio.gpiochip_open(CHIP)
            lgpio.gpio_claim_output(h, PIN_SERVO1)
            lgpio.gpio_claim_output(h, PIN_SERVO2)
            logging.info('GPIO inicializados correctamente.')
        except Exception:
            logging.exception('No se pudo inicializar GPIO. Ejecuta con permisos adecuados o usa --simulate-gpio')
            return
    else:
        logging.info('Modo simulación GPIO activado; no se usará lgpio.')

    # función set_angle adaptada
    def set_angle(chip, pin, angle):
        if simulate:
            logging.debug('SIM - set_angle(%s, %s, %s)', chip, pin, angle)
            return
        set_angle_lgpio(chip, pin, angle)

    # Abrir cámara
    if args.use_v4l2:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.max_fps and args.max_fps > 0:
        cap.set(cv2.CAP_PROP_FPS, float(args.max_fps))

    reader = None
    if args.threaded:
        reader = CameraReader(cap)
        reader.start()

    if not cap.isOpened():
        logging.error('No se pudo abrir la cámara. Verifica índice y permisos.')
        if not simulate:
            try:
                lgpio.gpio_free(h, PIN_SERVO1)
                lgpio.gpio_free(h, PIN_SERVO2)
                lgpio.gpiochip_close(h)
            except Exception:
                pass
        return

    logging.info('Iniciando tracking con cascadas (%s). CTRL+C para salir.', args.detector)

    frame_count = 0
    read_failures = 0
    reopen_attempts = 0
    process_counter = 0

    try:
        while True:
            if args.threaded and reader is not None:
                success, frame = reader.read()
            else:
                try:
                    success, frame = cap.read()
                except Exception:
                    success, frame = False, None

            if not success or frame is None:
                read_failures += 1
                logging.warning('Lectura fallida (consecutivas=%d)', read_failures)
                if read_failures >= args.reopen_threshold:
                    reopen_attempts += 1
                    logging.info('Reintentando abrir cámara (intento %d)...', reopen_attempts)
                    try:
                        cap.release()
                    except Exception:
                        pass
                    time.sleep(args.reopen_delay)
                    dev = Path(f'/dev/video{args.camera}')
                    if dev.exists():
                        try:
                            if args.use_v4l2:
                                cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
                            else:
                                cap = cv2.VideoCapture(args.camera)
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
                            if args.max_fps and args.max_fps > 0:
                                cap.set(cv2.CAP_PROP_FPS, float(args.max_fps))
                            if cap.isOpened():
                                logging.info('Cámara reabierta correctamente')
                                read_failures = 0
                                if args.threaded:
                                    if reader:
                                        reader.stop()
                                    reader = CameraReader(cap)
                                    reader.start()
                        except Exception:
                            logging.exception('Error reabriendo cámara')
                    else:
                        logging.warning('%s no existe; esperando...', dev)
                time.sleep(0.5)
                continue

            # lectura ok
            read_failures = 0
            frame_count += 1
            process_counter += 1
            frame_height, frame_width = frame.shape[:2]
            loop_start = time.time()

            # procesar cada N frames
            do_process = (args.process_every <= 1) or (process_counter % args.process_every == 0)
            faces = []
            if do_process:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # escala, minNeighbors y minSize pueden ajustarse
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

            target_servo1 = servo1_angle
            target_servo2 = servo2_angle

            if len(faces) > 0:
                # seleccionar la cara más grande
                x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
                cx = x + w // 2
                cy = y + h // 2

                error_x = cx - frame_width // 2
                error_y = cy - frame_height // 2

                if abs(error_x) < 50 and abs(error_y) < 50:
                    target_servo1 = DEFAULT_SERVO1
                    target_servo2 = DEFAULT_SERVO2
                else:
                    kx = 0.05
                    ky = 0.05
                    target_servo2 = servo2_angle - error_x * kx
                    target_servo1 = servo1_angle + error_y * ky

                target_servo1 = limit(target_servo1, SERVO1_MIN, SERVO1_MAX)
                target_servo2 = limit(target_servo2, SERVO2_MIN, SERVO2_MAX)

                now = time.time()
                if (now - last_move_time) > MOVE_INTERVAL:
                    try:
                        set_angle(CHIP, PIN_SERVO1, target_servo1)
                        set_angle(CHIP, PIN_SERVO2, target_servo2)
                        logging.info('Movimiento → Servo1=%d°, Servo2=%d°', int(target_servo1), int(target_servo2))
                    except Exception:
                        logging.exception('Error moviendo servos')
                    servo1_angle = target_servo1
                    servo2_angle = target_servo2
                    last_move_time = now

                if args.save_frames and (frame_count % 5 == 0):
                    annotated = frame.copy()
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(annotated, (cx, cy), 3, (0, 0, 255), -1)
                    fname = os.path.join('frames', f'frame_{frame_count:06d}.jpg')
                    cv2.imwrite(fname, annotated)
            else:
                # no face detected -> volver suave a centro si corresponde
                now = time.time()
                if (now - last_move_time) > MOVE_INTERVAL and (servo1_angle != DEFAULT_SERVO1 or servo2_angle != DEFAULT_SERVO2):
                    try:
                        set_angle(CHIP, PIN_SERVO1, DEFAULT_SERVO1)
                        set_angle(CHIP, PIN_SERVO2, DEFAULT_SERVO2)
                        logging.info('Rostro perdido → Volviendo a centro')
                    except Exception:
                        logging.exception('Error volviendo servos a centro')
                    servo1_angle = DEFAULT_SERVO1
                    servo2_angle = DEFAULT_SERVO2
                    last_move_time = now

            # limitar FPS
            if args.max_fps and args.max_fps > 0:
                elapsed = time.time() - loop_start
                target = 1.0 / float(args.max_fps)
                if elapsed < target:
                    time.sleep(target - elapsed)

    except KeyboardInterrupt:
        logging.info('Interrumpido por el usuario')

    finally:
        logging.info('Liberando recursos...')
        if args.threaded and reader:
            try:
                reader.stop()
            except Exception:
                pass
        try:
            cap.release()
        except Exception:
            pass
        if not simulate:
            try:
                lgpio.gpio_free(h, PIN_SERVO1)
                lgpio.gpio_free(h, PIN_SERVO2)
                lgpio.gpiochip_close(h)
            except Exception:
                pass
        logging.info('Finalizado')


if __name__ == '__main__':
    main()
