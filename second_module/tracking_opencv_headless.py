import cv2
import mediapipe as mp
import lgpio
import time
import os
import argparse
import logging
import threading
from pathlib import Path

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


def set_angle(chip, pin, angle):
	"""Convierte un ángulo (0–180°) en un pulso PWM y lo envía al pin.
	Este método activa el PWM por un corto periodo y lo detiene.
	"""
	# Normalizar ángulo
	angle = max(0, min(180, float(angle)))
	pulse = 0.0005 + (angle / 180.0) * 0.002
	duty = (pulse / 0.02) * 100
	try:
		lgpio.tx_pwm(chip, pin, FREQ, duty)
		# Mantener el pulso un breve instante
		time.sleep(0.03)
		# Detener PWM para liberar pin (comportamiento igual que en `tracking_opencv.py`)
		lgpio.tx_pwm(chip, pin, 0, 0)
	except Exception as e:
		logging.exception(f"Error enviando PWM al pin {pin}: {e}")


def limit(val, min_val, max_val):
	return max(min_val, min(max_val, val))


def parse_args():
	p = argparse.ArgumentParser(description="Face tracking headless para Raspberry Pi (MediaPipe + servos)")
	p.add_argument('--camera', '-c', type=int, default=0, help='Índice del dispositivo de cámara')
	p.add_argument('--width', type=int, default=640, help='Ancho del frame')
	p.add_argument('--height', type=int, default=480, help='Alto del frame')
	p.add_argument('--save-frames', action='store_true', help='Guardar frames anotados en ./frames para depuración')
	p.add_argument('--debug', action='store_true', help='Modo debug (logs más verbosos)')
	p.add_argument('--reopen-threshold', type=int, default=8, help='Cantidad de lecturas fallidas consecutivas antes de reabrir la cámara')
	p.add_argument('--reopen-delay', type=float, default=1.0, help='Segundos a esperar antes de reintentar abrir la cámara')
	p.add_argument('--use-v4l2', action='store_true', help='Forzar backend V4L2 al abrir la cámara (útil en Linux/RPi)')
	p.add_argument('--threaded', action='store_true', help='Usar lectura de cámara en hilo (reduce bloqueos de select)')
	p.add_argument('--process-every', type=int, default=2, help='Procesar detección cada N frames (reduce carga)')
	p.add_argument('--max-fps', type=float, default=10.0, help='Limitar tasa máxima de procesamiento (0 = sin límite)')
	return p.parse_args()


def main():
	args = parse_args()

	# Configurar logging
	logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
						format='[%(asctime)s] %(levelname)s: %(message)s')

	if args.save_frames:
		os.makedirs('frames', exist_ok=True)

	# Estado de los servos
	servo1_angle = DEFAULT_SERVO1
	servo2_angle = DEFAULT_SERVO2
	last_move_time = 0

	# Inicializar GPIO (lgpio espera descriptor de chip)
	try:
		h = lgpio.gpiochip_open(CHIP)
		lgpio.gpio_claim_output(h, PIN_SERVO1)
		lgpio.gpio_claim_output(h, PIN_SERVO2)
		logging.info('GPIO inicializados correctamente.')
	except Exception as e:
		logging.exception('No se pudo inicializar GPIO. Asegúrese de ejecutar con permisos y en Raspberry Pi.')
		return

	# Posición inicial
	try:
		set_angle(h, PIN_SERVO1, servo1_angle)
		set_angle(h, PIN_SERVO2, servo2_angle)
		logging.info(f'Posición inicial: Servo1={servo1_angle}°, Servo2={servo2_angle}°')
	except Exception:
		logging.exception('Fallo al posicionar servos inicialmente')

	# Inicializar MediaPipe Face Detection
	mp_face_detection = mp.solutions.face_detection
	face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

	# Captura de video (headless: no se mostrará ventana)
	if args.use_v4l2:
		cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
	else:
		cap = cv2.VideoCapture(args.camera)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
	if args.max_fps and args.max_fps > 0:
		# intentar fijar FPS en la cámara cuando sea posible
		cap.set(cv2.CAP_PROP_FPS, float(args.max_fps))

	# Clase simple para lectura en hilo (almacena el último frame válido)
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
				# sleep un poco para no saturar hilo
				time.sleep(0.005)

		def read(self):
			with self.lock:
				return self.latest

		def stop(self):
			self.running = False
			if self.thread:
				self.thread.join(timeout=0.5)

	reader = None
	if args.threaded:
		reader = CameraReader(cap)
		reader.start()

	if not cap.isOpened():
		logging.error('No se pudo abrir la cámara. Verifique índice y permisos.')
		lgpio.gpio_free(h, PIN_SERVO1)
		lgpio.gpio_free(h, PIN_SERVO2)
		lgpio.gpiochip_close(h)
		return

	logging.info('Iniciando seguimiento facial en modo headless. Presione CTRL+C para detener.')

	frame_count = 0
	read_failures = 0
	reopen_attempts = 0
	process_counter = 0
	last_bbox = None

	try:
		while True:
			# Lectura (posible modo threaded)
			if args.threaded and reader is not None:
				success, frame = reader.read()
			else:
				try:
					success, frame = cap.read()
				except Exception:
					success, frame = False, None

			if not success or frame is None:
				read_failures += 1
				logging.warning('Lectura de cámara fallida (consecutivas=%d).', read_failures)

				# Si alcanzamos el umbral, intentamos reabrir la cámara
				if read_failures >= args.reopen_threshold:
					reopen_attempts += 1
					logging.info('Intentando reabrir la cámara (intento %d)...', reopen_attempts)
					try:
						try:
							cap.release()
						except Exception:
							pass
					except Exception:
						logging.exception('Error liberando cámara antes de reabrir')
					# esperar un poco y comprobar si el dispositivo existe (ej. /dev/video0)
					time.sleep(args.reopen_delay)
					dev_path = Path(f"/dev/video{args.camera}")
					if dev_path.exists():
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
								logging.info('Reapertura de cámara exitosa')
								read_failures = 0
								# reiniciar reader si usamos modo threaded
								if args.threaded:
									try:
										if reader:
											reader.stop()
									except Exception:
										pass
									reader = CameraReader(cap)
									reader.start()
							else:
								logging.warning('Reapertura de cámara fallida. Reintentando más tarde...')
						except Exception:
							logging.exception('Error intentando reabrir la cámara')
					else:
						logging.warning('%s no existe; esperando antes de nuevo intento', dev_path)

				# Esperar un poco antes del siguiente read para no saturar el bus
				time.sleep(0.5)
				continue
			else:
				# Lectura correcta → resetear contador de fallos
				if read_failures:
					logging.debug('Lectura de cámara recuperada tras %d fallos', read_failures)
				read_failures = 0

			frame_count += 1
			process_counter += 1
			frame_height, frame_width = frame.shape[:2]

			# Control de FPS máximo
			loop_start = time.time()

			# Decidir si ejecutar detección en este frame
			do_process = (args.process_every <= 1) or (process_counter % args.process_every == 0)

			if do_process:
				rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				results = face_detector.process(rgb)
			else:
				results = None

			now = time.time()

			if results and results.detections:
				detection = results.detections[0]
				bbox = detection.location_data.relative_bounding_box
				last_bbox = bbox
				x_center = bbox.xmin + bbox.width / 2
				y_center = bbox.ymin + bbox.height / 2
				cx = int(x_center * frame_width)
				cy = int(y_center * frame_height)

				# Control de movimiento
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

				# Limitar ángulos
				target_servo1 = limit(target_servo1, SERVO1_MIN, SERVO1_MAX)
				target_servo2 = limit(target_servo2, SERVO2_MIN, SERVO2_MAX)

				# Solo mover cada MOVE_INTERVAL segundos
				if (now - last_move_time) > MOVE_INTERVAL:
					try:
						set_angle(h, PIN_SERVO1, target_servo1)
						set_angle(h, PIN_SERVO2, target_servo2)
						logging.info(f"Movimiento aplicado → Servo1={int(target_servo1)}°, Servo2={int(target_servo2)}°")
					except Exception:
						logging.exception('Error moviendo servos')

					servo1_angle = target_servo1
					servo2_angle = target_servo2
					last_move_time = now

				# Si se activa guardado de frames, anotar y guardar
				if args.save_frames and (frame_count % 5 == 0):
					x1 = int(bbox.xmin * frame_width)
					y1 = int(bbox.ymin * frame_height)
					x2 = int((bbox.xmin + bbox.width) * frame_width)
					y2 = int((bbox.ymin + bbox.height) * frame_height)
					annotated = frame.copy()
					cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
					cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
					fname = os.path.join('frames', f'frame_{frame_count:06d}.jpg')
					cv2.imwrite(fname, annotated)

			else:
				# Si no hay rostro: volver a posición por defecto cada MOVE_INTERVAL
				if (now - last_move_time) > MOVE_INTERVAL and (servo1_angle != DEFAULT_SERVO1 or servo2_angle != DEFAULT_SERVO2):
					try:
						set_angle(h, PIN_SERVO1, DEFAULT_SERVO1)
						set_angle(h, PIN_SERVO2, DEFAULT_SERVO2)
						logging.info(f"Rostro perdido → Volviendo a posición central ({DEFAULT_SERVO1}, {DEFAULT_SERVO2})")
					except Exception:
						logging.exception('Error volviendo servos a posición por defecto')

					servo1_angle = DEFAULT_SERVO1
					servo2_angle = DEFAULT_SERVO2
					last_move_time = now

			# Pausa para limitar FPS (si se solicita)
			if args.max_fps and args.max_fps > 0:
				elapsed = time.time() - loop_start
				target = 1.0 / float(args.max_fps)
				if elapsed < target:
					time.sleep(target - elapsed)

	except KeyboardInterrupt:
		logging.info('Interrumpido por el usuario (KeyboardInterrupt)')

	finally:
		logging.info('Liberando recursos...')
		try:
			lgpio.gpio_free(h, PIN_SERVO1)
			lgpio.gpio_free(h, PIN_SERVO2)
			lgpio.gpiochip_close(h)
		except Exception:
			logging.exception('Error liberando GPIO')
		try:
			cap.release()
		except Exception:
			logging.exception('Error liberando cámara')
		logging.info('Finalizado.')


if __name__ == '__main__':
	main()

