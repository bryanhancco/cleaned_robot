import os
import cv2
import numpy as np
import time

try:
    import tensorflow as tf
except Exception:
    tf = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'mnist_cnn_model.h5')

model = None
_is_running = False
cap = None  # referencia global a la cámara


def run(stop_event=None, socket_conn=None):
    global _is_running, model, cap

    if _is_running:
        print("⚠️ El detector ya está en ejecución. Ignorando nueva instancia.")
        return

    _is_running = True
    print("▶ Iniciando detector de números...")

    if tf is None:
        print("TensorFlow no está disponible.")
        _is_running = False
        return

    if model is None:
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            _is_running = False
            return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara.")
        _is_running = False
        return

    print("Detector activo. Presiona 'q' o envía comando de salida.")

    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])

    try:
        while _is_running:
            if stop_event is not None and stop_event.is_set():
                print("🛑 Stop event recibido.")
                break

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_green, upper_green)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                plausible_contours = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    x, y, w, h = cv2.boundingRect(contour)
                    if 1000 < area < 150000:
                        aspect_ratio = float(w) / h if h != 0 else 0
                        if 0.6 < aspect_ratio < 1.3:
                            plausible_contours.append((area, contour, (x, y, w, h)))

                if plausible_contours:
                    plausible_contours.sort(key=lambda x: x[0], reverse=True)
                    _, largest_contour, (x, y, w, h) = plausible_contours[0]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    roi = mask[y:y + h, x:x + w]

                    if roi.size > 0 and np.max(roi) > 0:
                        target_size = 28
                        padding = 4
                        max_dim = max(roi.shape[0], roi.shape[1])
                        scale_factor = min((target_size - 2 * padding) / max_dim, 1.0)
                        resized_roi = cv2.resize(
                            roi,
                            (int(roi.shape[1] * scale_factor),
                             int(roi.shape[0] * scale_factor)),
                            interpolation=cv2.INTER_AREA
                        )

                        final_roi = np.zeros((target_size, target_size), dtype=np.uint8)
                        start_x = (target_size - resized_roi.shape[1]) // 2
                        start_y = (target_size - resized_roi.shape[0]) // 2
                        final_roi[start_y:start_y + resized_roi.shape[0],
                                  start_x:start_x + resized_roi.shape[1]] = resized_roi
                        _, final_roi = cv2.threshold(final_roi, 127, 255, cv2.THRESH_BINARY)

                        inp = final_roi.astype('float32') / 255.0
                        inp = np.expand_dims(inp, axis=(0, -1))

                        preds = model.predict(inp, verbose=0)
                        digit = int(np.argmax(preds))
                        confidence = float(np.max(preds))

                        # 👉 Mostrar en ventana
                        cv2.putText(
                            frame,
                            f"{digit} ({confidence * 100:.1f}%)",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )

                        # 👉 Mostrar también en la consola
                        print(f"Número detectado: {digit} | Confianza: {confidence * 100:.1f}%")
                        
                        # 👉 Enviar el número detectado por socket a Flutter
                        try:
                            if socket_conn:
                                message = f"{digit}\n"
                                socket_conn.sendall(message.encode())
                                print(f"📤 Enviado a Flutter: {message.strip()}")
                            else:
                                print("⚠️ No se recibió socket_conn. No se puede enviar el número.")
                        except Exception as e:
                            print(f"❌ Error al enviar el número a Flutter: {e}")

            # Mostrar la imagen siempre
            cv2.imshow("Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Tecla 'q' pulsada. Saliendo del detector.")
                break

    except Exception as e:
        print("Error en detector:", e)

    finally:
        print("⏹ Liberando recursos del detector...")
        _is_running = False
        if cap is not None:
            cap.release()
            cap = None
        cv2.destroyAllWindows()
        time.sleep(0.2)
        print("✅ Detector detenido correctamente.")