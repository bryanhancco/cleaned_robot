import os
import cv2
import numpy as np
import tensorflow as tf

# Obtener ruta absoluta al archivo del modelo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'mnist_cnn_model.keras')

# Cargar el modelo pre-entrenado
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error al cargar el modelo desde {MODEL_PATH}: {e}")
    print("Asegúrate de haber entrenado y guardado el modelo correctamente.")
    exit()

def run(stop_event=None):
    """Ejecuta el detector de números. Si se pasa stop_event (threading.Event), comprueba para terminar."""
    # Inicializar la cámara
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    print("Presiona 'q' para salir (o enviar comando de salida desde la app).")

    # Definir el rango de color verde limón en HSV
    # Estos valores están bien como punto de partida. AJÚSTALOS con el script de trackbars
    # para que la máscara (cv2.imshow('Mask', mask)) sea lo más limpia posible para tu "6".
    lower_green_lemon = np.array([35, 100, 100])
    upper_green_lemon = np.array([85, 255, 255])

    while True:
        if stop_event is not None and stop_event.is_set():
            print("Stop event recibido. Saliendo de numbers.run().")
            break

        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame de la cámara.")
            break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green_lemon, upper_green_lemon)

    # --- Ajustes Morfológicos para la Máscara ---
    # Un kernel más pequeño puede ayudar a preservar detalles, pero también puede dejar más ruido.
    # Un "OPEN" seguido de un "CLOSE" puede ser una buena combinación.
    kernel_m = np.ones((3,3), np.uint8) # Kernel un poco más pequeño
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_m, iterations=1) # Elimina ruido
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_m, iterations=1) # Cierra pequeños huecos


    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    import os
    import cv2
    import numpy as np
    import tensorflow as tf
    import time

    # Obtener ruta absoluta al archivo del modelo
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, 'mnist_cnn_model.keras')

    # Cargar el modelo pre-entrenado
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error al cargar el modelo desde {MODEL_PATH}: {e}")
        print("Asegúrate de haber entrenado y guardado el modelo correctamente.")
        model = None


    def run(stop_event=None):
        """Ejecuta el detector de números. Si se pasa stop_event (threading.Event), comprueba para terminar."""
        if model is None:
            print("Modelo no disponible. Abortando run().")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara.")
            return

        print("Presiona 'q' para salir (o enviar comando de salida desde la app).")

        # Rango de color verde limón en HSV
        lower_green_lemon = np.array([35, 100, 100])
        upper_green_lemon = np.array([85, 255, 255])

        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    print("Stop event recibido. Saliendo de numbers.run().")
                    break

                ret, frame = cap.read()
                if not ret:
                    # pequeño retardo para evitar CPU al 100% si la cámara falla
                    time.sleep(0.1)
                    continue

                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower_green_lemon, upper_green_lemon)

                kernel_m = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_m, iterations=1)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_m, iterations=1)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
                        if max_dim == 0:
                            # evitar división por cero
                            pass
                        else:
                            scale_factor = min((target_size - 2 * padding) / max_dim, 1.0)
                            resized_roi_w = int(roi.shape[1] * scale_factor)
                            resized_roi_h = int(roi.shape[0] * scale_factor)

                            if resized_roi_w > 0 and resized_roi_h > 0:
                                resized_roi = cv2.resize(roi, (resized_roi_w, resized_roi_h), interpolation=cv2.INTER_AREA)
                                final_roi = np.zeros((target_size, target_size), dtype=np.uint8)
                                start_x = (target_size - resized_roi_w) // 2
                                start_y = (target_size - resized_roi_h) // 2
                                final_roi[start_y:start_y + resized_roi_h, start_x:start_x + resized_roi_w] = resized_roi
                                _, final_roi = cv2.threshold(final_roi, 127, 255, cv2.THRESH_BINARY)

                                inp = final_roi.astype('float32') / 255.0
                                inp = np.expand_dims(inp, axis=-1)
                                inp = np.expand_dims(inp, axis=0)

                                preds = model.predict(inp, verbose=0)
                                digit = int(np.argmax(preds))
                                confidence = float(np.max(preds))

                                cv2.putText(frame, f"{digit} ({confidence*100:.1f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow('Mask', mask)
                cv2.imshow('Detector de Digitos Verdes (Refinado)', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Tecla 'q' pulsada. Saliendo del detector de números.")
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


    if __name__ == '__main__':
        run()
