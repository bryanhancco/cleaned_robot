import cv2
import numpy as np
import time
import random
import os

try:
    import tensorflow as tf
except Exception:
    tf = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'mnist_cnn_model.h5')

# --- Configuración global ---
model = None
_is_running = False
cap = None

# --- Detección de números ---
def load_model():
    """Carga el modelo de TensorFlow."""
    global model
    if model is None:
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print("✅ Modelo cargado correctamente")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            return False
    return True

def detect_digit_from_frame(frame):
    """Extrae la ROI por color, procesa y devuelve (digit, confidence, vis_frame, mask)."""
    lower_green_lemon = np.array([35, 100, 100])
    upper_green_lemon = np.array([85, 255, 255])
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green_lemon, upper_green_lemon)
    kernel_m = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_m, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_m, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0, frame, mask

    plausible_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if 1000 < area < 150000:
            aspect_ratio = float(w) / h if h>0 else 0
            if 0.6 < aspect_ratio < 1.3:
                plausible_contours.append((area, contour, (x, y, w, h)))

    if not plausible_contours:
        return None, 0.0, frame, mask

    plausible_contours.sort(key=lambda x: x[0], reverse=True)
    _, _, (x, y, w, h) = plausible_contours[0]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi = mask[y:y+h, x:x+w]
    if roi.size == 0 or np.max(roi) == 0:
        return None, 0.0, frame, mask

    # preparar ROI para el modelo
    target_size = 28
    padding = 4
    max_dim = max(roi.shape[0], roi.shape[1])
    if max_dim == 0:
        return None, 0.0, frame, mask
    scale_factor = min((target_size - 2 * padding) / max_dim, 1.0)
    resized_roi_w = int(roi.shape[1] * scale_factor)
    resized_roi_h = int(roi.shape[0] * scale_factor)
    if resized_roi_w == 0 or resized_roi_h == 0:
        return None, 0.0, frame, mask
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
    return digit, confidence, frame, mask

# --- Sesión interactiva ---
def interactive_session(rounds=5, timeout_sec=15, accept_conf=0.6, stop_event=None):
    """Ejecuta una sesión interactiva de detección de números."""
    global _is_running, cap
    
    if _is_running:
        print("⚠️ La sesión interactiva ya está en ejecución.")
        return
        
    _is_running = True
    
    if tf is None:
        print("TensorFlow no está disponible.")
        _is_running = False
        return
        
    if not load_model():
        _is_running = False
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara.")
        _is_running = False
        return

    try:
        print("🎮 Iniciando sesión interactiva...")
        print("Instrucciones: Mostrar el número objetivo en color verde dentro del tiempo límite.")
        time.sleep(0.8)
        
        for r in range(rounds):
            if stop_event and stop_event.is_set():
                print("🛑 Stop event recibido.")
                break
                
            objetivo = random.randint(0,9)
            print(f"🎯 Ronda {r+1}/{rounds}: Mostrar el número {objetivo} - Tienes {timeout_sec} segundos")
            
            start = time.time()
            best_pred = None
            best_conf = 0.0
            
            while time.time() - start < timeout_sec:
                if stop_event and stop_event.is_set():
                    break
                    
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                frame = cv2.flip(frame, 1)
                pred, conf, vis, mask = detect_digit_from_frame(frame)
                
                if pred is not None and conf > best_conf:
                    best_conf = conf
                    best_pred = pred
                
                # Mostrar cuenta regresiva
                secs_left = int(timeout_sec - (time.time() - start))
                cv2.putText(vis, f"Objetivo: {objetivo}  Tiempo: {secs_left}s", (10,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                cv2.imshow('Mask', mask)
                cv2.imshow('Detector interactivo', vis)
                
                if pred == objetivo and conf >= accept_conf:
                    print("✅ ¡Correcto! Número reconocido correctamente.")
                    break
                    
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("⏹ Sesión interrumpida por el usuario.")
                    break
            else:
                # Se ejecuta si no se hizo break (tiempo agotado)
                if best_pred == objetivo and best_conf >= accept_conf:
                    print("✅ ¡Correcto! Número reconocido correctamente.")
                else:
                    if best_pred is not None:
                        print(f"❌ No reconocido. Se detectó {best_pred} con {best_conf*100:.1f}% de confianza.")
                    else:
                        print("❌ No se detectó ningún número. Mejora la iluminación o acerca el objeto.")
            
            if stop_event and stop_event.is_set():
                break
                
            time.sleep(1.0)
        
        if not (stop_event and stop_event.is_set()):
            print("🎉 ¡Sesión completada! Buen trabajo.")
            
    except Exception as e:
        print(f"❌ Error en sesión interactiva: {e}")
        
    finally:
        stop_interactive()

def stop_interactive():
    """Detiene la sesión interactiva y libera recursos."""
    global _is_running, cap
    print("⏹ Deteniendo sesión interactiva...")
    _is_running = False
    if cap is not None:
        cap.release()
        cap = None
    cv2.destroyAllWindows()
    print("✅ Sesión interactiva detenida correctamente.")

# --- Función principal para ejecutar desde otros módulos ---
def run_interactive(stop_event=None, rounds=5, timeout_sec=15, accept_conf=0.6):
    """Función principal para ejecutar la sesión interactiva desde otros módulos."""
    interactive_session(rounds=rounds, timeout_sec=timeout_sec, accept_conf=accept_conf, stop_event=stop_event)

# --- Ejecución directa del script ---
if __name__ == "__main__":
    try:
        run_interactive(rounds=5, timeout_sec=15, accept_conf=0.6)
    except KeyboardInterrupt:
        print("👋 Sesión finalizada por el usuario")
    finally:
        stop_interactive()