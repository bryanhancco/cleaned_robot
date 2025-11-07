import cv2
import mediapipe as mp
import numpy as np
import time
import random
import os

cv2.namedWindow("Seguimiento1", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Seguimiento1", 1280, 720)

mp_mano = mp.solutions.hands
mano = mp_mano.Hands()
mp_drawing = mp.solutions.drawing_utils #Configuraciones de para el funcionamiento de mp_hands

cap = cv2.VideoCapture(0) #apertura de la camara 

# Juego interactivo: 4 rondas pidiendo mano izquierda/derecha (sin audio)
def jugar_direcciones(rounds=4, timeout=15, hold_time=1.0):
    opciones = ["Derecha", "Izquierda"]
    print("Vamos a jugar. Te pediré que levantes la mano derecha o izquierda. Son cuatro rondas.")
    time.sleep(0.6)

    for r in range(rounds):
        objetivo = random.choice(opciones)
        print(f"Ronda {r+1}. Levanta la mano {objetivo.lower()}. Tienes {timeout} segundos.")
        start = time.time()
        hold_start = None
        success = False

        # loop de detección por ronda (usa la misma lógica presente más abajo)
        while time.time() - start < timeout and cap.isOpened():
            ret, frame = cap.read() #frame normal
            if not ret:
                continue

            frame2 = frame.copy() #frame2 pintando landmakers 4, 0 y 12
            frame3 = frame.copy() #frame3 pintando la linea entre el 8 y 0 
            frame4 = frame.copy() #frame4 pintando todos los landmakers de la mano

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultado = mano.process(frame_rgb)

            direction = None
            if resultado.multi_hand_landmarks:
                for landmarks in resultado.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame4, landmarks, mp_mano.HAND_CONNECTIONS)
                    hand_landmarks = landmarks.landmark
                    for landmark_id in [4, 0, 12]:
                        landmark = hand_landmarks[landmark_id]
                        height, width, _ = frame.shape
                        cx, cy = int(landmark.x * width), int(landmark.y * height)
                        cv2.circle(frame2, (cx, cy), 5, (0, 255, 0), -1)

                    landmark_0 = hand_landmarks[0]
                    landmark_8 = hand_landmarks[8]
                    height, width, _ = frame.shape
                    cx0, cy0 = int(landmark_0.x * width), int(landmark_0.y * height)
                    cx8, cy8 = int(landmark_8.x * width), int(landmark_8.y * height)
                    cv2.line(frame3, (cx0, cy0), (cx8, cy8), (0, 255, 0), 2)

                    # Determinar la dirección de la línea (misma lógica original)
                    if landmark_8.x > landmark_0.x:
                        direction = "Derecha"
                    else:
                        direction = "Izquierda"

                    cv2.putText(frame3, f'Direccion: {direction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            ventana1 = np.hstack((frame, frame4))
            ventana2 = np.hstack((frame2, frame3))
            ventana_final = np.vstack((ventana1, ventana2))

            # lógica de hold: aceptar cuando la dirección pedida se mantiene continuamente
            if direction == objetivo:
                if hold_start is None:
                    hold_start = time.time()
                elapsed = time.time() - hold_start
                cv2.putText(ventana_final, f"Holding: {elapsed:.1f}/{hold_time}s", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,0), 2)
                if elapsed >= hold_time:
                    success = True
                    print("¡Bien hecho! Correcto.")
                    break
            else:
                hold_start = None

            secs_left = int(timeout - (time.time() - start))
            cv2.putText(ventana_final, f"Tiempo: {secs_left}s  Objetivo: {objetivo}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
            cv2.imshow('Seguimiento1', ventana_final)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Saliendo del juego.")
                return

        if not success:
            if direction is None:
                print("No detecté tu mano. Intenta acercarte o mejorar la iluminación.")
            else:
                print(f"Casi. Yo vi la mano en {direction.lower()}. ¡Sigue intentándolo!")
        time.sleep(0.8)

    print("Hemos terminado las rondas. ¡Buen trabajo!")


if __name__ == '__main__':
    try:
        jugar_direcciones(rounds=4, timeout=15, hold_time=1.0)
    finally:
        cap.release()
        cv2.destroyAllWindows()
