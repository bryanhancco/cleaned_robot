import cv2
import numpy as np
import time
import random

# --- Interfaz por consola (sin audio) ---
def get_input(prompt="Tu respuesta: "):
    """Lee la entrada del usuario desde stdin y la normaliza.
    Devuelve la cadena en minúsculas ('' si el usuario sólo presiona Enter)."""
    try:
        texto = input(prompt)
        return texto.lower().strip()
    except Exception:
        return ""

# --- Dibujar figuras simples ---

def dibujar_figura(nombre, size=400):
    h = size
    w = size
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    center = (w // 2, h // 2)
    color = (0, 0, 0)
    thickness = 6

    if nombre == 'círculo' or nombre == 'circulo':
        cv2.circle(img, center, size//4, color, thickness)
    elif nombre == 'cuadrado':
        side = size//2
        top_left = (center[0]-side//2, center[1]-side//2)
        bottom_right = (center[0]+side//2, center[1]+side//2)
        cv2.rectangle(img, top_left, bottom_right, color, thickness)
    elif nombre == 'rectángulo' or nombre == 'rectangulo':
        wrect = int(size*0.6)
        hrect = int(size*0.4)
        top_left = (center[0]-wrect//2, center[1]-hrect//2)
        bottom_right = (center[0]+wrect//2, center[1]+hrect//2)
        cv2.rectangle(img, top_left, bottom_right, color, thickness)
    elif nombre == 'triángulo' or nombre == 'triangulo':
        pts = np.array([[center[0], center[1]-size//4], [center[0]-size//4, center[1]+size//4], [center[0]+size//4, center[1]+size//4]], np.int32)
        cv2.polylines(img, [pts], True, color, thickness)
    else:
        cv2.putText(img, 'Figura', (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 2)

    return img

# Sólo permitir estas cuatro figuras (variantes sin tilde incluidas)
ALLOWED_SHAPES = {"círculo","circulo","cuadrado","triángulo","triangulo","rectángulo","rectangulo"}


def filtrar_figura_detectada(nombre):
    """Normaliza y permite sólo círculo/cuadrado/triángulo/rectángulo; devuelve 'desconocida' si no."""
    if not nombre:
        return "desconocida"
    n = nombre.lower().strip()
    # normalizar acentos básicos
    n = n.replace("á","a").replace("í","i").replace("ó","o").replace("ú","u").replace("é","e")
    # comprobar permitido
    if n in ALLOWED_SHAPES:
        if "triang" in n:
            return "triángulo"
        if "cuadr" in n:
            return "cuadrado"
        if "rect" in n:
            return "rectángulo"
        if "cir" in n:
            return "círculo"
    return "desconocida"

# --- Modo: adivinar (muestra figura, el usuario escribe el nombre) ---

def iniciar_modo_adivinar():
    print("¡Hola! Vamos a jugar a adivinar. Yo te mostraré una figura en pantalla y tú me dirás su nombre.")
    print("Escribe 'listo' para empezar o 'salir' para volver al prompt.")

    while True:
        resp = get_input("Escribe 'listo' para comenzar o 'salir' para salir: ")
        if "listo" in resp:
            break
        elif "salir" in resp:
            print("Volviendo al menú.")
            return
        else:
            print("No entendí tu entrada. Por favor escribe 'listo' o 'salir'.")

    figuras = ['cuadrado', 'círculo', 'triángulo', 'rectángulo']
    for i in range(3):
        objetivo = random.choice(figuras)
        img = dibujar_figura(objetivo)
        ventana = 'Adivina la figura'
        cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
        cv2.imshow(ventana, img)
        cv2.setWindowProperty(ventana, cv2.WND_PROP_TOPMOST, 1)

        print(f"Mira la figura en la pantalla. ¿Qué figura es?")
        # Mantener la ventana visible y permitir que el usuario escriba su respuesta
        for _ in range(10):
            if cv2.getWindowProperty(ventana, cv2.WND_PROP_VISIBLE) < 1:
                break
            cv2.waitKey(100)

        respuesta = get_input("Escribe el nombre de la figura que viste: ")
        cv2.destroyWindow(ventana)
        cv2.waitKey(1)

        if respuesta == "":
            print("No ingresaste texto.")
            continue

        # Normalizar y verificar
        resp_norm = respuesta.replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u')
        objetivo_norm = objetivo.replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u')

        if objetivo_norm in resp_norm:
            print("¡Correcto! Muy bien.")
        else:
            print(f"Casi. Era un {objetivo}.")
        time.sleep(1)

    print("Hemos terminado el modo adivinar. ¡Buen trabajo!")


if __name__ == '__main__':
    try:
        iniciar_modo_adivinar()
    except KeyboardInterrupt:
        print("Interrumpido por teclado. Saliendo.")
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass