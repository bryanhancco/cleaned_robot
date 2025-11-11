import threading
import time
from typing import Optional

# Módulos
try:
    from features.number_recognition import numbers as numbers_mod
except Exception:
    numbers_mod = None

try:
    from features.figures_recognition import recognize_figures as figures_mod
except Exception:
    figures_mod = None

try:
    from features.color_recognition import color as color_mod
except Exception:
    color_mod = None

active_thread: Optional[threading.Thread] = None
active_stop_event: Optional[threading.Event] = None
active_name: Optional[str] = None
# registro de tiempos para evitar reintentos rápidos
_last_start_time: dict = {}

def start_module(name: str):
    global active_thread, active_stop_event, active_name

    # Evitar múltiples hilos
    if active_thread is not None and active_thread.is_alive():
        print(f"Ya hay un módulo activo: {active_name}. Deténlo antes de iniciar {name}.")
        return

    # evitar reintentos muy rápidos (debounce)
    now = time.time()
    last = _last_start_time.get(name, 0)
    if now - last < 1.0:
        print(f"Ignorando intento rápido de iniciar {name} (debounce).")
        return

    stop_ev = threading.Event()

    # Seleccionar módulo y función
    module_obj = None
    if name == 'n' and numbers_mod is not None:
        module_obj = numbers_mod
    elif name == 'fg' and figures_mod is not None:
        module_obj = figures_mod
    elif name == 'c' and color_mod is not None:
        module_obj = color_mod
    else:
        print(f"Módulo '{name}' no disponible.")
        return

    # comprobar si el módulo ya tiene un flag de ejecución
    if getattr(module_obj, '_is_running', False):
        print(f"El módulo {name} ya está corriendo (flag interno). No se iniciará otro.")
        return

    # obtener la función ejecutable preferida
    target = getattr(module_obj, 'run', None) or getattr(module_obj, 'main', None)

    if target is None:
        print(f"El módulo '{name}' no tiene función ejecutable.")
        return

    def runner():
        try:
            try:
                target(stop_ev)
            except TypeError:
                target()
        except Exception as e:
            print(f"Error en módulo {name}: {e}")

    th = threading.Thread(target=runner, daemon=True)
    active_thread = th
    active_stop_event = stop_ev
    active_name = name
    # marcar tiempo de inicio para debounce
    _last_start_time[name] = now
    th.start()
    print(f"Módulo {name} iniciado.")

def stop_active_module(name: str):
    global active_thread, active_stop_event, active_name

    if active_thread is None or not active_thread.is_alive():
        print("No hay ningún módulo activo.")
        return

    if active_name != name:
        print(f"El módulo activo es {active_name}; para detenerlo envía '{active_name}_salida'.")
        return

    if active_stop_event is not None:
        active_stop_event.set()
    active_thread.join(timeout=5)
    if active_thread.is_alive():
        print("El módulo no se detuvo dentro del tiempo esperado.")
    else:
        print(f"Módulo {name} detenido.")

    active_thread = None
    active_stop_event = None
    active_name = None

def handle_message(message: str):
    message = message.strip().lower()
    print(f"Comando recibido: {message}")

    if message == 'n_repaso':
        start_module('n')
    elif message == 'fg_repaso':
        start_module('fg')
    elif message == 'c_repaso':
        start_module('c')
    elif message == 'n_salida':
        stop_active_module('n')
    elif message == 'fg_salida':
        stop_active_module('fg')
    elif message == 'c_salida':
        stop_active_module('c')
    else:
        print(f"Comando desconocido: {message}")

def listen_socket(sock):
    while True:
        try:
            data = sock.recv(1024)
            if not data:
                print("Conexión cerrada por Flutter.")
                break
            message = data.decode().strip()
            if message:
                handle_message(message)
        except Exception as e:
            print("Error en la recepción de datos:", e)
            break

def main():
    print("Iniciando aplicación educativa DigitalHub Perú (modo consola)...")
    print("Introduce comandos por consola. Comandos válidos:\n  n_repaso, fg_repaso, c_repaso\n  n_salida, fg_salida, c_salida\nEscribe 'salir' o 'exit' para terminar.")

    try:
        while True:
            try:
                line = input('> ')
            except (EOFError, KeyboardInterrupt):
                print('\nInterrupción recibida. Saliendo...')
                break

            if not line:
                continue
            cmd = line.strip()
            if cmd.lower() in ('salir', 'exit', 'quit'):
                print('Saliendo...')
                break
            handle_message(cmd)

    finally:
        # intentar detener cualquier módulo activo antes de salir
        if active_thread is not None and active_thread.is_alive():
            print('Deteniendo módulo activo antes de salir...')
            if active_stop_event is not None:
                active_stop_event.set()
            active_thread.join(timeout=5)
        print('Aplicación finalizada (modo consola).')

if __name__ == '__main__':
    main()
