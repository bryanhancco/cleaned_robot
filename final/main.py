import threading
from connection.connection_manager import check_device, setup_adb_forward, start_socket_connection
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

def start_module(name: str):
    global active_thread, active_stop_event, active_name

    # Evitar múltiples hilos
    if active_thread is not None and active_thread.is_alive():
        print(f"Ya hay un módulo activo: {active_name}. Deténlo antes de iniciar {name}.")
        return

    stop_ev = threading.Event()

    # Seleccionar módulo y función
    if name == 'n' and numbers_mod is not None:
        if getattr(numbers_mod, '_is_running', False):
            print("El módulo numbers ya está corriendo. No se puede iniciar otro.")
            return
        target = getattr(numbers_mod, 'run', None)
    elif name == 'fg' and figures_mod is not None:
        target = getattr(figures_mod, 'run', None)
    elif name == 'c' and color_mod is not None:
        target = getattr(color_mod, 'run', None)
    else:
        print(f"Módulo '{name}' no disponible.")
        return

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
    print("Iniciando aplicación educativa DigitalHub Perú...")

    if not check_device():
        print("Conecta un dispositivo Android con depuración USB activada y vuelve a intentar.")
        return

    setup_adb_forward()
    sock = start_socket_connection()

    print("Conexión establecida. Esperando comandos desde Flutter...")
    listen_socket(sock)
    sock.close()
    print("Aplicación finalizada.")

if __name__ == '__main__':
    main()
