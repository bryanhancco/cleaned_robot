import threading
import time
from typing import Optional

# Módulos
try:
    from features.number_recognition import numbers_repaso as numbers_repaso_mod
    from features.number_recognition import numbers_practica as numbers_practica_mod
except Exception:
    numbers_repaso_mod = None
    numbers_practica_mod = None

try:
    from features.figures_recognition import figures_practica as figures_practica_mod
except Exception:
    figures_practica_mod = None
    print("xd")

try:
    from features.color_recognition import color_repaso as color_mod
except Exception:
    color_mod = None
    
try:
    from features.direction_recognition import direction_practica as direction_practica_mod
except Exception:
    direction_practica_mod = None

active_thread: Optional[threading.Thread] = None
active_stop_event: Optional[threading.Event] = None
active_name: Optional[str] = None

# registro de tiempos para evitar reintentos rápidos
_last_start_time: dict = {}


def start_module(name: str):
    """Inicia un módulo en un hilo separado."""
    global active_thread, active_stop_event, active_name

    # Evitar múltiples hilos
    if active_thread is not None and active_thread.is_alive():
        print(f"Ya hay un módulo activo: {active_name}. Deténlo antes de iniciar {name}.")
        return

    # Evitar reintentos muy rápidos
    now = time.time()
    last = _last_start_time.get(name, 0)
    if now - last < 1.0:
        print(f"Ignorando intento rápido de iniciar {name} (debounce).")
        return

    stop_ev = threading.Event()

    # Seleccionar módulo y función
    module_obj = None
    target = None
    
    if name == 'n_repaso' and numbers_repaso_mod is not None:
        module_obj = numbers_repaso_mod
        target = getattr(module_obj, 'run', None)
    elif name == 'n_practica' and numbers_practica_mod is not None:
        module_obj = numbers_practica_mod
        target = getattr(module_obj, 'run_interactive', None) or getattr(module_obj, 'run', None)
    elif name == 'fg_practica' and figures_practica_mod is not None:
        module_obj = figures_practica_mod
        target = getattr(module_obj, 'run_interactive', None) or getattr(module_obj, 'main', None)
    elif name == 'd_practica' and direction_practica_mod is not None:
        module_obj = direction_practica_mod
        target = getattr(module_obj, 'run', None) or getattr(module_obj, 'jugar_direcciones', None)
    elif name == 'c' and color_mod is not None:
        module_obj = color_mod
        target = getattr(module_obj, 'run', None) or getattr(module_obj, 'main', None)
    else:
        print(f"Módulo '{name}' no disponible o no importado correctamente.")
        return

    if getattr(module_obj, '_is_running', False):
        print(f"El módulo {name} ya está corriendo (flag interno). No se iniciará otro.")
        return

    if target is None:
        print(f"El módulo '{name}' no tiene función ejecutable.")
        return

    def runner():
        try:
            # Intentar con distintos parámetros posibles
            try:
                target(stop_ev)
            except TypeError:
                target()
        except Exception as e:
            print(f"Error en módulo {name}: {e}")
        finally:
            print(f"Módulo {name} finalizó su ejecución.")

    th = threading.Thread(target=runner, daemon=True)
    active_thread = th
    active_stop_event = stop_ev
    active_name = name
    _last_start_time[name] = now
    th.start()
    print(f"Módulo {name} iniciado.")


def stop_active_module(name: str):
    """Detiene un módulo activo si coincide el nombre."""
    global active_thread, active_stop_event, active_name

    if active_thread is None or not active_thread.is_alive():
        print("No hay ningún módulo activo.")
        return

    if active_name != name:
        print(f"El módulo activo es {active_name}; para detenerlo usa '{active_name}_salida'.")
        return

    if active_stop_event is not None:
        active_stop_event.set()
    
    active_thread.join(timeout=5)
    if active_thread.is_alive():
        print("El módulo no se detuvo dentro del tiempo esperado.")
    else:
        print(f"Módulo {name} detenido correctamente.")

    active_thread = None
    active_stop_event = None
    active_name = None


def handle_message(message: str):
    """Interpreta y ejecuta comandos de texto."""
    message = message.strip().lower()
    print(f"> Comando recibido: {message}")

    if message == 'n_repaso':
        start_module('n_repaso')
    elif message == 'n_practica':
        start_module('n_practica')
    elif message == 'fg_practica':
        start_module('fg_practica')
    elif message == 'd_practica':
        start_module('d_practica')
    elif message == 'c_repaso':
        start_module('c')
    elif message == 'n_repaso_salida':
        stop_active_module('n_repaso')
    elif message == 'n_practica_salida':
        stop_active_module('n_practica')
    elif message == 'fg_salida':
        stop_active_module('fg')
    elif message == 'c_salida':
        stop_active_module('c')
    elif message == 'd_salida':
        stop_active_module('d_practica')
    elif message in ('salir', 'exit', 'quit'):
        print("Saliendo del programa...")
        if active_stop_event:
            active_stop_event.set()
        return False
    else:
        print(f"Comando desconocido: {message}")
    return True


def main():
    print("=== Aplicación educativa DigitalHub Perú ===")
    print("Modo consola activado (sin conexión por socket).")
    print("Comandos disponibles:")
    print("  - n_repaso        → Reconocimiento continuo de números")
    print("  - n_practica      → Modo interactivo de práctica de números")
    print("  - fg_practica     → Reconocimiento de figuras")
    print("  - d_practica      → Practica de direcciones (mano izquierda/derecha)")
    print("  - c_repaso        → Reconocimiento de colores")
    print("  - [modulo]_salida → Detener módulo activo")
    print("  - salir           → Finalizar el programa")
    print("============================================")

    while True:
        try:
            cmd = input("\n> Ingresa comando: ").strip()
            if not handle_message(cmd):
                break
        except (KeyboardInterrupt, EOFError):
            print("\nInterrupción detectada. Saliendo...")
            if active_stop_event:
                active_stop_event.set()
            break

    print("Programa finalizado.")


if __name__ == '__main__':
    main()