import subprocess
import socket
import time


HOST = '127.0.0.1'
PORT = 5000


def check_device():
    """Verifica si hay un dispositivo Android conectado mediante ADB."""
    try:
        result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1 and 'device' in lines[1]:
            print("Dispositivo Android detectado.")
            return True
        print("No hay dispositivos conectados por ADB.")
        return False
    except Exception as e:
        print("Error ejecutando adb:", e)
        return False


def setup_adb_forward():
    """Configura el reenvío de puertos para la comunicación ADB."""
    try:
        subprocess.run(['adb', 'forward', f'tcp:{PORT}', 'tcp:6000'], check=True)
        print(f"Puertos enlazados correctamente (tcp:{PORT} ↔ tcp:6000).")
    except Exception as e:
        print("Error configurando ADB forward:", e)
        raise


def start_socket_connection():
    """Establece la conexión socket con Flutter."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Intentando conectar con Flutter...")
    while True:
        try:
            s.connect((HOST, PORT))
            print("Conexión establecida con Flutter.")
            return s
        except ConnectionRefusedError:
            print("Esperando conexión del otro extremo (Flutter)...")
            time.sleep(1)
        except Exception as e:
            print("Error al intentar conectar:", e)
            time.sleep(2)
