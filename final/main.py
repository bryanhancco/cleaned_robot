import threading
import importlib
from connection.connection_manager import check_device, setup_adb_forward, start_socket_connection


def handle_message(message: str):
    """Procesa los mensajes recibidos desde Flutter."""
    message = message.strip().lower()
    if message == "n_repaso":
        print("Ejecutando módulo de reconocimiento de números...")
        try:
            importlib.import_module('number_recognition.numbers')
        except Exception as e:
            print(f"Error al ejecutar el módulo de reconocimiento: {e}")
    else:
        print(f"Mensaje desconocido recibido: {message}")


def listen_socket(sock):
    """Escucha mensajes entrantes desde Flutter y los envía al manejador."""
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

    # Inicia el listener en el hilo principal (ya no hay interacción local)
    listen_socket(sock)

    sock.close()
    print("Aplicación finalizada.")


if __name__ == "__main__":
    main()
