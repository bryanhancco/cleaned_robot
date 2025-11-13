import socket
import threading

HOST = '127.0.0.1'
PORT = 5000

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

def receive():
    while True:
        try:
            data = s.recv(1024)
            if not data:
                break
            print("Mensaje desde Flutter:", data.decode().strip())
        except:
            break

# Hilo para escuchar sin bloquear
threading.Thread(target=receive, daemon=True).start()

# Loop principal: enviar datos manualmente o periódicamente
try:
    while True:
        msg = input("Enviar al Flutter: ")
        if msg.lower() == 'exit':
            break
        s.sendall((msg + "\n").encode())
except KeyboardInterrupt:
    pass
finally:
    s.close()
