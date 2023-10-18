import socket

SOCKET = socket.socket()
HOST = socket.gethostname()
PORT = 48620

SOCKET.connect((HOST, PORT))

print(SOCKET.recv(1024))

SOCKET.close()
