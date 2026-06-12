import serial
import time
import sys
import tty
import termios

ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
time.sleep(2)

def get_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return key

print("WASD로 조종하세요! q로 종료")
print("W: 전진 / S: 후진 / A: 좌회전 / D: 우회전 / 스페이스: 정지")

while True:
    key = get_key()
    if key == 'w':
        print("전진")
        ser.write(b'F')
    elif key == 's':
        print("후진")
        ser.write(b'B')
    elif key == 'a':
        print("좌회전")
        ser.write(b'L')
    elif key == 'd':
        print("우회전")
        ser.write(b'R')
    elif key == ' ':
        print("정지")
        ser.write(b'S')
    elif key == 'q':
        print("종료")
        ser.write(b'S')
        ser.close()
        break
