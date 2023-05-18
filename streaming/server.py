import pickle
import socket
import struct
from threading import Thread

import cv2
import imutils
import neoapi

from config import SERVER_PORT

image = None
running = True


def stream_video_thread():
    global image
    # exposure_time = 20000
    print('Starting stream video thread')

    try:
        camera = neoapi.Cam()
        camera.Connect()

        if camera.f.PixelFormat.GetEnumValueList().IsReadable('BGR8'):
            camera.f.PixelFormat.SetString('BGR8')
        elif camera.f.PixelFormat.GetEnumValueList().IsReadable('Mono8'):
            camera.f.PixelFormat.SetString('Mono8')
        else:
            print('No Supported PixelFormat')

        # camera.f.ExposureTime.Set(exposure_time)
        camera.f.ExposureAuto.Set('Continuous')
        camera.f.AcquisitionFrameRateEnable.value = True
        camera.f.AcquisitionFrameRate.value = 60

        while running:
            frame = camera.GetImage().GetNPArray()
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = imutils.resize(frame, height=720)
            image = frame

    except Exception as err:
        print(err)


def handle_client_thread(client_socket):
    global image

    message = b'up_to_date'

    while message != b'' and running:
        message = client_socket.recv(1024)

        if message == b'send_image' and image is not None:
            pckl = pickle.dumps(image)

            message = struct.pack("Q", len(pckl)) + pckl
            client_socket.sendall(message)

        if cv2.waitKey(1) == '13':
            client_socket.close()

    client_socket.close()


def socket_server_thread():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    host_name = socket.gethostname()
    host_ip = socket.gethostbyname(host_name)

    print('HOST IP:', host_ip)

    port = SERVER_PORT
    socket_address = ("0.0.0.0", port)

    # Socket Bind
    server_socket.bind(socket_address)

    # Socket Listen
    server_socket.listen(5)
    print("LISTENING AT:", socket_address)

    while True:
        client_socket, addr = server_socket.accept()
        print('GOT CONNECTION FROM:', addr)

        client_thread = Thread(target=handle_client_thread, args=[client_socket])
        client_thread.start()


if __name__ == '__main__':
    stream_thread = Thread(target=stream_video_thread, args=[])
    stream_thread.start()

    try:
        socket_server_thread()
    except KeyboardInterrupt:
        running = False
