#!/usr/bin/env python3

''' A simple Program for grabbing video from Baumer camera and converting it to opencv stream.
'''

import sys
import cv2
import neoapi
import socket, cv2, pickle, struct, imutils

# get images, display and store stream (opencv_cap)
result = 0
try:
    camera = neoapi.Cam()
    camera.Connect()

    isColor = True
    if camera.f.PixelFormat.GetEnumValueList().IsReadable('BGR8'):
        camera.f.PixelFormat.SetString('BGR8')
    elif camera.f.PixelFormat.GetEnumValueList().IsReadable('Mono8'):
        camera.f.PixelFormat.SetString('Mono8')
        isColor = False
    else:
        print('no supported pixelformat')
        sys.exit(0)

    camera.f.ExposureTime.Set(20000)
    camera.f.AcquisitionFrameRateEnable.value = True
    camera.f.AcquisitionFrameRate.value = 60

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    # Define the fps to be equal to 10. Also frame size is passed.
    # video = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc(*'MJPG'), 10,
    # video = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10,
    # video = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc(*'XVID'), 10,
    #                         (camera.f.Width.value, camera.f.Height.value), isColor)

    # Socket Create
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    host_name = socket.gethostname()
    host_ip = socket.gethostbyname(host_name)
    print('HOST IP:', host_ip)
    port = 9999
    socket_address = ("0.0.0.0", port)

    # Socket Bind
    server_socket.bind(socket_address)

    # Socket Listen
    server_socket.listen(5)
    print("LISTENING AT:", socket_address)

    # Socket Accept
    while True:
        client_socket, addr = server_socket.accept()
        print('GOT CONNECTION FROM:', addr)
        try:
            if client_socket:
                message = b'bjr'
                while message != b'':
                    message = client_socket.recv(1024)
                    print("got message from client: "+str(message))
                    if message == b'send_image':
                        print('sending image')
                    else:
                        continue

                    frame = camera.GetImage().GetNPArray()
                    # rotate 90 degrees
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    frame = imutils.resize(frame, height=720)
                    a = pickle.dumps(frame)
                    message = struct.pack("Q", len(a)) + a
                    client_socket.sendall(message)

                    cv2.imshow('TRANSMITTING VIDEO', frame)
                    if cv2.waitKey(1) == '13':
                        client_socket.close()
        except Exception as e:
            print(e)
            continue


except (neoapi.NeoException, Exception) as exc:
    print('error: ', exc)
    result = 1

sys.exit(result)
