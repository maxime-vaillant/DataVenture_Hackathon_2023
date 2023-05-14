import base64
import socket, cv2, pickle, struct
import numpy as np
from typing import Generator

from vidgear.gears import NetGear


class RemoteCamera:
    """
    This class is used to connect to a remote camera.
    It exposes a generator that yields the frames.
    """
    def __init__(self, host_ip: str, port: int):
        """
        RemoteCamera constructor.
        :param host_ip: Server IP address
        :param port: Server port
        """
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.host_ip = host_ip
        self.port = port
        self.connected = False

    def connect(self):
        """
        Connects to the server.
        """
        self.client_socket.connect((self.host_ip, self.port))
        self.connected = True

    def disconnect(self):
        """
        Disconnects from the server.
        """
        self.client_socket.close()
        self.connected = False

    def get_frame(self) -> np.ndarray:
        """
        fetches a frame from the camera
        :return: Frame
        :return:
        """
        if not self.connected:
            raise Exception("Not connected to server. Please run RemoteCamera.connect() first.")

        data = b""
        payload_size = struct.calcsize("Q")
        self.client_socket.send(b"send_image")
        while len(data) < payload_size:
            packet = self.client_socket.recv(4 * 1024)  # 4K
            if not packet: break
            data += packet
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        while len(data) < msg_size:
            data += self.client_socket.recv(4 * 1024)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        return pickle.loads(frame_data)


class RemoteCamera2:
    """
    This class is used to connect to a remote camera.
    It exposes a generator that yields the frames.
    """
    def __init__(self, host_ip: str, port: int):
        """
        RemoteCamera constructor.
        :param host_ip: Server IP address
        :param port: Server port
        """
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.host_ip = host_ip
        self.port = port
        self.connected = False

    def connect(self):
        """
        Connects to the server.
        """
        self.client_socket.connect((self.host_ip, self.port))
        self.connected = True

    def disconnect(self):
        """
        Disconnects from the server.
        """
        self.client_socket.close()
        self.connected = False

    def get_frame(self) -> np.ndarray:
        """
        fetches a frame from the camera
        :return: Frame
        :return:
        """
        if not self.connected:
            raise Exception("Not connected to server. Please run RemoteCamera.connect() first.")

        data = b""
        payload_size = struct.calcsize("Q")
        self.client_socket.send(b"send_image")
        while len(data) < payload_size:
            packet = self.client_socket.recv(16 * 1024)  # 16K
            if not packet: break
            data += packet
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        while len(data) < msg_size:
            data += self.client_socket.recv(16 * 1024)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        unpickled = pickle.loads(frame_data)
        # use opencv to decode jpeg
        frame = cv2.imdecode(unpickled, cv2.IMREAD_COLOR)
        return frame


if __name__ == "__main__":
    rc = RemoteCamera("192.168.10.125", 9999)
    rc.connect()
    while True:
        try:
            input()
        except KeyboardInterrupt:
            break
        frame = next(rc.get_frame())

        cv2.imshow("received image", frame)
        if cv2.waitKey(1) == '13':
            break

    rc.disconnect()
    cv2.destroyAllWindows()

