import cv2
import datetime
import glob
import io
import os
import numpy as np
import socket
import struct
import subprocess
import time

from PIL import Image

import detection_models




THIS_DIR = os.path.dirname(os.path.realpath(__file__))

def is_hour_changed(previous_time, current_time):
    return previous_time.hour != current_time.hour


def is_day_changed(previous_time, current_time):
    return previous_time.day != current_time.day


def is_next_10_minutes(previous_time, current_time):
    return previous_time.minute // 10 != current_time.minute // 10


def save_image(time, frame):
    imagefile_name = './image/' + time.strftime('%Y%m%d%H%M%S') + '.png'
    cv2.imwrite(imagefile_name, frame)
    print("Saved: {}".format(imagefile_name))


def create_gif(date):
    # Get file list
    image_dir = THIS_DIR + '/image'
    gif_filename = image_dir + '/' + date + ".gif"
    if os.path.exists(gif_filename):
        print("Already exists: {}".format(gif_filename))
        return True

    files = glob.glob(image_dir + '/' + date + '*.png')
    command = ["convert", "-delay", "50", "-loop", "0"]
    command.extend(sorted(files))
    command.append(gif_filename)
    return_code = subprocess.run(command).returncode
    if return_code == 0:
        print("Saved: {}".format(gif_filename))
        for f in files:
            os.remove(f)
        print("Removed original png files for : {}".format(gif_filename))
        return True
    else:
        print("Failed to save: {}".format(gif_filename))
        return False


if __name__ == '__main__':
    show_image = False
    # Create SSD object detection model
    ssd_model = detection_models.SSDModel()

    while True:
        server_socket = socket.socket()
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        server_socket.bind(('0.0.0.0', 8866))
        server_socket.listen(0)
        connected = False    
        while not connected:
            try:
                connection = server_socket.accept()[0].makefile('rb')
                connected = True
                print("Connected to a client")
            except Exception as e:
                time.sleep(0.1)
                pass

        # Accept a single connection and make a file-like object out of it
        previous_time = datetime.datetime.now()

        writer = None
        try:
            while True:
                # Read the length of the image as a 32-bit unsigned int. If the
                # length is zero, quit the loop
                image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
                if not image_len:
                    break
                # Construct a stream to hold the image data and read the image
                # data from the connection
                image_stream = io.BytesIO()
                image_stream.write(connection.read(image_len))
                # Rewind the stream, open it as an image with PIL and do some
                # processing on it
                image_stream.seek(0)
                image = Image.open(image_stream)
                #print('Image is %dx%d' % image.size)
                opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                current_time = datetime.datetime.now()
                if is_hour_changed(previous_time, current_time):
                    save_image(current_time, opencv_image)
                elif is_day_changed(previous_time, current_time):
                    date = previous_time.strftime("%Y%m%d")
                    create_gif(date)

                detections = ssd_model.detect(opencv_image)
                opencv_image = ssd_model.annotate(detections, opencv_image)
  
                previous_time = current_time
        except:
            pass

        finally:
            connection.close()
            server_socket.close()
            cv2.destroyAllWindows() 
            if writer:
                writer.release()

