import io
import socket
import struct
from PIL import Image
import cv2
import numpy as np
import datetime


def is_hour_changed(previous_time, current_time):
    return previous_time.hour != current_time.hour


def save_image(time, frame):
    imagefile_name = './image/' + time.strftime('%Y%m%d%H%M%S') + '.png'
    cv2.imwrite(imagefile_name, frame)
    print("Saved: {}".format(imagefile_name))


if __name__ == '__main__':
    # Start a socket listening fo   r connections on 0.0.0.0:8000 (0.0.0.0 means
    # all interfaces)
    server_socket = socket.socket()
    server_socket.bind(('0.0.0.0', 8866))
    server_socket.listen(0)

    # Accept a single connection and make a file-like object out of it
    connection = server_socket.accept()[0].makefile('rb')

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
            #cv2.imshow('image', opencv_image)

            current_time = datetime.datetime.now()
            if is_hour_changed(previous_time, current_time):
                save_image(current_time, opencv_image)

#            if writer:
#                writer.write(opencvImage)
#            key = cv2.waitKey(1)
#            if key & 0xFF == ord('q'):
#                break
#            elif key & 0xFF == ord('r'):
#                print('r is pressed')
#                if writer:
#                    # write frame
#                    end_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
#                    writer.release()
#                    print("video file: {} was saved.".format(video_filename))
#                    
#                else:
#                    start_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
#                    video_filename = "{}.avi".format(start_time)
#                    writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 1,  (640, 480))
#
            previous_time = current_time


    finally:
        connection.close()
        server_socket.close()
        if writer:
            writer.release()

