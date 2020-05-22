#!/usr/bin/python3

import io
import socket
import struct
import time
import picamera

while True:
    client_socket = socket.socket()
    connected = False
    
    while not connected:
        try:
            client_socket.connect(('192.168.178.14', 8866))
            connected = True
            print("Connected to the server")
        except Exception as e:
            time.sleep(0.1)
            pass
        
    # Make a file-like object out of the connection
    connection = client_socket.makefile('wb')
    try:
        camera = picamera.PiCamera()
        camera.resolution = (640, 480)
        # Start a preview and let the camera warm up for 2 seconds
        camera.start_preview()
        time.sleep(2)
    
        # Note the start time and construct a stream to hold image data
        # temporarily (we could write it directly to connection but in this
        # case we want to find out the size of each capture first to keep
        # our protocol simple)
        start = time.time()
        stream = io.BytesIO()
        for foo in camera.capture_continuous(stream, 'jpeg'):
            # Write the length of the capture to the stream and flush to
            # ensure it actually gets sent
            connection.write(struct.pack('<L', stream.tell()))
            connection.flush()
            # Rewind the stream and send the image data over the wire
            stream.seek(0)
            connection.write(stream.read())
            # If we've been capturing for more than 30 seconds, quit
    #        if time.time() - start > 30:
    #            break
            # Reset the stream for the next capture
            stream.seek(0)
            stream.truncate()
        # Write a length of zero to the stream to signal we're done
        connection.write(struct.pack('<L', 0))
    
    except (BrokenPipeError, IOError):
        print("Detected remote disconnect")
    except:
        connection.close()
    finally:
        client_socket.close()
        camera.close()
