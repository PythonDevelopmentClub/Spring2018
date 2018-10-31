import time

import cv2
import mss
import numpy


with mss.mss() as sct:
    # Part of the screen to capture

    monitor = sct.monitors[1]
    monitor = {"top": 40, "left": 0, "width": monitor["width"], "height": monitor["height"]}

    while "Screen capturing":
        last_time = time.time()

        # Get raw pixels from the screen, save it to a Numpy array
        img = numpy.array(sct.grab(monitor))

        # Display the picture
        cv2.imshow("OpenCV/Numpy normal", cv2.resize(img, (480, 270)))

        # Display the picture in grayscale
        # cv2.imshow('OpenCV/Numpy grayscale',
        #            cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))

        print("fps: {0}".format(1 / (time.time() - last_time)))

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break