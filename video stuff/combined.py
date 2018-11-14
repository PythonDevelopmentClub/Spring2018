import time

import cv2
import mss
import numpy


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)


last_boxes = []



if __name__ == '__main__':

    
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
    video_capture = cv2.VideoCapture(0)
    body_count = 0


    with mss.mss() as sct:
        # Part of the screen to capture

        monitor = sct.monitors[1]
        monitor = {"top": 40, "left": 0, "width": monitor["width"], "height": monitor["height"]}

        while "Screen capturing":
            last_time = time.time()

            # Get raw pixels from the screen, save it to a Numpy array
            img = numpy.array(sct.grab(monitor))
            frame = img
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            people, confidences = hog.detectMultiScale(gray, winStride=(8,8), padding=(16,16), scale=1.05)
            # Draw a rectangle around the faces
            for i in range(len(people)):
                (x, y, w, h) = people[i]
                confidence = confidences[i]
                if confidence > 0:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    body_img = frame[y:y+h, x:x+w]
                    # cv2.imwrite('body_'+str(body_count)+".jpg",body_img)
                    body_count += 1

            # Display the resulting frame
            imS = cv2.resize(frame, (960, 540)) 
            cv2.imshow('Video', imS)
            # Display the picture
            # cv2.imshow("OpenCV/Numpy normal", cv2.resize(img, (480, 270)))

            # Display the picture in grayscale
            # cv2.imshow('OpenCV/Numpy grayscale',
            #            cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))

            print("fps: {0}".format(1 / (time.time() - last_time)))

            # Press "q" to quit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
    while True:
        _,frame=cap.read()
        fgmask = fgbg.apply(frame)
        frame = cv2.bitwise_and(frame,frame,mask = fgmask)
        found,w=hog.detectMultiScale(frame, winStride=(8,8), padding=(32,32), scale=1.05)
        print found
        draw_detections(frame,found)
        cv2.imshow('feed',frame)
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break
    cv2.destroyAllWindows()