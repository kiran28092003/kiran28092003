#! /usr/bin/python

# Import necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2
import RPi.GPIO as GPIO  # For controlling the servo motor

# Initialize GPIO for servo motor
SERVO_PIN = 18  # GPIO pin connected to the servo
GPIO.setmode(GPIO.BCM)  # Use BCM GPIO numbering
GPIO.setup(SERVO_PIN, GPIO.OUT)

# Create PWM instance
pwm = GPIO.PWM(SERVO_PIN, 50)  # 50Hz frequency
pwm.start(0)

def set_servo_angle(angle):
    """
    Set the servo motor to a specific angle.
    :param angle: Desired angle (0 to 180 degrees)
    """
    duty = 2 + (angle / 18)  # Convert angle to duty cycle
    GPIO.output(SERVO_PIN, True)
    pwm.ChangeDutyCycle(duty)
    time.sleep(1)
    GPIO.output(SERVO_PIN, False)
    pwm.ChangeDutyCycle(0)

# Initialize 'currentname' to trigger only when a new person is identified.
currentname = "unknown"
encodingsP = "encodings.pickle"

# Load the known faces and embeddings
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())

# Initialize the video stream
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# Start the FPS counter
fps = FPS().start()

try:
    # Loop over frames from the video file stream
    while True:
        # Grab the frame and resize it
        frame = vs.read()
        frame = imutils.resize(frame, width=500)

        # Detect the face boxes
        boxes = face_recognition.face_locations(frame)
        encodings = face_recognition.face_encodings(frame, boxes)
        names = []

        # Loop over the facial embeddings
        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"

            # Check for matches
            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                name = max(counts, key=counts.get)

                if currentname != name:
                    currentname = name
                    print(f"Recognized: {currentname}")
                    
                    # If the recognized name is "Aditya," turn the servo motor
                    if currentname == "Aditya":
                        print("Turning servo motor to 90 degrees...")
                        set_servo_angle(90)

            names.append(name)

        # Loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 225), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)

        # Display the image
        cv2.imshow("Facial Recognition is Running", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        fps.update()

finally:
    # Cleanup GPIO and video resources
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    cv2.destroyAllWindows()
    vs.stop()
    pwm.stop()
    GPIO.cleanup()
