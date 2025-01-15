import RPi.GPIO as GPIO
import time
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import cv2

# Servo motor setup
SERVO_PIN = 18  # GPIO 18 (Pin 12 on Raspberry Pi)
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)

# Set up PWM for the servo motor
pwm = GPIO.PWM(SERVO_PIN, 50)  # 50Hz frequency
pwm.start(0)  # Initialize PWM with 0 duty cycle

def set_servo_angle(angle):
    """Sets the servo to the specified angle."""
    duty = 2 + (angle / 18)  # Duty cycle calculation for the angle
    GPIO.output(SERVO_PIN, True)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.5)
    GPIO.output(SERVO_PIN, False)
    pwm.ChangeDutyCycle(0)

# Initialize 'currentname' to trigger only when a new person is identified
currentname = "unknown"
encodingsP = "encodings.pickle"

# Load the known faces and embeddings along with OpenCV's Haar cascade for face detection
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())

# Initialize the video stream and allow the camera sensor to warm up
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# Start the FPS counter
fps = FPS().start()

# State variables for door and timer
door_open = False
last_face_time = time.time()

# Loop over frames from the video file stream
try:
    while True:
        # Grab the frame from the threaded video stream
        frame = vs.read()
        frame = imutils.resize(frame, width=500)

        # Detect the face boxes
        boxes = face_recognition.face_locations(frame)
        encodings = face_recognition.face_encodings(frame, boxes)
        names = []

        face_detected = False

        # Loop over the facial embeddings
        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"  # Default name

            # Check if there's a match
            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # Count votes for each recognized face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # Determine the name with the highest votes
                name = max(counts, key=counts.get)

                if name == "Aditya" and not door_open:
                    print(f"[INFO] {name} detected. Opening the door.")
                    set_servo_angle(90)  # Open the door
                    door_open = True

            if name == "Aditya":
                face_detected = True

            names.append(name)

        # Draw bounding boxes and names on the frame
        for ((top, right, bottom, left), name) in zip(boxes, names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 225), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # If no face is detected, start the timer
        if not face_detected:
            if door_open and time.time() - last_face_time > 3:  # 3 seconds timeout
                print("[INFO] No face detected. Closing the door.")
                set_servo_angle(0)  # Close the door
                door_open = False
        else:
            last_face_time = time.time()  # Reset the timer if a face is detected

        # Display the frame
        cv2.imshow("Facial Recognition is Running", frame)
        key = cv2.waitKey(1) & 0xFF

        # Quit when 'q' key is pressed
        if key == ord("q"):
            break

        # Update the FPS counter
        fps.update()

except KeyboardInterrupt:
    print("\n[INFO] Exiting program.")

finally:
    # Clean up
    fps.stop()
    print("[INFO] Elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))
    cv2.destroyAllWindows()
    vs.stop()
    pwm.stop()
    GPIO.cleanup()
