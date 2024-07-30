import cv2
import mediapipe as mp
import numpy as np

# Step 1: Import Libraries
import cv2
import mediapipe as mp
import numpy as np

# Step 2: Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Step 3: Initialize Video Capture
cap = cv2.VideoCapture(0)

# Variables to store drawing points
drawing_points = []
drawing = False
initial_point = None

# Function to recognize shapes
def recognize_shape(contour):
    shape = "Unidentified"
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    if len(approx) == 3:
        shape = "Triangle"
    elif len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "Square" if ar >= 0.95 and ar <= 1.05 else "Rectangle"
    elif len(approx) > 4:
        shape = "Circle"
    return shape

# Step 4: Capture and Process Each Frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Initialize list to store landmark coordinates
            landmark_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                # Get the coordinates
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_list.append([cx, cy])

            # Check if the index finger is up (not curled)
            if landmark_list[8][1] < landmark_list[7][1] < landmark_list[6][1]:
                # Set initial point and start drawing
                if not drawing:
                    initial_point = (landmark_list[8][0], landmark_list[8][1])
                    drawing = True
                # Add the coordinates of the index finger tip to the drawing points
                drawing_points.append((landmark_list[8][0], landmark_list[8][1]))
            else:
                # Stop drawing when finger is down
                drawing = False
                initial_point = None
                drawing_points = []

    # Create a blank image to draw shapes
    drawing_frame = np.zeros_like(frame)

    # Draw the points on the frame
    if drawing_points:
        cv2.polylines(drawing_frame, [np.array(drawing_points)], False, (255, 255, 255), 2)

    # Convert the drawing frame to grayscale and find contours
    gray = cv2.cvtColor(drawing_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Recognize shapes
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Adjust the area threshold as needed
            shape = recognize_shape(contour)
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(frame, shape, (cX - 50, cY - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Step 5: Display the Frame
    cv2.imshow('Hand Gesture Drawing and Shape Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 6: Release Resources
cap.release()
cv2.destroyAllWindows()
