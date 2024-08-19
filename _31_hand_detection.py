import cv2
import mediapipe as mp

video = cv2.VideoCapture(0)
h = mp.solutions.hands
hands = h.Hands()
mp_drawing = mp.solutions.drawing_utils

while True:
    succ, frame = video.read()

    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    print(result.multi_hand_landmarks)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks,
                                      mp.solutions.hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 255, 0)))


        for id, lm in enumerate(hand_landmarks.landmark): # Sahip olunan landmark sayısına kadar indexini alır
            #print(id, lm)
            h, w, c = frame.shape
            x, y = int(lm.x * w), int(lm.y * h)
            print(id, x, y)

            if id == 5:
                cv2.circle(frame, (x, y), 7, (0, 255, 0), -1)
            if id == 8:
                cv2.circle(frame, (x, y), 7, (0, 0, 255), -1)
    cv2.imshow('Video', frame)
    cv2.waitKey(1)

