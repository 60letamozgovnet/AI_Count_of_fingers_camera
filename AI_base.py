import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model('modelzzz.h5')

background = None
hand = None
frames_elapsed = 0
FRAME_HEIGHT = 500
FRAME_WIDTH = 650
BG_WEIGHT = 0.5
stable_frames = 0
required_stable_frames = 20

region_top = 0
region_bottom = 300
region_left = 350
region_right = FRAME_WIDTH
capture = cv2.VideoCapture(0)
count = 0
cnt_of_rows = 0


class HandData:
    def __init__(self, top, bottom, left, right, centerX):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.centerX = centerX
        self.prevCenterX = 0
        self.isInFrame = False
        self.isWaving = False
        self.fingers = 0

    def update(self, top, bottom, left, right):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def check_for_waving(self, centerX):
        self.prevCenterX = self.centerX
        self.centerX = centerX
        if abs(self.centerX - self.prevCenterX) > 5:
            self.isWaving = True
        else:
            self.isWaving = False


def write_on_image(frame):
    cv2.rectangle(frame, (region_left, region_top), (region_right, region_bottom), (255, 255, 255), 2)


def get_region(frame):
    region = frame[region_top:region_bottom, region_left:region_right]
    gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    gray_region = cv2.GaussianBlur(gray_region, (7, 7), 0)

    thresholded = cv2.adaptiveThreshold(
        gray_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2
    )

    kernel = np.ones((3, 3), np.uint8)
    thresholded = cv2.erode(thresholded, kernel, iterations=1)

    return thresholded


def get_average(region):
    global background, frames_elapsed, hand
    if background is None:
        background = region.copy().astype("float")

    if hand is None:
        hand = HandData((0, 0), (0, 0), (0, 0), (0, 0), 0)

    if frames_elapsed % 30 == 0 and hand.isInFrame:
        cv2.accumulateWeighted(region, background, BG_WEIGHT)

    return background


def segment(region):
    global hand
    if hand is None:
        hand = HandData((0, 0), (0, 0), (0, 0), (0, 0), 0)

    diff = cv2.absdiff(background.astype(np.uint8), region)
    _, thresholded_region = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    thresholded_region = cv2.erode(thresholded_region, kernel, iterations=1)
    thresholded_region = cv2.dilate(thresholded_region, kernel, iterations=2)

    contours, _ = cv2.findContours(thresholded_region.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        hand.isInFrame = False
        return
    else:
        segmented_region = max(contours, key=cv2.contourArea)
        if cv2.contourArea(segmented_region) < 500:
            hand.isInFrame = False
            return
        else:
            hand.isInFrame = True
            return (thresholded_region, segmented_region)


def prepare_image_for_prediction(region):
    img_resized = cv2.resize(region, (150, 150))
    img_resized = img_resized / 255.0
    img_final = img_resized.reshape(1, 150, 150, 1)
    return img_final


def predict_fingers_from_model(region):
    img_processed = prepare_image_for_prediction(region)
    prediction = model.predict(img_processed)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    return predicted_class, confidence

while cnt_of_rows < 100:
    ret, frame = capture.read()
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    frame = cv2.flip(frame, 1)

    region = get_region(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('r'):
        frames_elapsed = 0
        stable_frames = 0

    if frames_elapsed < 30:
        background = get_average(region)
    else:
        region_pair = segment(region)
        if region_pair is not None:
            (thresholded_region, segmented_region) = region_pair
            cv2.drawContours(region, [segmented_region], -1, (255, 255, 255), 2)
            cv2.imshow("Segmented Image", thresholded_region)
            count += 1

            # Predict fingers using the model
            predicted_fingers, model_confidence = predict_fingers_from_model(thresholded_region)

            print(f"Predicted Fingers: {predicted_fingers}, Confidence: {model_confidence}")

            if model_confidence < 0.2:
                hand.fingers = 0
            else:
                hand.fingers = predicted_fingers

            if count == 70:
                cnt_of_rows += 1
                count = 0
                frames_elapsed = 0

    write_on_image(frame)
    cv2.imshow("Camera Input", frame)
    frames_elapsed += 1

capture.release()
cv2.destroyAllWindows()
