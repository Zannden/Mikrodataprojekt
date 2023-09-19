# import the opencv library
import cv2
import numpy as np
from tensorflow import keras
from keras.models import load_model


# image
# img = cv2.imread("photo.jpg")
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# define a video capture object
vid = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
WIDTH = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
HEIGHT = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Recognition models
# Nmber recognition
model = load_model("digits.h5")


def prediction(image, model):
    img = cv2.resize(image, (28, 28))
    img = img / 255
    img = img.reshape(1, 28, 28, 1)
    predict = model.predict(img, verbose=None)
    prob = np.amax(predict)
    class_index = np.argmax(model.predict(img, verbose=None), axis=1)
    result = class_index[0]
    if prob < 0.75:
        result = 0
        prob = 0
    return result, prob


while True:
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    frame_copy = frame.copy()
    frame_copy = cv2.rotate(frame_copy, cv2.ROTATE_180)
    frame_copy = cv2.flip(frame_copy, 0)
    frame_gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)

    # Set box size
    bbox_size = (60, 60)
    # Set box top left corner X and Y positions
    bbox = [
        (int(WIDTH // 2 - bbox_size[0] // 2), int(HEIGHT // 2 - bbox_size[1] // 2)),
        (int(WIDTH // 2 + bbox_size[0] // 2), int(HEIGHT // 2 + bbox_size[1] // 2)),
    ]

    img_cropped = frame[bbox[0][1] : bbox[1][1], bbox[0][0] : bbox[1][0]]
    img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (200, 200))
    cv2.imshow("guessbox", img_gray)

    result, probability = prediction(img_gray, model)
    cv2.putText(
        frame_copy,
        f"Prediction: {result}",
        (40, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame_copy,
        "Probability:" + "{:.2f}".format(probability),
        (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )

    # Draw a rectangle
    cv2.rectangle(frame_copy, bbox[0], bbox[1], (0, 255, 0), 3)

    # Display the resulting frame
    cv2.imshow("frame", frame_copy)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
