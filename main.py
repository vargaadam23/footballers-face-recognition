import cv2
import numpy as np
from helpers import takePicture, menu, resizeAndPad
from fisherfaces import prepare_training_data, predict

COUNTODOWN_TIME = 3

IMAGE_SIZE = (400,400)

FIRST_PICTURE = "FirstPicture"



def menuHandler():
    action = menu()

    if action == "1":
        takePicture(FIRST_PICTURE)
        return FIRST_PICTURE+".png"
    if action == "2":
        return input("Image path:")
    if action == "0":
        exit()

    return False


def main():
    print("Preparing data...")
    faces, labels, subjects = prepare_training_data("data")
    print("Data prepared")

    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))
    #Folosim modelul Fisher face
    face_recognizer = cv2.face.FisherFaceRecognizer_create()

    face_recognizer.train(faces, np.array(labels))

    print("Predicting images...")

    while(True):
        test_img = cv2.imread(menuHandler())

        predicted_img = predict(test_img, face_recognizer, subjects)
        print("Prediction complete")

        cv2.imshow("Subject", resizeAndPad(predicted_img, (400, 500), 127))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.destroyAllWindows()

    exit()

main()
