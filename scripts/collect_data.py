import cv2
import os

GESTURES = ['palm', 'fist', 'thumbs_up', 'thumbs_down', 'okay', 'peace']
DATASET_PATH = '../dataset'

def create_dirs():
    for gesture in GESTURES:
        os.makedirs(os.path.join(DATASET_PATH, gesture), exist_ok=True)

def main():
    create_dirs()
    cap = cv2.VideoCapture(0)
    print("Press the number key (1-{}) for each gesture, 'q' to quit.".format(len(GESTURES)))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Collect Gestures', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        for idx, gesture in enumerate(GESTURES):
            if key == ord(str(idx+1)):
                img_name = os.path.join(DATASET_PATH, gesture, f"{gesture}_{len(os.listdir(os.path.join(DATASET_PATH, gesture)))}.jpg")
                cv2.imwrite(img_name, frame)
                print(f"Saved {img_name}")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
