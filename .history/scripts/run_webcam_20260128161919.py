import cv2
from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")  # se descargará solo la primera vez

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No pude abrir la webcam. Prueba con índice 1 o 2.")

    print("YOLO OK. Presiona 'q' para salir.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, conf=0.25, verbose=False)
        annotated = results[0].plot()  # dibuja cajas

        cv2.imshow("yolo webcam", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
