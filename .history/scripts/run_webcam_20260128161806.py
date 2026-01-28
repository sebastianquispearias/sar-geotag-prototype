import cv2

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No pude abrir la webcam (index 0). Prueba cambiar webcam_index en configs o usar otro índice.")

    print("Webcam OK. Presiona 'q' para salir.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("No pude leer frame.")
            break

        cv2.imshow("sar-geotag-prototype | webcam test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
