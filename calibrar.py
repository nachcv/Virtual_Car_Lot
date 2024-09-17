import cv2
import numpy as np
import time
#Función sacada del apartado de prado de la asignatura donde por medio de un Charuco de 6x8 
#calibraremos la camara y lo guardaremos las medidas obtenidas  en camara.py que será posteriormente
#usado para representar las imágenes sobre los arucos.

def calibrar_camara():
    # Definición del diccionario de marcadores y el tablero Charuco
    DICCIONARIO = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    tablero = cv2.aruco.CharucoBoard((6, 8), 0.03, 0.02, DICCIONARIO)
    tablero.setLegacyPattern(True)  # Esto es para evitar problemas con tableros antenriores.

    # Inicialización del detector  para el Charuco
    detector = cv2.aruco.CharucoDetector(tablero)

    # creamos y abrimos la camara para calibrar
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    # Ajuste de los parámetros de configuración
    wframe = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    hframe = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #Variables para esquinas y los marcadores detectados en la calibración
    esquinas = []
    marcadores = []

    # Configuración del tiempo de captura y de las variables que usaremos
    CPS = 1
    tiempo = 1.0 / CPS
    final = False
    n = 0
    antes = time.time()

    # Bucle de captura y detección del tablero Charuco, hasta que se presione una tecla para finalizarlo.
    while not final:
        ret, frame = cap.read()
        if not ret:
            final = True
        else:
            if time.time() - antes > tiempo:
                bboxs, ids, _, _ = detector.detectBoard(frame)
                if ids is not None and ids.size > 8:
                    antes = time.time()
                    cv2.aruco.drawDetectedCornersCharuco(frame, bboxs, ids)
                    esquinas.append(bboxs)
                    marcadores.append(ids)
                    n += 1
            cv2.putText(frame, str(n), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            cv2.imshow("WEBCAM", frame)
            if cv2.waitKey(20) > 0:
                final = True

    # Liberación de la captura y cierre de ventanas para que no haya problemas posteriormente con otros usos de la cámara.
    cap.release()
    cv2.destroyAllWindows()

    # Procesamiento de resultados de calibración si se capturaron imágenes
    if n == 0:
        print("No se han capturado imágenes para hacer la calibración")
    else:
        print("Espera mientras calculo los resultados de calibración de la cámara...")

        cameraMatrixInt = np.array([[1000, 0, hframe / 2],
                                    [0, 1000, wframe / 2],
                                    [0, 0, 1]])
        distCoeffsInt = np.zeros((5, 1))
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
        criteria = (cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9)

        ret, cameraMatrix, distCoeffs, rvec, tvec, stdInt, stdExt, errores = cv2.aruco.calibrateCameraCharucoExtended(
            charucoCorners=esquinas,
            charucoIds=marcadores,
            board=tablero,
            imageSize=(hframe, wframe),
            cameraMatrix=cameraMatrixInt,
            distCoeffs=distCoeffsInt,
            flags=flags,
            criteria=criteria
        )

        # Creación (si no existe) del archivo camara.py donde se almacenan los datos de calibrado
        with open('camara.py', 'w') as fichero:
            fichero.write("import numpy as np\n")
            fichero.write("cameraMatrix = np.")
            fichero.write(repr(cameraMatrix))
            fichero.write("\ndistCoeffs = np.")
            fichero.write(repr(distCoeffs))
            fichero.close()
            print("Los resultados de calibración se han guardado en el fichero camara.py")

if __name__ == "__main__":
    calibrar_camara()
