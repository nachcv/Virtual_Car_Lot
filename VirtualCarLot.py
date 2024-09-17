
#Librerías que usaremos en el programa
import tkinter as tk
import json
import os
import numpy as np
import cv2
import face_recognition as fr
import queue
import threading
import speech_recognition as sr
from PIL import Image, ImageTk

#Vector Temporal y archivo donde guardaremos los datos en vectores.
known_users = []
data_file = "informacion.txt"

#Clase para el registro de usuarios
class RegistroUsuario(tk.Toplevel):
    #Interfaz para el registro de usuarios
    def __init__(self, parent, face_encoding):
        super().__init__(parent)
        self.parent = parent
        self.title("Registro de Nuevo Usuario")
        self.geometry("500x250")
        self.configure(bg="LightBlue")

        self.label_nombre = tk.Label(self, text="Nombre:")
        self.label_nombre.pack()
        self.entry_nombre = tk.Entry(self)
        self.entry_nombre.pack()

        self.label_pregunta1 = tk.Label(self, text="Preferencia de Procedencia(Alemania,Italia,Japón,Francia):")
        self.label_pregunta1.pack()
        self.entry_pregunta1 = tk.Entry(self)
        self.entry_pregunta1.pack()

        self.label_pregunta2 = tk.Label(self, text="Marca de Interés entre las disponibles(Mercedes,Audi,Toyota,Ferrari,Bugatti):")
        self.label_pregunta2.pack()
        self.entry_pregunta2 = tk.Entry(self)
        self.entry_pregunta2.pack()

        self.label_pregunta3 = tk.Label(self, text="Color Seleccionado(Rojo,Negro,Blanco,Gris):")
        self.label_pregunta3.pack()
        self.entry_pregunta3 = tk.Entry(self)
        self.entry_pregunta3.pack()

        self.btn_guardar = tk.Button(self, text="Guardar", command=lambda: self.guardar_usuario(face_encoding))
        self.btn_guardar.pack(pady=10)
    
    #Recoger los datos de la pestaña de registro.
    def guardar_usuario(self, face_encoding):
        nombre = self.entry_nombre.get()
        pregunta1 = self.entry_pregunta1.get()
        pregunta2 = self.entry_pregunta2.get()
        pregunta3 = self.entry_pregunta3.get()

        if nombre and pregunta1 and pregunta2 and pregunta3:
            # Guardar los datos en la lista de usuarios
            datos_usuario = (nombre, pregunta1, pregunta2, pregunta3, face_encoding)
            known_users.append(datos_usuario)

            # Actualizar la lista en el archivo
            save_known_faces(data_file)

            print(f"Nuevo usuario registrado: {nombre}")
            self.destroy()
        else:
            print("Todos los campos son obligatorios")

class App:
    #Pestaña de uso del usuario.
    def __init__(self, root):
        self.root = root
        self.procedencia = None
        self.marca = None
        self.color = None
        self.root.title("Virtual Car Lot")
        self.root.geometry("800x600")  

        load_known_faces(data_file)  # Cargar datos de los usuarios al iniciar

        self.sesion = 0
        self.current_face_encoding = None
        self.current_user = None
        self.stop_voice_thread = threading.Event()  #Parar la hebra de voz para que no genere bug
        self.voice_thread = None

        # Cargar la imagen del logo usando Pillow
        self.logo_image = Image.open("logo.jpg") 
        self.logo_image = self.logo_image.resize((600, 500), Image.LANCZOS)  # Redimensiona la imagen si es necesario
        self.logo_photo = ImageTk.PhotoImage(self.logo_image)
        self.root.configure(bg="Black")

        # Crear un label para la imagen del logo y colocarlo en la ventana
        self.logo_label = tk.Label(root, image=self.logo_photo)
        self.logo_label.pack(pady=10)
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        self.btn_inicio_sesion = tk.Button(button_frame, text="Iniciar Sesión", command=self.iniciar_sesion)
        self.btn_inicio_sesion.pack(side=tk.LEFT, padx=5)
        self.btn_catalogo = tk.Button(button_frame, text="Catalogo", command=self.procesar_aruco, state=tk.DISABLED)
        self.btn_catalogo.pack(side=tk.LEFT, padx=5)
        self.btn_cerrar_sesion = tk.Button(button_frame, text="Cerrar Sesión", command=self.cerrar_sesion, state=tk.DISABLED)
        self.btn_cerrar_sesion.pack(side=tk.LEFT, padx=5)
        btn_cerrar = tk.Button(button_frame, text="Cerrar Interfaz", command=self.cerrar_interfaz)
        btn_cerrar.pack(side=tk.LEFT, padx=5)

        self.actualizar_boton_catalogo()

    def cerrar_interfaz(self):
        self.stop_voice_thread.set()
        self.root.destroy()
    #Hacer que el botón de catálogo solo pueda ser usado por un usuario registrado.
    def actualizar_boton_catalogo(self):
        if self.sesion > 0:
            self.btn_catalogo.config(state=tk.NORMAL)
            self.btn_inicio_sesion.config(text="Cerrar Sesión", command=self.cerrar_sesion)
            self.btn_cerrar_sesion.config(state=tk.NORMAL)
        else:
            self.btn_catalogo.config(state=tk.DISABLED)
            self.btn_inicio_sesion.config(text="Iniciar Sesión", command=self.iniciar_sesion)
            self.btn_cerrar_sesion.config(state=tk.DISABLED)
    #Inicio de sesión
    def iniciar_sesion(self):
        camara = cv2.VideoCapture(0)
        face_detected, face_encoding = self.Reconocimiento(camara)
        camara.release()
        if face_detected:
            user_name, user_info = self.reconocer_usuario(face_encoding)
            if user_name:
                _, procedencia, marca, color, _ = user_info
                self.procedencia = procedencia
                self.marca = marca
                self.color = color

                self.current_user = user_info
                print(f"Bienvenido de nuevo, {user_name}!")
                self.sesion = 1
                self.actualizar_boton_catalogo()
            else:
                RegistroUsuario(self.root, face_encoding) 
        else:
            print("No se detectó ninguna cara")
    #Reconocimiento facial del usuario.
    def Reconocimiento(self, camara):
        face_detected = False
        face_encoding = None
        if camara.isOpened():
            while True:
                ret, frame = camara.read()
                if not ret:
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = fr.face_locations(rgb_frame)
                if face_locations:
                    face_detected = True
                    face_encoding = fr.face_encodings(rgb_frame, face_locations)[0]
                cv2.imshow("Reconocimiento", frame)
                if face_detected:
                    break
                if cv2.waitKey(1) == 27:
                    break
        cv2.destroyAllWindows()
        return face_detected, face_encoding
    #Uso del reconocimiento facial para entrar con uno de los usuarios registrados.
    def reconocer_usuario(self, face_encoding):
        for user_info in known_users:
            encoding = user_info[-1]  # El último elemento del vector es el encoding facial
            if fr.compare_faces([encoding], face_encoding)[0]:
                return user_info[0], user_info

        return None, None

    def cerrar_sesion(self):
        self.sesion = 0
        self.current_user = None
        self.actualizar_boton_catalogo()
        print("Sesión cerrada")
    #Reconocimiento de voz usado dentro del programa.
    def reconocimiento_voz(self, q):
        r = sr.Recognizer()
        mic = sr.Microphone()
        while not self.stop_voice_thread.is_set():
            try:
                with mic as source:
                    print("¿Qué quieres ver?")
                    r.adjust_for_ambient_noise(source)
                    audio = r.listen(source, timeout=5)
                texto = r.recognize_google(audio, language='es-ES')
                print(f"Usted dijo: {texto}")
                q.put(texto)
            except sr.UnknownValueError:
                print("No se entiende la orden.")
                q.put(None)
    #Función para la interacción con el aruco.
    def procesar_aruco(self):
        self.stop_voice_thread.clear()  
        mensaje = ""
        if os.path.exists('camara.py'):
            import camara
        else:
            print("Es necesario realizar la calibración de la cámara.")
            exit()
        coches = [cv2.imread(f"Coches/Coche{i}.jpg") for i in range(1, 8)]
        fichas = [cv2.imread(f"Fichas/Ficha{i}.jpg") for i in range(1, 7)]
        DIC = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
        parametros = cv2.aruco.DetectorParameters()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("No se pudo acceder a la cámara.")
            return
        hframe = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        wframe = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print("Tamaño del frame de la cámara: ", wframe, "x", hframe)
        matrix, roi = cv2.getOptimalNewCameraMatrix(camara.cameraMatrix, camara.distCoeffs, (wframe, hframe), 1, (wframe, hframe))
        roi_x, roi_y, roi_w, roi_h = roi
        q = queue.Queue()
        self.voice_thread = threading.Thread(target=self.reconocimiento_voz, args=(q,))
        self.voice_thread.daemon = True
        self.voice_thread.start()
        coche_actual = coches[6]
        while not self.stop_voice_thread.is_set():
            ret, framebgr = cap.read()
            if not ret:
                print("Error al capturar el frame de la cámara.")
                break
            framerectificado = cv2.undistort(framebgr, camara.cameraMatrix, camara.distCoeffs, None, matrix)
            framerecortado = framerectificado[roi_y:roi_y + roi_h, roi_x: roi_x + roi_w].copy()
            corners, ids, rejected = cv2.aruco.detectMarkers(framerecortado, DIC, parameters=parametros)
            if len(corners) > 0:
                for i in range(len(corners)):
                    cv2.polylines(framerecortado, [corners[i].astype(int)], True, (230, 204, 255), 4)
                    pts1 = np.float32(corners[i][0])
                    width = int(max(np.linalg.norm(corners[i][0][0] - corners[i][0][1]),
                                    np.linalg.norm(corners[i][0][2] - corners[i][0][3])))
                    height = int(max(np.linalg.norm(corners[i][0][0] - corners[i][0][3]),
                                     np.linalg.norm(corners[i][0][1] - corners[i][0][2])))
                    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
                    try:
                        mensaje = q.get_nowait()
                    except queue.Empty:
                        mensaje = None
                    if mensaje == "muéstrame Bugatti":
                        coche_actual = coches[0]
                    elif mensaje == "muéstrame Ferrari":
                        coche_actual = coches[1]
                    elif mensaje == "muéstrame Mercedes":
                        coche_actual = coches[2]
                    elif mensaje == "muéstrame Toyota":
                        coche_actual = coches[3]
                    elif mensaje == "muéstrame Audi":
                        coche_actual = coches[4]
                    elif mensaje == "muéstrame Urus":
                        coche_actual = coches[5]
                    elif mensaje == "muéstrame ficha Bugatti":
                        coche_actual = fichas[0]
                    elif mensaje == "muéstrame ficha Ferrari":
                        coche_actual = fichas[1]
                    elif mensaje == "muéstrame ficha Mercedes":
                        coche_actual = fichas[2]
                    elif mensaje == "muéstrame ficha Toyota":
                        coche_actual = fichas[3]
                    elif mensaje == "muéstrame ficha Audi":
                        coche_actual = fichas[4]
                    elif mensaje == "muéstrame ficha Urus":
                        coche_actual = fichas[5]
                    elif mensaje == "muéstrame preferencia":
                        if self.procedencia == "Francia" and self.color == "Negro":
                            coche_actual = coches[0]
                        elif self.procedencia == "Italia" and self.color == "Rojo":
                            coche_actual = coches[1]
                        elif self.procedencia == "Alemania" and self.color == "Negro":
                            coche_actual = coches[2]
                        elif self.procedencia == "Alemania" and self.color == "Blanco":
                            coche_actual = coches[3]
                        elif self.procedencia == "Japón" and self.color == "Gris":
                            coche_actual = coches[4]
                        elif self.procedencia == "Italia" and self.color == "Gris":
                            coche_actual = coches[5]
                    elif mensaje == "salir":
                        self.stop_voice_thread.set()
                        break
                    elif mensaje == "muéstrame más caro":
                        coche_actual = coches[0]
                    #Creación del muro blanco de exposición y ajuste de las imágenes.
                    muro_blanco = np.ones((height, width, 3), dtype=np.uint8) * 255
                    original_height, original_width = coche_actual.shape[:2]
                    aspect_ratio = original_width / original_height
                    if width / aspect_ratio < height:
                        new_width = width
                        new_height = int(width / aspect_ratio)
                    else:
                        new_height = height
                        new_width = int(height * aspect_ratio)
                    coche_redimensionado = cv2.resize(coche_actual, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    x_offset = (width - new_width) // 2
                    y_offset = (height - new_height) // 2
                    muro_blanco[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = coche_redimensionado
                    M = cv2.getPerspectiveTransform(pts2, pts1)
                    dst = cv2.warpPerspective(muro_blanco, M, (roi_w, roi_h))
                    mask = cv2.warpPerspective(np.ones_like(muro_blanco) * 255, M, (roi_w, roi_h))
                    mask_inv = cv2.bitwise_not(mask)
                    img1_bg = cv2.bitwise_and(framerecortado, framerecortado, mask=cv2.cvtColor(mask_inv, cv2.COLOR_BGR2GRAY))
                    img2_fg = cv2.bitwise_and(dst, dst, mask=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))
                    framerecortado = cv2.add(img1_bg, img2_fg)
            cv2.imshow("Reconocimiento de Coches", framerecortado)
            key = cv2.waitKey(10)
            if key == ord(' ') or (mensaje and mensaje.lower() == "salir"):
                self.stop_voice_thread.set()
                break
            elif key == 27:
                self.stop_voice_thread.set()
                break
        cap.release()
        cv2.destroyAllWindows()
#Guardar datos de los usuarios.
def save_known_faces(file_path):
    data = {
        "users": [
            {
                "name": user[0],
                "Procedencia": user[1],
                "Marca": user[2],
                "Color": user[3],
                "encoding": user[4].tolist() if user[4] is not None else None
            }
            for user in known_users
        ]
    }
    with open(file_path, 'w') as f:
        json.dump(data, f)
#Cargar los datos de los usuarios.
def load_known_faces(file_path):
    global known_users
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            known_users = [
                (
                    user["name"],
                    user["Procedencia"],
                    user["Marca"],
                    user["Color"],
                    np.array(user["encoding"]) if user["encoding"] is not None else None
                )
                for user in data["users"]
            ]
    except FileNotFoundError:
        known_users = []

def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
