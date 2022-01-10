import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import cv2
import dlib
import pickle
import argparse

# Contruir el modelo
def construir_modelo():
    backbone = tf.keras.applications.EfficientNetB2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights=None
    )
    model = tf.keras.Sequential([
        backbone,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    return model

model = construir_modelo()
# Cargar los pesos guardados
model.load_weights("models/best_weights.h5")


# Cargar LabelEncoder 
def cargar_obj(name):
    pickle_obj = open(f"{name}.pck","rb")
    obj = pickle.load(pickle_obj)
    return obj

Le = cargar_obj("models/LabelEncoder")



def procesar_imagen(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, [96, 96], method="bilinear")
    image = tf.expand_dims(image, 0)
    return image

def prediccion_tiempo_real(image, model, encoder_):
    prediction = model.predict(image)
    prediction = np.argmax(prediction, axis = 1)
    return encoder_.inverse_transform(prediction)[0]

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-video', help='Ruta del video')
    args = parser.parse_args()

    if args.video:
        # ruta del video
        option = args.video
    else:
        # camara de la computadora
        option = 0


    VideoCapture = cv2.VideoCapture(option)

    detector = dlib.get_frontal_face_detector()

    while True:
        ret, frame = VideoCapture.read() 
        
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        if len(rects) >= 1:
            for rect in rects:
                (x, y, w, h) = rect_to_bb(rect)
                img = gray[y-10 : y+h+10 , x-10 : x+w+10]
                
                if img.shape[0] == 0 or img.shape[1] == 0:
                    cv2.imshow("Frame", frame)
                    
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    img = procesar_imagen(img)
                    out = prediccion_tiempo_real(img, model, Le)
                    cv2.rectangle(frame, (x, y), (x+w, y+h),(0, 255, 0), 2)
                    z = y-15 if y-15 > 15 else y+15
                    cv2.putText(frame, str(out), (x, z), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
                    
            cv2.imshow("Frame", frame)
                
        else:
            cv2.imshow("Frame", frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    VideoCapture.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()


