## Reconocimiento de Expresiones Faciales (Emociones) CNN

Esta aplicacion reconoce expresiones faciales (emociones) en videos o desde la camara de la computadora a traves de una red neurona convolucional. Estas expresiones faciales son: angry, disgust, fear, happy, neutral, sad, surprise.

Para la creacion, entrenamiento y evaluacion de la red neuronal convoluciona se uso tensorflow, numpy y matplotlib.

Para el reconocimiento en tiempo real del video se uso tensorflow, numpy, opencv y dlib.

Para ejecutar el consola es necesario usar la bandera "-video" si se quiere usar un video guardado en la pc.

Ejemplo:

`python Reconocimiento_Emociones_Realtime.py -video video.mp4`

Aqui "-video" es la bandera para elegir un video guardado y "video.mp4" es el archivo de video que se analizara.

Si no se usa ninguna bandera por defecto abrira la camara de la computadora.

Ejemplo:

`python Reconocimiento_Emociones_Realtime.py`
