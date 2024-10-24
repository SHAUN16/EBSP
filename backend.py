from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import cv2
import numpy as np

import base64
import io
import tensorflow as tf
import pandas as pd
import time
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import dotenv 

app = FastAPI()

# CORS setup to allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Update with your frontend address
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained model (e.g., TensorFlow/Keras model)
model = tf.keras.models.load_model('./model_new_training_optimal.h5')

def emotion_recog(frame):
    model.load_weights('./model_weights_training_optimal.weights.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1:"Sad", 2:"Happy", 3: "Calm"}

    # frame = cv2.imread("image1.jpg")
    # facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # for jupyter
    facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # for colab
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 255), 3)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        return frame, None  # No face detected
    #cv2_imshow(frame)
    return frame, emotion_dict[maxindex]


def record_video(seconds):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    frame_height, frame_width, _ = frame.shape
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
    
    # Capture video for a specific duration (e.g., 5 seconds)
    capture_duration = seconds  # in seconds
    start_time = cv2.getTickCount() / cv2.getTickFrequency()
    
    print("Processing Video...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            out.release()
            break
        output, output_emotion = emotion_recog(frame)
        out.write(output)
        
        elapsed_time = (cv2.getTickCount() / cv2.getTickFrequency()) - start_time
        if elapsed_time >= capture_duration:
            break
        # Press 'q' to exit the loop early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    out.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Done processing video for {capture_duration} seconds")
    return output_emotion

def get_songs_list(mood):
    # read the preprocessed songs file
    df = pd.read_csv("./Updated_songs.csv")
    
    if not mood:
        shuffled_songs = df.sample(frac=1)
        return shuffled_songs
    
    else:
        # filter the dataframes based on mood
        filtered_list = df[df["mood"] == mood]
        
        # convert to list
        return filtered_list


@app.get("/getEmotion")
async def get_emotion():
    output = record_video(5)
    return output

@app.get("/getSongs/{mood}")
async def get_songs_by_mood(mood):
    if mood is None or mood == "null":
        mood = None
        
    songs_data = get_songs_list(mood)
    songs = [
    {
        "name":row["song_name"],
        "uri":row["uri"],
    }
    for _, row in songs_data.iterrows()
    ]
    return songs[0:min(5,len(songs))]

@app.get("/playtrack/{song_uri}")
async def play_track(song_uri):
    os.system("open /Applications/Spotify.app")
    time.sleep(4)

    # Replace 'YOUR_CLIENT_ID', 'YOUR_CLIENT_SECRET', and 'YOUR_REDIRECT_URI' with your actual Spotify credentials
    client_id = '734434dda671410ebb08597ecaa2253c'
    client_secret = 'de55f43549e348b98ac369866dafe593'
    redirect_uri = 'http://localhost:5173/callback'  # Make sure this matches your Spotify app settings

    # Set up the Spotify OAuth object with user authentication
    scope = 'user-modify-playback-state user-read-playback-state'
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri, scope=scope))

    # Get the list of user's available devices
    devices = sp.devices()
    print(devices)
    device_id = None

    # Check if there are available devices
    if devices['devices']:
        device_id = devices['devices'][0]['id']  # Use the first available device

    # Start playback of the specified track with the selected device
    sp.start_playback(device_id=device_id, uris=[song_uri])
