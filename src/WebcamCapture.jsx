import axios from "axios";
import React, { useRef, useState } from 'react';

const WebcamCapture = () => {
    const webcamRef = useRef(null);
    const [recording, setRecording] = useState(false);
    const [songs, setSongs] = useState([]);

    const recordVideo = async () => {
        // setRecording(true);
        // const frames = [];

        // // Capture frames for 5 seconds
        // const duration = 5; // seconds
        // const startTime = new Date().getTime();

        // while (new Date().getTime() - startTime < duration * 1000) {
        //     const imageSrc = webcamRef.current.getScreenshot();
        //     frames.push(imageSrc);
        // }

        // setRecording(false);

        const { data: mood } = await axios.get(`http://localhost:5000/getEmotion`, {
            headers: {
                "Content-Type": "application/json"
            }
        });
        console.log("mood::", mood);

        const { data: songs } = await axios.get(`http://localhost:5000/getSongs/${mood}`, {
            headers: {
                "Content-Type": "application/json"
            }
        })

        console.log("Songs::", songs);
        setSongs(songs);
    }
    /**
     * 
     * @param {string} song_uri 
     */
    const handleRedirect = async (song_uri) => {
        // let spotify_url = song_uri;
        // window.open(spotify_url, "_blank");
        await axios.get(`http://localhost:5000/playtrack/${song_uri}`);
    }

    return (
        <div>
            {/* <Webcam
                audio={false}
                ref={webcamRef}
                hidden
            /> */}
            <button onClick={recordVideo}>
                Detect Current Emotion
            </button>
            {songs.length > 0 &&
                <div>
                    <h4>Songs List:</h4>
                    <ul>
                        {
                            songs?.map((song, idx) => {
                                return <li
                                    key={idx}
                                    onClick={() => handleRedirect(song?.uri)}
                                    style={{ cursor: 'pointer' }}
                                >
                                    {song?.name}
                                </li>
                            })
                        }
                    </ul>
                </div>}
        </div>
    );


}

export default WebcamCapture;
