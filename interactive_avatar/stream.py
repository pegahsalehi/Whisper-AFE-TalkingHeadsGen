import cv2
import pyaudio
import pygame
import wave
import moviepy.editor as mp
import os
import subprocess

base_path = 'trial/results'
video_name = f'./{base_path}/Listening.mp4'
last_played = 2

def save_audio_h(vname):
    clip = mp.VideoFileClip(vname)
    clip.audio.write_audiofile(vname.replace('mp4', 'wav'))
    
def save_audio(vname):
    subprocess.call(['ffmpeg', '-i', vname, vname.replace('mp4', 'wav')])
    
def get_audio(vname, fps):

    all_song = []

    wf = wave.open(vname.replace('mp4', 'wav'), "rb")
    CHUNK = wf.getframerate() // fps
   

    p = pyaudio.PyAudio()

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
                frames_per_buffer=CHUNK)

    data = wf.readframes(CHUNK)

    while data:
        all_song.append(data)
        data = wf.readframes(CHUNK)

    return stream, all_song

def play(vname):

    cap = cv2.VideoCapture(vname)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    save_audio(vname)
    s, song = get_audio(vname, fps)

    clock = pygame.time.Clock()

    ind = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            video = cv2.resize(frame,(600,600))
            cv2.imshow('playing video ...', video)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            s.write(song[ind])
            ind += 1

            clock.tick(fps)
        else:
            break

    cap.release()
    #cv2.destroyAllWindows()

if __name__ == '__main__':

    while True:

        cap = cv2.VideoCapture(video_name)
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        save_audio(video_name)
        s2, song2 = get_audio(video_name, fps)

        clock = pygame.time.Clock()

        ind = 0

        while cap.isOpened():

            if os.path.isfile(f'./{base_path}/{last_played + 1}.mp4'):
                play(f'./{base_path}/{last_played + 1}.mp4')
                last_played += 1

            ret, frame = cap.read()


            if ret:
                video = cv2.resize(frame,(600,600))
                cv2.imshow('playing video ...', video)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                s2.write(song2[ind])
                ind += 1

                clock.tick(fps)
            else:
                break

        cap.release()
        cv2.destroyAllWindows()
