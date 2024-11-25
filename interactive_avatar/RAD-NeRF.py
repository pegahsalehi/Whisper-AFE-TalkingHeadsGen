import os
import os.path
from glob import glob
import glob
from IPython.display import HTML
from base64 import b64encode

Pose_start = 0 
Pose_end = 100 

def get_latest_file(path):
  dir_list = glob.glob(path)
  dir_list.sort(key=lambda x: os.path.getmtime(x))
  return dir_list[-1]
c = 3
while True:

    if os.path.isfile(f'./data/speech.wav'):
        
        %run asr_whisper.py 

        %run test.py -O --torso \
            --pose data/donya2/transforms_train.json \
            --data_range {Pose_start} {Pose_end} \
            --ckpt data/donya2/ngp_ep0031.pth \
            --aud data/speech_whisper.npy \
            --bg_img data/donya2/bc.jpg \
            --workspace trial

        Video = get_latest_file(os.path.join('trial', 'results', '*.mp4'))
        Video_aud = Video.replace('.mp4', '_aud.mp4')
        ! ffmpeg -y -i {Video} -i data/speech.wav -c:v copy -c:a aac {Video_aud}
                
        source = './trial/results/ngp_ep0031_aud.mp4'
        dest = './trial/results/{}.mp4'.format(c)
        os.rename(source, dest)
        
        
        c = c + 1
        os.remove('./data/speech.wav')
