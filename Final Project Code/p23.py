from flask import Flask, render_template,flash, request, redirect, url_for,jsonify
import pandas as pd
import numpy as np
import json
import os
from speakerDiarization import main
from pydub import AudioSegment
import scipy.io.wavfile
import librosa
from scipy.io import wavfile
import scipy.io


ALLOWED_EXTENSIONS = {'mp3','m4a','mp4','wav'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

drive, tcase_dir = os.path.splitdrive(os.path.abspath(__file__))
path=drive+ tcase_dir
#paths = "/".join(path.split(os.sep)[:-1])+"/Output 2/"
final_path="/".join(path.split(os.sep)[:-1])+"/Final Result (Primary Audio and ASR text)/"

def diarize(paths,src):
    file1=main(paths+src, embedding_per_second=1.2, overlap_rate=0.4)
    print(file1)
    l={}
    sp=0
    for line in file1:
        s=line.split()
        if(s[0]=='========='):
            l[int(s[1])]=[]
            sp=int(s[1])
        else:
            l[sp].append([s[0],s[2]])
    time=[]
    for i in l:
        time.append([])
        for j in l[i]:
            t1=j[0].split(".")[0]
            t2=j[0].split(".")[1]
            t0=t1.split(":")[0]
            t1=t1.split(":")[1]
            sum1=int(t1)*1000+int(t2)+int(t0)*1000*60
            t1=j[1].split(".")[0]
            t2=j[1].split(".")[1]
            t0=t1.split(":")[0]
            t1=t1.split(":")[1]
            sum2=int(t1)*1000+int(t2)+int(t0)*1000*60
            time[-1].append([sum1,sum2])
    print(time)
    if(len(time)==1):
        print("Just one speaker")
        print(paths+src)
        audio_file= paths+src
        audio = AudioSegment.from_wav(audio_file)
        audio.export(final_path+src, format="wav")
    else:
        print(paths+src)
        audio_file= paths+src
        audio = AudioSegment.from_wav(audio_file)
        hop_length = 256
        frame_length = 512
        s=[]
        for i in time:
            su=0
            for j in i:
                audio_chunk=audio[j[0]:j[1]]
                audio_chunk.export( final_path+"chunk.wav", format="wav")
                sr,data=scipy.io.wavfile.read(final_path+"chunk.wav")
                #print(data)
                #su+=sum([i**2 for i in list(data)])
                su+=np.sum(data.astype(float)**2)
                print("su :",su)
                print(type(su))
            s.append(su)
        print(s)
        print(max(s))
        primary_sp=s.index(max(s))
        print("Speaker :",primary_sp)
        combined = AudioSegment.empty()
        for i in time[primary_sp]:
          audio_chunk=audio[i[0]:i[1]]
          combined+=audio_chunk
        combined.export(final_path+src, format="wav")
        os.remove(final_path+"chunk.wav")
