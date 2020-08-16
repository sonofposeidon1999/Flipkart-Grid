import os
#import magic
import urllib.request
#from app import app
from pydub import AudioSegment
from flask import Flask, flash, request, redirect, render_template,jsonify
from werkzeug.utils import secure_filename
from PART1.test_api import noise_removal
from p23 import diarize
import requests
headers = {'Authorization' : 'Token 3715119fd7753d33bedbd3c2832752ee7b0a10c7'}
data = {'user' : '310' ,'language' : 'HI'}
url = 'https://dev.liv.ai/liv_transcription_api/recordings/'

drive, tcase_dir = os.path.splitdrive(os.path.abspath(__file__))
path=drive+ tcase_dir

app = Flask(__name__)
app.secret_key = "secret key"
drive, tcase_dir = os.path.splitdrive(os.path.abspath(__file__))
path=drive+ tcase_dir
paths = "/".join(path.split(os.sep)[:-1])+"/"

ALLOWED_EXTENSIONS = set(['mp3','m4a','wav','ogg','mpeg','flac'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app.config['UPLOAD_FOLDER']=r'Uploaded Audio'

def asr(file):
    files = {'audio_file' : open(file,'rb')}
    res = requests.request("POST",url, headers = headers, data = data, files = files)
    return res.json()['transcriptions'][0]['utf_text']

@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_file():
	if request.method == 'POST':
		# check if the post request has the files part
		if 'files[]' not in request.files:
			flash('No file part')
			return redirect(request.url)
		files = request.files.getlist('files[]')
		for file in files:
			if file.filename == '':
				flash('No selected file')
				return redirect(request.url)
			if file and allowed_file(file.filename):
				filename = file.filename
				file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
				print("done uploading :",app.config['UPLOAD_FOLDER']+"/"+filename)
				file_ext=filename.split(".")[-1]
				dst=filename.split(".")[0:-1]
				dst="".join(dst)
				# file_ext=e[-1]
				dst=dst+".wav"
				print(dst,file_ext)
				audio = AudioSegment.from_file(app.config['UPLOAD_FOLDER']+"/"+filename, file_ext)
				audio.export('Output 1(Conversion to wav)/'+dst, format="wav")
				try:
					noise_removal(paths+'Output 1(Conversion to wav)/',paths+'Output 2/')
					diarize("/".join(path.split(os.sep)[:-1])+"/Output 2/",dst)
				except:
					diarize("/".join(path.split(os.sep)[:-1])+"/Output 1(Conversion to wav)/",dst)
				text=asr("/".join(path.split(os.sep)[:-1])+"/Final Result (Primary Audio and ASR text)/"+dst)
				with open("/".join(path.split(os.sep)[:-1])+"/Final Result (Primary Audio and ASR text)/"+dst.split(".")[0]+".txt", "w+", encoding="utf-8") as f:
					f.write(text)
			os.remove(paths+'Output 2/'+dst)
			os.remove(paths+'Output 1(Conversion to wav)/'+dst)
		return jsonify(Path_to_Audio="/".join(path.split(os.sep)[:-1])+"/Final Result (Primary Audio and ASR text)/")

if __name__ == "__main__":
	app.run(threaded=False)
