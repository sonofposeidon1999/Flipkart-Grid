import os
#import magic
import urllib.request
#from app import app
from pydub import AudioSegment
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename




app = Flask(__name__)
app.secret_key = "secret key"

ALLOWED_EXTENSIONS = set(['mp3','m4a','wav','ogg','mpeg'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app.config['UPLOAD_FOLDER']=r'upload'

@app.route('/', methods=['POST'])
def upload_file():
	if request.method == 'POST':
		# check if the post request has the files part
		if 'files[]' not in request.files:
			flash('No file part')
			return redirect(request.url)
		files = request.files.getlist('files[]')
		print(files)
		for file in files:
			if file.filename == '':
				flash('No selected file')
				return redirect(request.url)
			if file and allowed_file(file.filename):
                                try:
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
                                        audio.export('converted/'+dst, format="wav")
                                except:
                                        filename = file.filename
                                        file.save(os.path.join('C:/Users/91908/Desktop/error', filename))
		return "done"

if __name__ == "__main__":
	app.run()
