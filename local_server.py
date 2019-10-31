import os
#import magic
import urllib.request
from app import app
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import requests

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'pkl'])
REMOTE_URL = "http://ba79c317-12f2-4427-925d-e5f764e996de.southeastasia.azurecontainer.io/score"

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_file():
	if request.method == 'POST':
		file = request.files['file']
		response = requests.post(REMOTE_URL, files={'file': file})
		print(response.content)
		('File successfully uploaded')
		return redirect('/')

if __name__ == "__main__":
    app.run(port=5000)