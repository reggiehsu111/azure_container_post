import os
#import magic
import urllib.request
from app import app
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'pkl'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('remote.html')

@app.route('/', methods=['POST'])
def upload_file():
	if request.method == 'POST':
		print(request)
		file = request.files['file']
		file.save("remote.pdf")
		return redirect('/')

if __name__ == "__main__":
    app.run(port=5001)