from flask import Flask

UPLOAD_FOLDER = './'

app = Flask(__name__)
app.secret_key = "secret key"