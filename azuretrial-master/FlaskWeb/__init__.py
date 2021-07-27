"""
The flask application package.
"""
from flask import Flask
import os
app = Flask(__name__)
app.secret_key = "secret key"
#app.config['UPLOAD_FOLDER'] = 'C:/Users/simhalapathir/Downloads/exampleflask-master/exampleflask-master'
app.config['UPLOAD_FOLDER'] = os.getcwd()
app.config['ALLOWED_EXTENSIONS'] = set(['txt','xlsx'])
wsgi_app = app.wsgi_app #Registering with IIS

import FlaskWeb.views
