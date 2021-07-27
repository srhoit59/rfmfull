"""
Routes and views for the flask application.
"""
import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory,flash,session
from werkzeug import secure_filename
from FlaskWeb import app
import urllib.request
from io import StringIO,BytesIO
from azure.storage.blob import BlockBlobService
import pandas as pd

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def logie():
    session.permanent = False
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        return render_template('index.html')

@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )
@app.route('/upload')
def upload():
	return render_template('upload.html',
        year=datetime.now().year)

@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='Contact',
        year=datetime.now().year,
        message='Your contact page.'
    )

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='Your application description page.'
    )

@app.route('/forecast')
def forecast():
    print("done")
    return render_template(
        'index1.html',
        title='exec',
        year=datetime.now().year,
        message='Your exec page.'
    )
    
    
@app.route('/uploader', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        #file = request.files['file']
        uploaded_files = request.files.getlist("file[]")
        filenames = []
        for file in uploaded_files:
            
            if file.filename == '':
                flash('No file selected for uploading')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                blobservice = BlockBlobService(account_name='flaskstorage', account_key='M9Hax/c6wKCdVXIcmBafad35/ctWW2OQJQynRMrM29D+mfZXWW53MF0Sthsf0cmWN+/XukVg/aZQ/6XBAB4cgg==') 
                df=pd.read_excel(file.stream)
                data= BytesIO()
                df.to_excel(data, index=False)
                data=bytes(data.getvalue())
                data=BytesIO(data)
                blobservice.create_blob_from_stream('htflaskcontainer',filename,data)
                data.close()
                filenames.append(filename)
                flash('File(s) successfully uploaded to Blob','filename')
        return redirect('/upload')

@app.route('/download')
def download_file():
	url = 'https://codeload.github.com/fogleman/Minecraft/zip/master'
	response = urllib.request.urlopen(url)
	data = response.read()
	response.close()

@app.route('/exec')
def parse(name=None):
    from FlaskWeb import total
    print("done")
    return "Forecasting for TOTAL SDU has been completed"
@app.route('/exec1')
def parse1(name=None):
    from FlaskWeb import ASG
    print("done")
    return "Forecasting for DU ASG has been completed"
    
@app.route('/exec2')
def parse2(name=None):
    from FlaskWeb import MKT
    print("done")
    return "Forecasting for DU MKT has been completed"
@app.route('/exec3')
def parse3(name=None):
    from FlaskWeb import MOBILE
    print("done")
    return "Forecasting for DU MOBILE has been completed"
@app.route('/exec4')
def parse4(name=None):
    from FlaskWeb import SSE
    print("done")
    return "Forecasting for DU SSE has been completed"
@app.route('/exec5')
def parse5(name=None):
    from FlaskWeb import TTS
    print("done")
    return "Forecasting for DU TTS has been completed"
@app.route('/exec6')
def parse6(name=None):
    from FlaskWeb import EEG
    print("done")
    return "Forecasting for DU EEG has been completed"
@app.route('/exec7')
def parse7(name=None):
    from FlaskWeb import MTS
    print("done")
    return "Forecasting for DU MTS has been completed"
@app.route('/exec8')
def parse8(name=None):
    from FlaskWeb import total
    print("done Total")
    from FlaskWeb import ASG
    print("done ASG")
    from FlaskWeb import MKT
    print("done MKT")
    from FlaskWeb import MOBILE
    print("done Mobile")
    from FlaskWeb import SSE
    print("done SSE")
    from FlaskWeb import TTS
    print("done TTS")
    from FlaskWeb import EEG
    print("done EEG")
    from FlaskWeb import MTS
    print("done MTS")
    return "Forecasti ng completed for all"
    return render_template(
        'index1.html',
        title='exec',
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        year=datetime.now().year,
        message='Your exec page.'
    )
    
@app.route('/login', methods=['POST'])
def do_admin_login():
    j=0
    blobservice = BlockBlobService(account_name='flaskstorage', account_key='M9Hax/c6wKCdVXIcmBafad35/ctWW2OQJQynRMrM29D+mfZXWW53MF0Sthsf0cmWN+/XukVg/aZQ/6XBAB4cgg==') 
    byte_stream = BytesIO()
    blobservice.get_blob_to_stream(container_name='htflaskcontainer', blob_name='Authorisation.xlsx', stream=byte_stream)
    byte_stream.seek(0)
    userpass=pd.read_excel(byte_stream)
    byte_stream.close()
    for i in range(len(userpass['User-Name'])):
        if request.form['username'] == userpass['User-Name'][i]: 
            if request.form['password'] == userpass['Password'][i]:
                session['logged_in'] = True
                
            else:
                flash("Password for the given username didn't match!")
                return redirect('/')
        else:       
             j= j+1
             print('gh',j)
             if j >= len(userpass['User-Name']):
                 print(j)
                 flash("Seems like you are not authorised to access this Website")
    return redirect('/')

@app.route("/logout",methods=['POST'])
def logout():
    if request.method == 'POST':
        session['logged_in'] = False
        return redirect('/')

