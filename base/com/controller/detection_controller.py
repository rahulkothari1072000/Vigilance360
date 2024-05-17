from datetime import datetime
import matplotlib
matplotlib.use('agg')  # Set the backend to 'agg'

# Now you can import other Matplotlib functions
import matplotlib.pyplot as plt
import bcrypt
from flask import Flask, Response, jsonify, session, render_template, request, \
    redirect, url_for,flash
from flask_wtf import FlaskForm
from werkzeug.security import check_password_hash
from wtforms import FileField, SubmitField, StringField, DecimalRangeField, \
    PasswordField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired, NumberRange
import os
import cv2
from base.com.service.detection_service import InnerCamera, OuterCamera, \
    DisplayObject
from base.com.vo.detection_vo import AdminVO,DetectionVO
from base.com.dao.detection_dao import AdminDAO,DetectionDAO
from base import app
import hashlib
valid_username = "admin"
valid_password = "admin"
app.config['SECRET_KEY']='kavisha'
session_camera={}
login_required_message="Login Required"



class UploadFileFrom(FlaskForm):
    file = FileField("File",validators =[InputRequired()])
    submit = SubmitField("Run")

@app.after_request
def clear_cache(response):
    response.headers.add("Cache-controll","no-store")
    return response
    
        

def generate_frames_inner(path_x):
    try:
        session_camera["admin"]=InnerCamera(path_x)
        yolo_output= session_camera['admin'].inner_camera_detection()
        for detection_ in yolo_output:
            ref,buffer=cv2.imencode('.jpg',detection_)

            frame=buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')
    except Exception as e:
        print(e)

    
def generate_frames_outer(path_x):
    try:
        session_camera["admin"]=OuterCamera(path_x)
        yolo_output_outside= session_camera['admin'].outside_camera_detection()
        for detection_ in yolo_output_outside:
            ref,buffer=cv2.imencode('.jpg',detection_)

            frame=buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')
    except Exception as e:
        print(e)       


@app.route("/", methods=['GET'])
def login_page():
    
    

    
    return render_template("login.html")


@app.route("/login", methods=['POST'])
def login():
    # Fetch the provided username and password from the request form
    # session.clear()
    username = request.form.get("username")
    password = request.form.get("password")

    # Create an instance of AdminDAO
    admin_dao = AdminDAO()

    # Fetch the admin data from the database by username
    admin = admin_dao.get_admin_by_username(username)

    # Check if the admin exists
    if admin:
        # Compare the provided password with the stored hashed password
        if bcrypt.checkpw(password.encode('utf-8'),
                          admin.login_password.encode('utf-8')):
            session['login_id'] = admin.login_id
            # If the password matches, redirect to the home page
            return redirect(url_for('selectcamera'))
        else:
            # If the password doesn't match, return to the login page with an error message
            
            error_message = "Invalid credentials. Please try again."
            return render_template("login.html", error_message=error_message)
    else:
        # If the admin does not exist, return to the login page with an error message
        error_message = "Username not found. Please try again."
        return render_template("login.html", error_message=error_message)



@app.route("/selectcamera")
def selectcamera():
    try:
        if session.get('login_id', 0) > 0:
            return render_template('select-camera.html')
        else:
        
            return redirect('/')
    except Exception as e:
        return render_template('errorPage.html', error=e)
   
            

        
 
    

@app.route("/source", methods=["POST"])
def sourceforinner():
    session['video_path']=""

    
    camera_location = request.form.get('camera')
    if camera_location == 'inner':
        return redirect('http://127.0.0.1:5059/inner-camera')
    elif camera_location == 'outer':
        return redirect('http://127.0.0.1:5059/outer-camera')
    else:
        return 'Invalid camera location'
   


@app.route('/inner-camera')
def inner_camera():
    session['video_path']=""
    return render_template('inner.html')

@app.route('/outer-camera')
def outer_camera():
    session['video_path']=""
    return render_template('outer.html')

@app.route('/inner-upload-video', methods=['GET','POST'])
def inner_upload_video():
    form= UploadFileFrom()
    if form.validate_on_submit():
        path_url=r'base/static/uploads'
        file = form.file.data
        file.save(os.path.join(path_url,secure_filename(file.filename)))

        session['video_path']= os.path.join(path_url, secure_filename(file.filename))

    return render_template('inner-upload-video.html', form=form)

@app.route('/inner-video')
def inner_video():
    if not session.get("video_path"):
        return Response()

    return Response(generate_frames_inner(path_x=session.get('video_path', None)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')





@app.route('/outer-upload-video', methods=['GET','POST'])
def outer_upload_video():
    form= UploadFileFrom()
    if form.validate_on_submit():
        path_url=r'base/static/uploads'
        file = form.file.data
        file.save(os.path.join(path_url,secure_filename(file.filename)))

        session['video_path']= os.path.join(path_url, secure_filename(file.filename))

    return render_template('outer-upload-video.html', form=form)


@app.route('/outer-video')
def outer_video():
    if not session.get("video_path"):
        return Response()

    return Response(generate_frames_outer(path_x=session.get('video_path', None)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')





@app.route("/inner_webcam", methods=['GET','POST'])
def inner_webcam():
    session['video_path']=""
    # session.clear()
    return render_template('innerweb.html')


@app.route("/outer_webcam", methods=['GET','POST'])
def outer_webcam():
    session['video_path']=""
    # session.clear()
    return render_template('outerweb.html')



@app.route('/inner_web')
def inner_web():
    session['video_path']=0
    return Response(generate_frames_inner(path_x=0),mimetype='multipart/x-mixed-replace; boundary=frame',)

@app.route('/outer_web')
def outside_web():
    session['video_path']=0
    return Response(generate_frames_outer(path_x=0),mimetype='multipart/x-mixed-replace; boundary=frame',)


@app.route('/inner_camera_stop', methods=["GET", "POST"])
def result_inner():
    try:
        session['video_path']=""
    
        detect_vo=DetectionVO()
        detect_dao=DetectionDAO()
        input_video_path, output_video_path, camera_location, detection_source, detection_starting_time, total_detected_person, data, occupancy_anomaly_time,weapon, detection_completion_time,graph_path, created_on, modified_on = session_camera['admin'].stop_inner_camera()
        detect_vo.input_video_path=input_video_path
        detect_vo.output_video_path=output_video_path
        detect_vo.camera_location=camera_location
        detect_vo.detection_source=detection_source
        detect_vo.detection_starting_time=detection_starting_time
        detect_vo.total_detected_person=total_detected_person
        detect_vo.detection_statistics=data
        detect_vo.occupancy_anomaly=occupancy_anomaly_time
        detect_vo.weapon_detected=weapon
        detect_vo.detection_completion_time=detection_completion_time
        detect_vo.graph_path=graph_path
        
        detect_vo.created_on=created_on
        detect_vo.modified_on=modified_on
        detect_vo.created_by = session.get('login_id')
        detect_vo.modified_by = session.get('login_id')
        
        detect_dao.insert_person_counts(detect_vo)
        detection = detect_dao.get_all_counts()
        person_data=DisplayObject.list_from_json(detection)
    
        
        

        return render_template('history.html', data=person_data)

    except Exception as e:
        return render_template('errorPage.html', error=e)

@app.route('/outer_camera_stop', methods=["GET", "POST"])
def result_outer():
    try:
        session['video_path']=""
    

        detect_vo=DetectionVO()
        detect_dao=DetectionDAO()
        input_video_path, output_video_path, camera_location, detection_source, detection_starting_time, total_detected_person, data, occupancy_anomaly_time, weapon, detection_completion_time,graph_path, created_on, modified_on = session_camera['admin'].stop_outer_camera()
        detect_vo.input_video_path=input_video_path
        detect_vo.output_video_path=output_video_path
        detect_vo.camera_location=camera_location
        detect_vo.detection_source=detection_source
        detect_vo.detection_starting_time=detection_starting_time
        detect_vo.total_detected_person=total_detected_person
        detect_vo.detection_statistics=data
        detect_vo.occupancy_anomaly=occupancy_anomaly_time
        detect_vo.weapon_detected=weapon
        detect_vo.detection_completion_time=detection_completion_time
        detect_vo.graph_path=graph_path
        
        detect_vo.created_on=created_on
        detect_vo.modified_on=modified_on
        detect_vo.created_by = session.get('login_id')
        detect_vo.modified_by = session.get('login_id')
        
        detect_dao.insert_person_counts(detect_vo)
        detection = detect_dao.get_all_counts()
        person_data=DisplayObject.list_from_json(detection)
        
        # session.clear()

        return render_template('history.html', data=person_data)

    except Exception as e:
        return render_template('errorPage.html', error=e)    



@app.route("/output/<variable>", methods=["GET"])
def output(variable):
    
    return render_template('output_stream.html', data=variable)

@app.route("/uploads/<variable>", methods=["GET"])
def uploads(variable):
  
    return render_template('input_stream.html', data=variable)


    



@app.route("/graph/<path>", methods=["GET"])    
def visual(path):
    detect_vo=DetectionVO()
    detect_dao=DetectionDAO()
    person_statistics=detect_dao.get_statistics(path)
    # session.clear()

    person_data=DisplayObject.list_from_json(person_statistics)
    return render_template('graph.html' , data=person_data)

@app.route("/history")
def show_history_page():
    session['video_path']=""
    detect_vo=DetectionVO()
    detect_dao=DetectionDAO()
    detection = detect_dao.get_all_counts()
    person_data=DisplayObject.list_from_json(detection)
    
    # session.clear()
    return render_template('history.html', data=person_data)


@app.route("/about", methods=["GET"])
def about():
  
    return render_template('about-us.html')
    

@app.route('/logout')
def logout():
    try:
        session.clear()
        return redirect('/')

    except Exception as e:
        return render_template('errorPage.html', error=e)
