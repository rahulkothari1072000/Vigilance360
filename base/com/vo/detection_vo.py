from base import db
from base import app
from datetime import datetime
import time


class AdminVO(db.Model):
    __tablename__ = 'login_table'
    login_id = db.Column(db.Integer, primary_key=True)
    login_username = db.Column(db.String(100), unique=True, nullable=False)
    login_password = db.Column(db.String(100), nullable=False)
    created_on = db.Column(db.Integer, default=lambda: int(time.time()))  
    modified_on = db.Column(db.Integer, onupdate=lambda: int(time.time()))  


class DetectionVO(db.Model):
    __tablename__ = 'detection_table' 
    detection_id = db.Column(db.Integer, primary_key=True)
    input_video_path = db.Column(db.String(255))
    output_video_path = db.Column(db.String(255))
    camera_location = db.Column(db.String(100))
    detection_source = db.Column(db.String(100))
    detection_starting_time = db.Column(db.Integer)
    total_detected_person = db.Column(db.Integer)
    detection_statistics = db.Column(db.Text)
    occupancy_anomaly = db.Column(db.Text)
    weapon_detected=db.Column(db.Boolean)
    detection_completion_time = db.Column(db.Integer)
    graph_path = db.Column(db.String(255))
    created_on = db.Column(db.Integer)
    modified_on = db.Column(db.Integer)
    created_by = db.Column(db.Integer, db.ForeignKey('login_table.login_id'))
    modified_by = db.Column(db.Integer ,db.ForeignKey('login_table.login_id'))

with app.app_context():
    db.create_all()