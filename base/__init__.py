from datetime import timedelta

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

app = Flask(__name__, static_url_path='/static', static_folder='static')

CORS(app)

app.secret_key = 'EB;%n=*\B]?=&-})]o3bP?[!Sr(I?j^JaFbv(!ya+_lhvTA<CUV*QZm33b%}cy)'

app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:root@localhost:3306/dashboard?charset=latin1'

app.config['SQLALCHEMY_ECHO'] = True

app.config['SQLALCHEMY_MAX_OVERFLOW'] = 0

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

app.config['UPLOAD_PATH'] = 'base/static/uploads'

app.config['OUTPUT_PATH'] = 'base/static/outputs'

app.config['MODEL_PATH'] = 'base/static/models'

db = SQLAlchemy(app)

from base.com import controller
