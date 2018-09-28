from flask import Flask
from config import Config
from flask_googlemaps import GoogleMaps


app = Flask(__name__, static_url_path = '/static')
app.config.from_object(Config)
GoogleMaps(app)

from app import routes

