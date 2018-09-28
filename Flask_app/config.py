import os


class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    GOOGLEMAPS_KEY = "AIzaSyAhtNGGZt-GN2AFdTWKlAAuzkGPszcQj8s"
