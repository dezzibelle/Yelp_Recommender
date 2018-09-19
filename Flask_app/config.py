import os


class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    GOOGLEMAPS_KEY = "AIzaSyBUE6ZvtYrWkIU1y7Q5pgpjqZRkLH_Y1Uo"