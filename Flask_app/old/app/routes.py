from flask import render_template
from flask import jsonify
from app import app
from app.forms import ChoiceForm
from flask import flash, redirect
from flask_googlemaps import googlemap
from Yelp_Erin_recent import process_restaurants
from flask import jsonify
import pandas as pd
import numpy as np
from flask import Flask, Response, render_template, request
import json


@app.route('/', methods=['GET', 'POST'])
@app.route('/index',methods=['GET', 'POST'])
def index():
    form = ChoiceForm()
    mymap = googlemap("my_awesome_map", style = "height:450px;width:1000px; margin:0;", zoom = 11, lat=36.114647, lng=-115.172813)
    list_recomm = []
    if form.validate_on_submit():
        #flash('User1_choice: {},  User2_choice: {}, located at ={}'.format(
            #form.choice1.data, form.choice2.data, form.zipcode.data))
        df_rest = pd.read_pickle("newdf_LVrestaurants25samples.pkl")
        df = process_restaurants(df_rest, form.choice1.data, form.choice2.data, user_zip = form.zipcode.data)
        mymap = googlemap("my_awesome_map", style = "height:600px;width:1100px; margin:0;", zoom = 11, lat=36.114647, lng=-115.172813, markers=[{
                'icon': 'http://maps.google.com/mapfiles/ms/icons/green-dot.png',
                'lat':  df.latitude.values[x],
                'lng':  df.longitude.values[x],
                'infobox': df.name.values[x]} for x in range(0,5)])
        list_recomm = [df.name.values[x] for x in range(0,5)]
                #flash(df[2]['name'] + ', ' + df[3]['name'] + ', ' + df[4]['name'] + 'and  ' + df[5]['name'])
        return render_template('index.html', title='Choosing the best restaurant', form=form, list_recomm=list_recomm, mymap= mymap)
    return render_template('index.html', title='Choosing the best restaurant', form=form, list_recomm=list_recomm, mymap= mymap)

df_rest = pd.read_pickle("newdf_LVrestaurants25samples.pkl")
df_rest['address'] = df_rest.address.fillna(' ')
df_rest['OPTIONS'] = df_rest['name']+' ('+df_rest['address']+')'

NAMES = list(set(df_rest.OPTIONS)) #set gets rid off duplicates in the column
print(NAMES[0:5])
@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    search = request.args.get('term')
    app.logger.debug(search)
    return jsonify(json_list = NAMES)
