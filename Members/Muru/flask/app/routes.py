from flask import render_template
from app import app
from app.forms import LoginForm
from flask import flash, redirect

@app.route('/', methods=['GET', 'POST'])
@app.route('/index',methods=['GET', 'POST'])
def index():
    form = LoginForm()
    if form.validate_on_submit():
        flash('zipcode requested for user {}, remember_me={}'.format(
            form.zipcode.data, form.remember_me.data))
        return redirect('/index')
    return render_template('index.html', title='Choosing the best restaurant', form=form)