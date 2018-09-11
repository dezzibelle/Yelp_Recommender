from flask_wtf import FlaskForm
from wtforms import StringField, BooleanField, SubmitField, IntegerField
from wtforms.validators import DataRequired, Length, NumberRange

class LoginForm(FlaskForm):
    zipcode = IntegerField('Zipcode', validators=[DataRequired(), NumberRange(min=501, max=99950)])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('submit')