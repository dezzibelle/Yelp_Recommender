from flask_wtf import FlaskForm
from wtforms import TextField, TextAreaField, StringField, BooleanField, SubmitField, IntegerField
from wtforms.validators import DataRequired, Length, NumberRange

class ChoiceForm(FlaskForm):
    zipcode = IntegerField(validators=[DataRequired(), NumberRange(min=501, max=99950)])
    choice1 = TextField ('choice1',validators=[DataRequired()])
    choice2 = TextField (validators=[DataRequired()])
    location = BooleanField('Share my ocation')
    submit = SubmitField('Submit')