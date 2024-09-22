# ----------------------------------------
# filename FlaskTest.py
# purpose Hello World Flask application
# following https://flask.palletsprojects.com/en/3.0.x/quickstart/
# author Joyce Weiner
# revision 1.0
# revision history 1.0 - initial Flask app
# run with python -m flask --app FlaskTest run
# ----------------------------------------

from flask import Flask
app=Flask(__name__)

@app.route("/")
def hello_world():
    return "<p> Hello World! </p>"