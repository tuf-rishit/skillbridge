from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from ai_engine import SkillBridgeAI
import os

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "skillbridge-secret-2026")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///skillbridge.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize database

db = SQLAlchemy(app)
ai_engine = SkillBridgeAI()

# -------------------------------------------------
# Database Models
# -------------------------------------------------

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class UserProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    skills = db.Column(db.Text, nullable=False)
    interests = db.Column(db.Text, nullable=False)
    education = db.Column(db.String(200), nullable=False)
    experience_years = db.Column(db.Integer, default=0)


# -------------------------------------------------
# Routes
# -------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/signup')
def signup():
    return render_template('signup.html')


@app.route('/profile')
def profile():
    return render_template('profile.html')


@app.route('/results')
def results():
    return render_template('results.html')


# -------------------------------------------------
# Database Initialization
# -------------------------------------------------

with app.app_context():
    db.create_all()


# -------------------------------------------------
# Main
# -------------------------------------------------

if __name__ == '__main__':
    app.run(debug=False)

