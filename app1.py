"""
SkillBridge - AI-Driven Skill Mapping & Career Alignment Platform
Main Flask Application
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from ai_engine import SkillBridgeAI
import os

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "skillbridge-secret-2026")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///skillbridge.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
ai_engine = SkillBridgeAI()

# ─────────────────────────────────────────
# Database Models
# ─────────────────────────────────────────

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    profiles = db.relationship("UserProfile", backref="user", lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class UserProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    skills = db.Column(db.Text, nullable=False)          # comma-separated
    interests = db.Column(db.Text, nullable=False)       # comma-separated
    education = db.Column(db.String(200), nullable=False)
    experience_years = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, server_default=db.func.now())


# ─────────────────────────────────────────
# Auth Routes
# ─────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        data = request.get_json() or request.form
        name = data.get("name", "").strip()
        email = data.get("email", "").strip().lower()
        password = data.get("password", "")

        if not name or not email or not password:
            return jsonify({"error": "All fields are required."}), 400

        if User.query.filter_by(email=email).first():
            return jsonify({"error": "Email already registered."}), 409

        user = User(name=name, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        session["user_id"] = user.id
        session["user_name"] = user.name
        return jsonify({"message": "Account created!", "redirect": "/profile"}), 201

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        data = request.get_json() or request.form
        email = data.get("email", "").strip().lower()
        password = data.get("password", "")

        user = User.query.filter_by(email=email).first()
        if not user or not user.check_password(password):
            return jsonify({"error": "Invalid email or password."}), 401

        session["user_id"] = user.id
        session["user_name"] = user.name
        return jsonify({"message": "Login successful!", "redirect": "/profile"}), 200

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


# ─────────────────────────────────────────
# Profile & Analysis Routes
# ─────────────────────────────────────────

@app.route("/profile", methods=["GET", "POST"])
def profile():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        data = request.get_json() or request.form
        skills_raw = data.get("skills", "")
        interests_raw = data.get("interests", "")
        education = data.get("education", "")
        experience = int(data.get("experience_years", 0))

        # Save profile
        user_profile = UserProfile(
            user_id=session["user_id"],
            skills=skills_raw,
            interests=interests_raw,
            education=education,
            experience_years=experience,
        )
        db.session.add(user_profile)
        db.session.commit()

        return jsonify({"message": "Profile saved!", "redirect": "/results"}), 200

    return render_template("profile.html", user_name=session.get("user_name"))


@app.route("/results")
def results():
    if "user_id" not in session:
        return redirect(url_for("login"))

    profile = (
        UserProfile.query
        .filter_by(user_id=session["user_id"])
        .order_by(UserProfile.created_at.desc())
        .first()
    )
    if not profile:
        return redirect(url_for("profile"))

    skills = [s.strip() for s in profile.skills.split(",") if s.strip()]
    interests = [i.strip() for i in profile.interests.split(",") if i.strip()]

    # Run AI engine
    recommendations = ai_engine.recommend_careers(skills, interests, profile.education)
    skill_gaps = ai_engine.identify_skill_gaps(skills, recommendations)
    courses = ai_engine.recommend_courses(skill_gaps)
    resume_tips = ai_engine.generate_resume_tips(skills, recommendations)
    domain = ai_engine.classify_domain(skills, interests)

    return render_template(
        "results.html",
        user_name=session.get("user_name"),
        skills=skills,
        interests=interests,
        education=profile.education,
        recommendations=recommendations,
        skill_gaps=skill_gaps,
        courses=courses,
        resume_tips=resume_tips,
        domain=domain,
    )


# ─────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Quick analysis without saving to DB (demo endpoint)."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    skills = data.get("skills", [])
    interests = data.get("interests", [])
    education = data.get("education", "Bachelor's")

    recommendations = ai_engine.recommend_careers(skills, interests, education)
    skill_gaps = ai_engine.identify_skill_gaps(skills, recommendations)
    courses = ai_engine.recommend_courses(skill_gaps)
    domain = ai_engine.classify_domain(skills, interests)

    return jsonify({
        "domain": domain,
        "recommendations": recommendations,
        "skill_gaps": skill_gaps,
        "courses": courses,
    })


# ─────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        print("✅ Database tables created.")
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
