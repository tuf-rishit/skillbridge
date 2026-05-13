# 🌉 SkillBridge
**AI-Driven Skill Mapping & Career Alignment Platform**
Minor Project 2026 — Kanishka Rajak, Harshita Soni, Olisha Patel

---

## 📁 Project Structure

```
skillbridge/
├── app.py              # Flask application & routes
├── ai_engine.py        # AI/ML engine (recommendation, classification, NLP)
├── requirements.txt    # Python dependencies
└── templates/
    ├── index.html      # Landing page
    ├── signup.html     # User registration
    ├── login.html      # User login
    ├── profile.html    # Skill & interest input (Step 2: Profile Input)
    └── results.html    # Career recommendations output (Step 4: Results)
```

---

## 🚀 Setup & Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python app.py
```

### 3. Open in Browser
```
http://localhost:5000
```

---

## 🤖 AI Concepts Used

| Concept | Where | Details |
|---------|-------|---------|
| **Content-Based Filtering** | Career Recommendations | TF-IDF vectors + cosine similarity to match user profile with career database |
| **Naive Bayes Classification** | Domain Classification | Classifies user into Tech / Design / Management / Marketing |
| **NLP / TF-IDF** | Resume Parsing | Extracts keywords and matched skills from resume text |
| **Skill Gap Analysis** | Gap Identification | Compares user skills vs. required skills for top matches |

---

## 🔌 API Endpoint (No Login Required)

```
POST /api/analyze
Content-Type: application/json

{
  "skills": ["Python", "SQL", "Problem Solving"],
  "interests": ["Technology", "Data"],
  "education": "Bachelor's Degree"
}
```

**Response:**
```json
{
  "domain": "Tech",
  "recommendations": [...],
  "skill_gaps": [...],
  "courses": [...]
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python / Flask |
| Database | SQLite (via SQLAlchemy) |
| AI / ML | scikit-learn (TF-IDF, Naive Bayes, Cosine Similarity) |
| NLP | TF-IDF Vectorizer |

---

## 📈 Future Scope
- AI Chatbot for real-time guidance
- Job Portal Integration
- Internship Recommendations
- Personality Test Integration
