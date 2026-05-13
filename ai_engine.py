"""
SkillBridge - AI Engine
Handles:
  - Career Recommendation (Content-Based Filtering)
  - Domain Classification (Supervised ML)
  - Skill Gap Identification
  - Course Recommendations
  - NLP-based Resume Keyword Extraction
  - Resume Tips Generator
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
import numpy as np

# ─────────────────────────────────────────
# Career Knowledge Base
# ─────────────────────────────────────────

CAREER_DATABASE = [
    {
        "title": "Software Developer",
        "required_skills": ["python", "java", "javascript", "data structures", "algorithms",
                            "git", "problem solving", "html", "css", "react"],
        "interests": ["technology", "coding", "software", "web development", "apps"],
        "domain": "Tech",
    },
    {
        "title": "Data Analyst",
        "required_skills": ["excel", "sql", "python", "statistics", "tableau", "power bi",
                            "data visualization", "problem solving", "pandas", "numpy"],
        "interests": ["data", "analytics", "business", "numbers", "research"],
        "domain": "Tech",
    },
    {
        "title": "AI / ML Engineer",
        "required_skills": ["python", "machine learning", "deep learning", "tensorflow",
                            "scikit-learn", "mathematics", "statistics", "nlp", "pytorch"],
        "interests": ["artificial intelligence", "technology", "research", "data"],
        "domain": "Tech",
    },
    {
        "title": "UI/UX Designer",
        "required_skills": ["figma", "adobe xd", "user research", "wireframing", "prototyping",
                            "css", "html", "creativity", "design thinking"],
        "interests": ["design", "creativity", "art", "user experience", "technology"],
        "domain": "Design",
    },
    {
        "title": "Product Manager",
        "required_skills": ["communication", "leadership", "agile", "jira", "roadmapping",
                            "problem solving", "analytics", "stakeholder management"],
        "interests": ["business", "technology", "strategy", "management", "leadership"],
        "domain": "Management",
    },
    {
        "title": "Digital Marketing Specialist",
        "required_skills": ["seo", "google ads", "social media", "content writing",
                            "analytics", "email marketing", "creativity", "communication"],
        "interests": ["marketing", "social media", "business", "creativity", "communication"],
        "domain": "Marketing",
    },
    {
        "title": "Cybersecurity Analyst",
        "required_skills": ["networking", "linux", "python", "ethical hacking", "firewall",
                            "penetration testing", "cryptography", "problem solving"],
        "interests": ["security", "technology", "hacking", "networking"],
        "domain": "Tech",
    },
    {
        "title": "Cloud Engineer",
        "required_skills": ["aws", "azure", "gcp", "docker", "kubernetes", "linux",
                            "devops", "ci/cd", "python", "networking"],
        "interests": ["cloud", "technology", "infrastructure", "devops"],
        "domain": "Tech",
    },
    {
        "title": "Business Analyst",
        "required_skills": ["excel", "sql", "communication", "requirements gathering",
                            "problem solving", "documentation", "stakeholder management"],
        "interests": ["business", "analysis", "strategy", "management"],
        "domain": "Management",
    },
    {
        "title": "Graphic Designer",
        "required_skills": ["photoshop", "illustrator", "creativity", "typography",
                            "color theory", "branding", "figma"],
        "interests": ["art", "design", "creativity", "visual communication"],
        "domain": "Design",
    },
]

COURSE_DATABASE = {
    "python": {
        "name": "Python for Everybody",
        "platform": "Coursera",
        "url": "https://www.coursera.org/specializations/python",
    },
    "machine learning": {
        "name": "Machine Learning Specialization",
        "platform": "Coursera (Andrew Ng)",
        "url": "https://www.coursera.org/specializations/machine-learning-introduction",
    },
    "sql": {
        "name": "SQL for Data Science",
        "platform": "Coursera",
        "url": "https://www.coursera.org/learn/sql-for-data-science",
    },
    "javascript": {
        "name": "JavaScript Algorithms and Data Structures",
        "platform": "freeCodeCamp",
        "url": "https://www.freecodecamp.org/learn/javascript-algorithms-and-data-structures/",
    },
    "figma": {
        "name": "UI/UX Design with Figma",
        "platform": "YouTube / Designcode",
        "url": "https://www.youtube.com/results?search_query=figma+tutorial",
    },
    "aws": {
        "name": "AWS Cloud Practitioner Essentials",
        "platform": "AWS Training",
        "url": "https://aws.amazon.com/training/learn-about/cloud-practitioner/",
    },
    "docker": {
        "name": "Docker & Kubernetes: The Complete Guide",
        "platform": "Udemy",
        "url": "https://www.udemy.com/course/docker-and-kubernetes-the-complete-guide/",
    },
    "deep learning": {
        "name": "Deep Learning Specialization",
        "platform": "Coursera (Andrew Ng)",
        "url": "https://www.coursera.org/specializations/deep-learning",
    },
    "seo": {
        "name": "SEO Training Course",
        "platform": "HubSpot Academy",
        "url": "https://academy.hubspot.com/courses/seo-training",
    },
    "communication": {
        "name": "Improve Your English Communication Skills",
        "platform": "Coursera",
        "url": "https://www.coursera.org/specializations/improve-english",
    },
    "statistics": {
        "name": "Statistics with Python",
        "platform": "Coursera",
        "url": "https://www.coursera.org/specializations/statistics-with-python",
    },
    "react": {
        "name": "React - The Complete Guide",
        "platform": "Udemy",
        "url": "https://www.udemy.com/course/react-the-complete-guide-incl-redux/",
    },
    "linux": {
        "name": "Introduction to Linux",
        "platform": "edX (Linux Foundation)",
        "url": "https://www.edx.org/course/introduction-to-linux",
    },
    "leadership": {
        "name": "Leading Teams",
        "platform": "Coursera (Michigan)",
        "url": "https://www.coursera.org/learn/leading-teams",
    },
}

DOMAIN_LABELS = ["Tech", "Design", "Management", "Marketing"]

# ─────────────────────────────────────────
# NLP Utilities
# ─────────────────────────────────────────

def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).strip()


def _extract_keywords(text: str) -> list[str]:
    """Simple keyword extractor using TF-IDF on a single document."""
    tokens = _normalize(text).split()
    stopwords = {"and", "the", "for", "with", "from", "this", "that", "have",
                 "are", "was", "has", "but", "not", "will", "can", "more"}
    return [t for t in tokens if t not in stopwords and len(t) > 2]


# ─────────────────────────────────────────
# Main AI Engine Class
# ─────────────────────────────────────────

class SkillBridgeAI:

    def __init__(self):
        self._vectorizer = TfidfVectorizer()
        self._build_career_vectors()
        self._train_classifier()

    # ── Internal Setup ──────────────────────────────────────────

    def _build_career_vectors(self):
        """Build TF-IDF vectors for all career profiles."""
        career_docs = [
            " ".join(c["required_skills"] + c["interests"])
            for c in CAREER_DATABASE
        ]
        self._career_matrix = self._vectorizer.fit_transform(career_docs)

    def _train_classifier(self):
        """Train a Naive Bayes domain classifier."""
        X_train, y_train = [], []
        for career in CAREER_DATABASE:
            doc = " ".join(career["required_skills"] + career["interests"])
            X_train.append(doc)
            y_train.append(DOMAIN_LABELS.index(career["domain"]))

        X_vec = self._vectorizer.transform(X_train)
        self._classifier = MultinomialNB()
        self._classifier.fit(X_vec, y_train)

    # ── Public Methods ──────────────────────────────────────────

    def recommend_careers(
        self,
        skills: list[str],
        interests: list[str],
        education: str,
        top_n: int = 5,
    ) -> list[dict]:
        """
        Content-Based Filtering: matches user profile against career database
        using cosine similarity on TF-IDF vectors.
        """
        user_doc = " ".join(
            [_normalize(s) for s in skills]
            + [_normalize(i) for i in interests]
            + _extract_keywords(education)
        )
        user_vec = self._vectorizer.transform([user_doc])
        similarities = cosine_similarity(user_vec, self._career_matrix).flatten()

        top_indices = similarities.argsort()[::-1][:top_n]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            career = CAREER_DATABASE[idx]
            match_level = (
                "High Match" if score >= 0.4
                else "Medium Match" if score >= 0.2
                else "Potential Match"
            )
            results.append({
                "title": career["title"],
                "domain": career["domain"],
                "match_level": match_level,
                "similarity_score": round(score, 3),
                "required_skills": career["required_skills"],
            })

        return results

    def classify_domain(self, skills: list[str], interests: list[str]) -> str:
        """Classify user into a career domain using Naive Bayes."""
        user_doc = " ".join(
            [_normalize(s) for s in skills]
            + [_normalize(i) for i in interests]
        )
        vec = self._vectorizer.transform([user_doc])
        prediction = self._classifier.predict(vec)[0]
        return DOMAIN_LABELS[prediction]

    def identify_skill_gaps(
        self,
        user_skills: list[str],
        recommendations: list[dict],
    ) -> list[str]:
        """
        Identify skills required by top recommended careers that the user lacks.
        """
        user_skills_normalized = {_normalize(s) for s in user_skills}
        gap_set = set()

        for career in recommendations[:3]:  # Focus on top 3 matches
            for skill in career["required_skills"]:
                if _normalize(skill) not in user_skills_normalized:
                    gap_set.add(skill)

        return sorted(gap_set)

    def recommend_courses(self, skill_gaps: list[str]) -> list[dict]:
        """Map skill gaps to course recommendations."""
        courses = []
        matched = set()

        for gap in skill_gaps:
            gap_lower = _normalize(gap)
            for key, course in COURSE_DATABASE.items():
                if key in gap_lower or gap_lower in key:
                    if key not in matched:
                        courses.append({
                            "skill": gap,
                            "course_name": course["name"],
                            "platform": course["platform"],
                            "url": course["url"],
                        })
                        matched.add(key)

        return courses[:8]  # Limit to 8 suggestions

    def generate_resume_tips(
        self,
        skills: list[str],
        recommendations: list[dict],
    ) -> list[str]:
        """Generate contextual resume tips based on profile and matches."""
        tips = []

        top_career = recommendations[0]["title"] if recommendations else "your target role"
        domain = recommendations[0]["domain"] if recommendations else "Tech"

        tips.append(f"Tailor your resume headline to '{top_career}' to pass ATS filters.")
        tips.append(
            f"Quantify your achievements — e.g., 'Improved processing speed by 30%' instead of 'Improved performance'."
        )

        if domain == "Tech":
            tips.append("Add a dedicated 'Skills' section listing programming languages, frameworks, and tools.")
            tips.append("Include links to your GitHub profile and live project demos.")
        elif domain == "Design":
            tips.append("Add a link to your online portfolio (Behance/Dribbble) at the top of your resume.")
            tips.append("List design tools (Figma, Adobe XD) prominently in the skills section.")
        elif domain == "Management":
            tips.append("Highlight leadership experiences with measurable team/project outcomes.")
            tips.append("Use action verbs like 'Led', 'Managed', 'Coordinated', 'Delivered'.")
        elif domain == "Marketing":
            tips.append("Include metrics like 'Grew social following by 40%' or 'Increased CTR by 25%'.")
            tips.append("List certifications from Google, HubSpot, or Meta prominently.")

        if skills:
            keyword_list = ", ".join(skills[:4])
            tips.append(f"Use job-description keywords ({keyword_list}) throughout your resume to improve ATS ranking.")

        tips.append("Keep your resume to 1 page if you have under 3 years of experience.")

        return tips

    def parse_resume_text(self, resume_text: str) -> dict:
        """
        NLP: Extract skills and keywords from raw resume text using TF-IDF.
        """
        keywords = _extract_keywords(resume_text)

        # Match against known skills in our database
        all_known_skills = set()
        for career in CAREER_DATABASE:
            all_known_skills.update(career["required_skills"])

        found_skills = [
            skill for skill in all_known_skills
            if all(word in resume_text.lower() for word in skill.split())
        ]

        return {
            "extracted_keywords": keywords[:20],
            "matched_skills": found_skills,
        }
