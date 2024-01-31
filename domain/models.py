from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class JobLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    job_start_timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    job_end_timestamp = db.Column(db.DateTime)
    status = db.Column(db.String(20), nullable=False, default='started')
    app_id = db.Column(db.String(50), nullable=False)
    services_completed = db.Column(db.String(255))
