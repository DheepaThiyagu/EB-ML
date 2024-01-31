# from flask import current_app as app
from domain.models import db
from flask import current_app
from sqlalchemy import text
from datetime import datetime, timedelta
from appp import retrain

def schedule_jobs(app):
    # app_ids = fetch_applications(app)
    applications= fetch_applications(app)
    with app.app_context():
        # for app_id in app_ids:
        for wm_appl in applications:
            print("{}: scheduling for {}".format(datetime.now(), wm_appl.id))
            process_data(wm_appl)
            # app.scheduler.add_job(func=lambda: process_data(wm_appl), trigger='date', run_date=datetime.now() + timedelta(seconds=1))

def process_data(wm_appl):
    print("{} :processing data started for {}".format(datetime.now(), wm_appl.id))
    retrain(wm_appl)
    # Function to extract data from OpenSearch, do operations, and push data to OpenSearch
    print("{} :processing data completed for {}".format(datetime.now(), wm_appl.id))

    # Log completion details in the database
    # job_log = JobLog.query.filter_by(app_id=app_id, status='started').order_by(JobLog.job_start_timestamp.desc()).first()
    # if job_log:
    #     job_log.job_end_timestamp = datetime.utcnow()
    #     job_log.status = 'completed'
    #     # job_log.services_completed = ', '.join(list_of_service_ids)  # Replace with your actual list
    #     db.session.commit()

def fetch_applications(app):
    print("{} :fetching applications".format(datetime.now()))
    with app.app_context():
        sql_query = text("SELECT * FROM application WHERE predictive_ai_enabled=true;")
        applications = db.session.execute(sql_query).fetchall()
        # return [app.id for app in applications]
        return applications
