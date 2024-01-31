from apscheduler.schedulers.background import BackgroundScheduler
from jobs.operations import schedule_jobs
from datetime import datetime, timedelta
def configure_scheduler(app):
    scheduler = BackgroundScheduler()
    # scheduler.add_job(func=schedule_jobs, trigger='cron', hour=Config.SCHEDULER_HOUR)
    # scheduler.add_job(func=lambda: schedule_jobs(app), trigger='cron', second='0,10,20,30,40,50')
    scheduler.add_job(func=lambda: schedule_jobs(app), trigger='date', run_date=datetime.now() + timedelta(seconds=2))
    # scheduler.add_job(func=lambda: schedule_jobs(app), trigger='cron', hour=1, minute=00, second=0)

    # Add other scheduler configurations here

    scheduler.start()
    return scheduler

# scheduler.start()
