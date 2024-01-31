
from flask import Flask
from domain.models import db
from jobs.scheduler import configure_scheduler
# from config import Config
import logging
app = Flask(__name__)
# app.config.from_object(Config)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://watermelon:watermelon123@ec2-13-229-67-11.ap-southeast-1.compute.amazonaws.com:30003/wmebservices'
db.init_app(app)

if __name__ == "__main__":
    with app.app_context():
        logging.info("starting app")
        scheduler = configure_scheduler(app)
        app.scheduler = scheduler
        app.run(debug=False)