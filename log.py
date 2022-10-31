# coding:utf-8

import sys
import logging
from logging.handlers import TimedRotatingFileHandler
import os, datetime

def init_logging(filename_prefix='?'):
    LOG_PATH = '/Users/jayhsu/work/competition/money_laundry/log'
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)

    root = logging.getLogger()
    root.handlers=[]
    level = logging.INFO
    filename = f'{LOG_PATH}/{filename_prefix}_{datetime.datetime.now().strftime("%Y-%m-%d")}.log'
    #logformat = '%(asctime)s %(levelname)s %(module)s.%(funcName)s Line:%(lineno)d %(message)s'
    logformat = '%(asctime)s %(levelname)s: %(message)s (%(filename)s:%(lineno)d)'
    timeformat = "%m-%d %H:%M:%S"
    logFormatter = logging.Formatter(logformat, timeformat)
    
    hdlr = TimedRotatingFileHandler(filename, "midnight", 1, 14)
    hdlr.setFormatter(logFormatter)
    root.addHandler(hdlr)
    root.setLevel(level)
    
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(CustomFormatter())
    consoleHandler.setLevel(logging.INFO)
    root.addHandler(consoleHandler)
    root.setLevel(level)


class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s %(levelname)s: %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)    