# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""


import logging
from pathlib import Path


class CustomFormatter(logging.Formatter):
    def format(self, record):
        # Check if the filename starts with '__' or is '__main__.py'
        if record.filename == "__main__.py":
            # Extract the parent package name
            record.filename = Path(record.pathname).parent.name
        return super().format(record)


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "defaultFormatter": {
            "format": "%(asctime)s - %(filename)s - %(levelname)s - %(message)s",
            "datefmt": "%y-%m-%d %H:%M:%S",
        },
        "pkgFormatter": {
            "()": CustomFormatter,  # Use the custom formatter
            "format": "%(asctime)s - %(filename)s - %(levelname)s - %(message)s",
            "datefmt": "%y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "consoleHandler": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "defaultFormatter",
            "stream": "ext://sys.stdout",
        },
        "EconsoleHandler": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "defaultFormatter",
            "stream": "ext://sys.stderr",
        },
        "fileHandler": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "defaultFormatter",
            "filename": "eis_report.log",
            "mode": "a",
        },
        "RfileHandler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "pkgFormatter",
            "filename": "eis_report.log",
            "mode": "a",
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 2,  # Keep up to 5 backup files
        },
    },
    "loggers": {
        "eis_analysis": {
            "level": "WARNING",
            "handlers": ["RfileHandler"],
            "propagate": False,
        },
        # 'eis_logger': {
        #     'level': 'WARNING',
        #     # 'handlers': ['consoleHandler', 'fileHandler'],
        #     'handlers': ['fileHandler'],
        #     'propagate': False,
        # },
    },
    # 'root': {
    #     'level': 'NOTSET',
    #     'handlers': [],
    # },
}


def setup_logging():
    """Basic logging configuration."""
    logging.config.dictConfig(LOGGING_CONFIG)  # type: ignore
