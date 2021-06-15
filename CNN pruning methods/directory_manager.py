# -*- coding: utf-8 -*-

import shutil
import os

from Logger import Logger as log


def remove_logs():
    shutil.rmtree('./logs/', ignore_errors=True)
    log.DebugSuccess("logs folder clear")


def create_logs_dir():
    if os.path.isdir('./logs'):
        return
    os.mkdir('./logs')
    log.DebugSuccess("logs folder created")


def create_summary_dir():
    if os.path.isdir('./summary'):
        return
    os.mkdir('./summary')
    log.DebugSuccess("summary folder created")


def create_dir_in_summaries(name):
    if os.path.isdir('./summary/'+name):
        log.DebugWarning(f"Folder {name} already exist under summaries folder, nothing created")
        return
    os.mkdir('./summary/'+name)
    log.DebugSuccess(f"Folder {name} created")


def prepare_directory():
    remove_logs()
    create_logs_dir()
    create_summary_dir()
