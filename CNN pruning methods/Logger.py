# -*- coding: utf-8 -*-

from enum import Enum


class Types(Enum):
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    SUCCESS = '\033[92m'
    INFO = '\033[94m'
    ENDC = '\033[0m'


class Logger(object):
    # This class can be extended
    @staticmethod
    def DebugWarning(msg: str):
        print(f"(EvoPruning) {Types.WARNING.value}Warning: {msg}{Types.ENDC.value}")
    
    @staticmethod
    def DebugError(msg: str):
        raise Exception(f"(EvoPruning) {Types.ERROR.value}Error: {msg}{Types.ENDC.value}")
    
    @staticmethod
    def DebugSuccess(msg: str):
        print(f"(EvoPruning) {Types.SUCCESS.value}Success: {msg}{Types.ENDC.value}")
    
    @staticmethod
    def DebugInfo(msg: str):
        print(f"(EvoPruning) {Types.INFO.value}Info: {msg}{Types.ENDC.value}")
    
    @staticmethod
    def DebugPlain(msg: str):
        print(f"(EvoPruning): {msg}")
