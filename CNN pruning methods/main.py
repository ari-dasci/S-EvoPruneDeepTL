# -*- coding: utf-8 -*-

import argparse

from experiment import Experiment
from Logger import Logger as log
from saver import results_mean, load_json

parser = argparse.ArgumentParser()
parser.add_argument('--runs',dest='runs', type=int, default=5)
parser.add_argument('--dataset',dest='dataset',type=str,default="")
arguments = parser.parse_args()

dataset_names = list(load_json(arguments.dataset).keys())

for i in range(arguments.runs):
    log.DebugWarning(f"___ EXPERIMENTATION RUN {i+1}/{arguments.runs} STARTED ___")
    for name in dataset_names:
        Experiment(name, arguments.dataset, folder_name=f"{name}_{i}")

# Sacar la media de todos los runs
for name in dataset_names:
    data = [load_json(f"./summary/{name}_{i}/{name}_results.json") for i in range(arguments.runs)]
    results_mean(data, f'mean_{name.lower()}.json')

