import json
import os
import pickle

with open('config.json', 'r') as file:
    config = json.load(file)

def load_device_csv (company_id):
    devices = None
    path = config['path_devices'] % (config['path_cache'], company_id)
    if os.path.isfile(path):
        devices = []
        with open(path, 'r') as file:
            devices = file.read().split('\n')
            devices = devices[1:]
            print('loaded %d devices from cache' % len(devices))
    return devices

def save_device_csv (company_id, devices):
    path = config['path_devices'] % (config['path_cache'], company_id)
    with open(path, 'w') as file:
        file.write('DEVICES\n')
        for device in devices:
            file.write('%s\n' % device)

def load_company_data (company_id):
    data = None
    path = config['path_data'] % (config['path_cache'], company_id)
    if os.path.isfile(path):
        data = {}
        with open(path, 'rb') as file:
            data = pickle.load(file)
            print('loaded %d data frames' % len(data))
    return data

def save_company_data (company_id, data):
    path = config['path_data'] % (config['path_cache'], company_id)
    with open(path, 'wb') as file:
        pickle.dump(data, file)

def load_company_analysis (company_id):
    analysis = None
    path = config['path_analysis'] % (config['path_cache'], company_id)
    if os.path.isfile(path):
        analysis = {}
        with open(path, 'rb') as file:
            analysis = pickle.load(file)
            print('loaded %d analysis data frames' % len(analysis))
    return analysis

def save_company_analysis (company_id, analysis):
    path = config['path_analysis'] % (config['path_cache'], company_id)
    with open(path, 'wb') as file:
        pickle.dump(analysis, file)

def load_report ():
    report = {}
    for company_id in config['companies']:
        report[company_id] = {}
        report[company_id]['data'] = load_company_data(company_id)
        report[company_id]['analysis'] = load_company_analysis(company_id)
    return report
