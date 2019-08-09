import json
import os

import analysis
import api
import cache

with open('config.json', 'r') as file:
    config = json.load(file)

def load_data ():
    api.get_token()
    devices = load_devices()
    data = load_devices_data(devices)
    analysis = load_analysis(data)
    return devices, data, analysis

def load_devices ():
    print('loading devices')
    complete = {}
    for company_id in config['companies']:
        print('checking for existing device list for company %s' % company_id)
        devices = cache.load_device_csv(company_id)
        if devices is None:
            _, devices = api.get_assets(company_id)
            print('saving %d device ids for %s' % (len(devices), company_id))
            cache.save_device_csv(company_id, devices)
        complete[company_id] = devices
    return complete

def load_devices_data (devices):
    print('loading device data')
    complete = {}
    for company_id, devices in devices.items():
        data = cache.load_company_data(company_id) or {}
        print('fetching device data from Athena for %s' % company_id)
        data, added = api.get_device_data(data, devices)
        complete[company_id] = data
        if added > 0:
            print('saving %d device data entrie(s) for %s' % (added, company_id))
            cache.save_company_data(company_id, data)
    return complete

def load_analysis(devices_data):
    print('loading analysis')
    complete = {}
    for company_id, device_data in devices_data.items():
        data = cache.load_company_analysis(company_id) or {}
        print('analyzing device data for %s' % company_id)
        data, added = analysis.process_company_analysis(company_id, data, device_data)
        complete[company_id] = data
        if added > 0:
            print('saving %d analysis entrie(s) for %s' % (added, company_id))
            cache.save_company_analysis(company_id, data)
    return complete
