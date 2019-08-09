import json
import os
import requests
import time
from datetime import datetime, timedelta
from pyathena import connect
from pyathena.util import as_pandas
from tqdm import tqdm_notebook

with open('config.json', 'r') as file:
    config = json.load(file)

def get_start ():
    start = get_stop() - timedelta(config['days'])
    return datetime(start.year, start.month, start.day, 0, 0, 0, 0)

def get_stop ():
    stop = datetime.now() - timedelta(1)
    return datetime(stop.year, stop.month, stop.day, 23, 59, 59, 999)

def get_request (url, headers={}, params={}):
    if 'token' in config:
        headers = { 'authorization': 'Bearer %s' % config['token'] }
    r = requests.get(config['url_api'] + url, headers=headers, params=params)
    return r.json() if r.status_code == 200 else None
    
def get_token ():
    params = {
        'username': os.environ['PARSYL_USERNAME'],
        'password': os.environ['PARSYL_PASSWORD']
    }
    data = get_request('/oauth/token', params=params)
    config['token'] = data['access_token'] if data else None

def get_assets (company_id, asset_type='refrigerator'):
    data = get_request('/companies/%s/assets' % company_id)
    
    now = int(round(time.time() * 1000))
    assets, devices = [], []
    
    for d in data:
        if d['properties']['type'] != asset_type:
            continue
        
        asset = {
            'id': d['id'],
            'name': d['name'],
            'type': d['properties']['type'],
            'location': d['locationName'],
            'devices': []
        }
        
        if 'devices' in d:
            for device in d['devices']:
                if device['containerAssociationTime'] > now:
                    continue
                if 'containerDisassociationTime' in device and device['containerDisassociationTime'] < now:
                    continue
                if device['deviceId'] in config['skip_devices']:
                    continue
                devices.append(device['deviceId'])
                asset['devices'].append(device['deviceId'])

        assets.append(asset)
    
    return assets, devices

def get_asset_devices ():
    params = { 'assetAssociation': 'true' }
    data = get_request('/companies/%s/devices' % config['company_id'], params=params)
    return [ (lambda x: x['deviceId'])(x) for x in data ] if data else []

def get_device_data (data={}, devices=[]):
    added = 0
    start = int(get_start().strftime('%s')) * 1000
    stop = int(get_stop().strftime('%s')) * 1000

    c = connect(s3_staging_dir='s3://parsyl-athena-output-production-useast1', region_name='us-east-1')
    
    for device_id in tqdm_notebook(devices, desc='device data loaded'):
        if device_id in data:
            continue

        stmt = """
            SELECT time, temperature, humidity
            FROM parsyl_device_data_database.parsyl_data_lake_production_useast1_v3
            WHERE device=%(device_id)s AND temperature IS NOT NULL AND time >= %(start)d AND time <= %(stop)d
            ORDER BY time
        """
        try:
            with c.cursor() as cursor:
                cursor.execute(stmt, { 'device_id': device_id, 'start': start, 'stop': stop })
                data[device_id] = as_pandas(cursor)
                added += 1
        except Exception as e:
            print('ERROR querying device data - {}'.format(e))
    
    c.close()
    
    return data, added
