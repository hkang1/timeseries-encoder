import math
import numpy as np

def normalize_point (point, base_x):
    return [ point[0] - base_x, *point[1:] ]

def linear_regression (data, base_x):
    sums = { "x": 0., "y": 0., "xy": 0., "xs": 0., "ys": 0. }
    
    for d in data:
        npx = normalize_point(d, base_x)
        sums['x'] += npx[0]
        sums['y'] += npx[1]
        sums['xy'] += npx[0] * npx[1]
        sums['xs'] += npx[0] ** 2
        sums['ys'] += npx[1] ** 2
    
    denominator = len(data) * sums['xs'] - sums['x'] * sums['x']
    m = (len(data) * sums['xy'] - sums['x'] * sums['y']) / denominator
    b = (sums['y'] * sums['xs'] - sums['x'] * sums['xy']) / denominator
    
    return m, b

def sum_of_squared_errors (data, m, b, base_x):
    sse = 0.
    for i, d in enumerate(data):
        npx = normalize_point(d, base_x)
        y = m * npx[0] + b
        e = npx[1] - y
        sse += e ** 2
    return sse

def triangle_area (p1, p2, p3):
    return abs((p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2.)
    
def split_bucket (data, threshold):
    buckets = [None] * threshold
    size = (len(data) - 2) / (threshold - 2)
    for i, d in enumerate(data):
        if i == 0:
            buckets[i] = { "data": [d], "sse": None }
        elif i == len(data) - 1:
            buckets[threshold - 1] = { "data": [d], "sse": None }
        else:
            bi = math.floor(i / size) + 1
            if buckets[bi] is None:
                buckets[bi] = { "data": [d], "sse": None }
            else:
                buckets[bi]['data'].append(d)
    return buckets

def average_bucket_point (data, base_x):
    count = len(data)
    sums = [ 0, 0 ]
    for d in data:
        npx = normalize_point(d, base_x)
        sums[0] += npx[0]
        sums[1] += npx[1]
    return sums

def rank_triangle_three_buckets (buckets, base_x):
    points = []
    
    for index, bucket in enumerate(buckets):
        if index == 0 or index == len(buckets) - 1:
            points.append(bucket['data'][0])
        else:
            # Set the previous point to be the last selected point
            point_prev = normalize_point(points[-1], base_x)
            
            # Set the next point to be the average point of the next bucket
            bucket_next = buckets[index + 1]
            point_next = average_bucket_point(bucket_next['data'], base_x)
            
            # Find the bucket point that will maximize the triangle area
            ttb = { "point": None, "area": 0 }
            
            for d in bucket['data']:
                npx = normalize_point(d, base_x)
                area = triangle_area(point_prev, npx, point_next)
                if area > ttb['area'] or ttb['point'] is None:
                    ttb = { "point": d, "area": area }
            
            points.append(ttb['point'])

    return np.array(points)
    
def largest_triangle_dynamic (data, threshold=512):
    # No need to downsample if already less than threshold sample size
    if len(data) < threshold:
        return data

    buckets = split_bucket(data, threshold)
    
    # Use the base x date object to use as an offset to normalize regression line calculation
    base_x = data[0][0]
    
    # Number of iterations of merging or spliting buckets based on SSE
    iterations = math.floor(len(data) * 10 / threshold)
    
    for i in range(iterations):
        min_sse = { "index": None, "value": None }
        max_sse = { "index": None, "value": None }
        
        for index, bucket in enumerate(buckets):
            bucket_prev = buckets[index - 1] if index != 0 else None
            bucket_next = buckets[index + 1] if index != len(buckets) - 1 else None
            
            # Skip buckets previously calculated and not previously merged/split
            if bucket['sse'] is None:
                # Form overlapping buckets (includes one data point from previous and next buckets)
                overlapping_bucket = bucket['data'].copy()
                
                # Add the last point from the previous bucket if applicable
                if bucket_prev:
                    overlapping_bucket.insert(0, bucket_prev['data'][-1])
                
                # Add the first point from the next bucket if applicable
                if bucket_next:
                    overlapping_bucket.append(bucket_next['data'][0])
                
                # Calculate linear regression for overlapping buckets
                m, b = linear_regression(overlapping_bucket, base_x)
                
                # Calculate the sum of squared errors to determine which buckets to split or merge
                bucket['sse'] = sum_of_squared_errors(overlapping_bucket, m, b, base_x)
            
            # Find the bucket with the largest SSE with more than one item (to allow a split)
            if max_sse['value'] is None or (bucket['sse'] > max_sse['value'] and len(bucket['data']) > 1):
                max_sse['index'] = index
                max_sse['value'] = bucket['sse']
            
            # Find a pair of adjacent buckets with the smallest SSE
            if index != 0:
                sum_sse = bucket_prev['sse'] + bucket['sse']
                if min_sse['value'] is None or sum_sse < min_sse['value']:
                    min_sse['index'] = index - 1
                    min_sse['value'] = sum_sse
            
        # Because we are spliting the largest SSE bucket first
        # if the merging buckets come after split buckets,
        # the index should go up by one
        if min_sse['index'] and min_sse['index'] > max_sse['index']:
            min_sse['index'] += 1

        # Split the largest bucket
        bucket_max = buckets[max_sse['index']]
        mid_index = round(len(bucket_max['data']) / 2.)
        half0 = { "data": bucket_max['data'][0:mid_index], "sse": None }
        half1 = { "data": bucket_max['data'][mid_index:], "sse": None }
        buckets = [ *buckets[:max_sse['index']], half0, half1, *buckets[max_sse['index']+1:] ]

        # Merge the smallet buckets
        bucket_min0 = buckets[min_sse['index']]
        bucket_min1 = buckets[min_sse['index']+1]
        merged = { "data": [ *bucket_min0['data'], *bucket_min1['data'] ], "sse": None }
        buckets = [ *buckets[:min_sse['index']], merged, *buckets[min_sse['index']+2:] ]
        
    return rank_triangle_three_buckets(buckets, base_x)

