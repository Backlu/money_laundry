# coding: utf-8

import os
import json
import requests
from functools import wraps
import time
import logging

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        t = time.time()-ts
        logging.info(f'func:{f.__name__}: {t:.2f} sec')
        return result
    return wrap

def send_inference_msg_to_slack(msg):
    s_url = 'https://hooks.slack.com/services/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'
    dict_headers = {'Content-type': 'application/json'}
    dict_payload = {"text": msg}
    json_payload = json.dumps(dict_payload)
    rtn = requests.post(s_url, data=json_payload, headers=dict_headers)