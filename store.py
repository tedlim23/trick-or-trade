import json
from bybit import BybitApi

class DBWriter():
    def __init__(self):
        # bybit initiate
        with open("./mybit.json", 'r') as f:
            apis = json.load(f)
            f.close()
        k = apis["KEY"]
        sr = k = apis["SECRET"]
        bybit = BybitApi(k, sr)
        ## db initiate

    def store_data(self):
        ## load last timestamp
        ## get new kline
        ## reformat
        ## put it in

    def reformat(self):
        ## reformat json data