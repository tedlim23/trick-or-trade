import json
from bybit import BybitApi
from pymongo import MongoClient
from configparser import ConfigParser

def set_testdb():
    karg = {
        "host": '127.0.0.1', 
        "port": 27017, 
    }
    client = MongoClient(**karg)
    return client

class DBWriter():
    def __init__(self, test=False):
        # bybit initiate
        with open("./mybit.json", 'r') as f:
            apis = json.load(f)
            f.close()
        k = apis["KEY"]
        sr = k = apis["SECRET"]
        self.bybit = BybitApi(k, sr)
        ## db initiate
        self.client = None
        if test:
            self.client = set_testdb()
            self.db = self.client.mongo0
            self.collection = self.db.mongo01
        else:
            self.set_mongodb()
            self.db = self.client.trick_or_trade
            self.collection = self.db.klines

    def set_mongodb(self):
        config = ConfigParser()
        config.read('setting/initial_setting.ini')
        server = config['SERVER']

        # mongo db setting
        karg = {
            "host": server['IP'], 
            "port": int(server['PORT']), 
            "username": server['ID'], 
            "password": server['PW'], 
            "authSource":server['PROJECT']
        }

        self.client = MongoClient(**karg)

    def store_data(self):
        ## load last timestamp
        query = {"type": "kline"}
        documents = self.collection.find(query)
        doc_len = self.collection.count_documents(query)
        last_doc = documents.skip(doc_len-1)[0]
        last_timestamp = last_doc["start_at"]

        ## get new kline
        new_kline = self.bybit.get_kline(last_timestamp)

        ## reformat
        new_kline = self.reformat(new_kline)

        ## put it in
        x = self.collection.insert_many(new_kline)
        

    def reformat(self, data):
        ## reformat json data

        for d in data:
            d["type"] = "kline"
        
        return data


if __name__=="__main__":
    dbwriter = DBWriter(test=True)
    ## get new kline
    new_kline = dbwriter.bybit.get_kline_sample()
    ## reformat
    new_kline = dbwriter.reformat(new_kline)
    ## put it in
    x = dbwriter.collection.insert_many(new_kline)