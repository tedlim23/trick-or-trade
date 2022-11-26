from typing import Optional
from fastapi import FastAPI
import bybit
import time
import datetime
from datetime import datetime
from pymongo import MongoClient
import json

##################         BYBIT        ######################
app = FastAPI()
api_key='키입력'
api_secret='키입력'
secFor6Month = 60 * 60 * 24 * 30

#############        DB        ######################
info = bybit.BybitApi(api_key, api_secret)
dbServer = MongoClient("mongodb://localhost:27017/")
dbInfo = dbServer['trading']
dbDocu = dbInfo['trading']


#uvicorn main:app --reload
# http://127.0.0.1:8000/docs

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/kline/{symbol}") #https://bybit-exchange.github.io/docs/futuresV2/linear/#t-querykline
def get_kline_info(symbol: str, interval: Optional[str] = "1", limit: Optional[str]="1", from_time:Optional[str]="2022-01-01 00:00:00" ):
    convertTimeStamp = str(int(time.mktime(datetime.strptime(from_time, '%Y-%m-%d %H:%M:%S').timetuple())))
    result = info.get_kline_info(symbol="BTCUSDT", interval=interval, limit=limit, from_time=convertTimeStamp)["result"]
    return result

@app.get("/kline/db_save/{symbol}")
def set_kline_db_save(symbol: str, interval: Optional[str] = "1", from_time:Optional[str]="2022-01-01 00:00:00" ):
    convertTimeStamp = str(int(time.mktime(datetime.strptime(from_time, '%Y-%m-%d %H:%M:%S').timetuple())))

    if int(time.time()) - int(convertTimeStamp) > secFor6Month:
        tempTimeStamp = int(convertTimeStamp)
        for item in range(int(secFor6Month/(200*60))):
            time.sleep(3)  #ref site : https://bybit-exchange.github.io/docs/futuresV2/linear/#t-ipratelimits
            result = info.get_kline_info(symbol="BTCUSDT", interval=interval, limit="200", from_time=convertTimeStamp)["result"]
            dbDocu.insert_many(result)
            tempTimeStamp += (200 * 60)
    else:
        intervalTimeStamp = int(time.time()) - int(convertTimeStamp)
        tempTimeStamp = int(convertTimeStamp)
        for item in range(int(intervalTimeStamp/(200*60))):
            time.sleep(3)
            result = info.get_kline_info(symbol="BTCUSDT", interval=interval, limit="200", from_time=str(tempTimeStamp))["result"]
            dbDocu.insert_many(result)
            tempTimeStamp += (200 * 60)

        if intervalTimeStamp % (200*60) != 0:
            time.sleep(3)
            result = info.get_kline_info(symbol="BTCUSDT", interval=interval, limit="200", from_time=str(tempTimeStamp))["result"]
            dbDocu.insert_many(result)

    return {result:"success"}

@app.get("/kline/db_read/{symbol}")
def get_kline_db_read(symbol: str):
    result = dbInfo['trading'].find({},{'_id':False})
    # return len(list(result))
    return list(result)[0:5]