from typing import List, Tuple
import uvicorn
from bson import ObjectId, errors
from fastapi import Depends, FastAPI, HTTPException, Query, status
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
import json

app = FastAPI()
motor_client = AsyncIOMotorClient(
    "mongodb://localhost:27017"
)  # Connection to the whole server

if __name__=="__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 8800)