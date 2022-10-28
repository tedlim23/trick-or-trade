from datetime import datetime
from math import ceil

def get_last_1min():
    dt = datetime.now()  # Get timezone naive now
    seconds = dt.timestamp()
    return int(ceil(seconds-60))

def date_to_timestamp(date):
    dt = datetime.strptime(date, "%d/%m/%y")  # Get timezone naive now
    seconds = dt.timestamp()
    return int(ceil(seconds))
