import time
import copy
from lxml import html
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta

class EODHistoricalData:
    def __init__(self, api_token, lookback):
        self.api_token = api_token
        self.lookback = lookback
        self.min_date = datetime.today() - timedelta(days=lookback)
        
    def get_news_data(self, stock):
        news = []
        all_dates = pd.date_range(self.min_date, datetime.today())

        nb_months = max(1, len(all_dates) // 30)
        len_month = len(all_dates) // nb_months

        for i in range(nb_months):
            temp_min_date = all_dates[i * len_month]
            temp_max_date = all_dates[min(len(all_dates) - 1, (i + 1) * len_month)]
            URL = 'https://eodhistoricaldata.com/api/news?api_token={}&s={}&from={}&to={}&offset=0&limit=1000'.format(self.api_token, stock, 
                                                                                                                      temp_min_date.strftime('%Y-%m-%d'), 
                                                                                                                      temp_max_date.strftime('%Y-%m-%d'))
            page = requests.get(URL)
            temp_news = json.loads(page.content)
            news = news + temp_news
        dates = [n['date'] for n in news]
        titles = [n['title'] for n in news]
        urls = [n['link'] for n in news]
        data = pd.DataFrame({'date': dates, 
                    'title': titles,
                    'url': urls})
        return data