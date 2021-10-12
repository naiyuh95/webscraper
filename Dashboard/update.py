import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pickle
import time
import requests
from selenium import webdriver
from bs4 import BeautifulSoup,NavigableString  
from datetime import datetime, timedelta
import glob, os
import xlrd
from urllib3.exceptions import ProtocolError
import time
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import shutil
from pathlib import Path
import gensim
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
import pyLDAvis
import pyLDAvis.gensim

pd.options.mode.chained_assignment = None  # default='warn'

#Set Absolute path
path = Path(__file__).parent.absolute()
path = str(path)
path = path.replace("\\", "/")
path = path + "/"

#Helper functions
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s Done:%s/%s' % (prefix, bar, percent, suffix,iteration,total), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def get(url, retries=5):
    try:
        r = requests.get(url)
        return r
    except ConnectionResetError as err:
        print("error is ... ")
        print(err)
        if retries < 1:
            raise ConnectionResetError('No more retries!')
        time.sleep(8)
        return get(url, retries - 1)

#generate sentiment score
sid = SentimentIntensityAnalyzer()
def get_sentiment(sentence):
    return sid.polarity_scores(sentence)

    
def driver_get(driver,url, retries=5):
    try:
        r = driver.get(url)
        return r
    # except ConnectionResetError as err:
        # print("error is ... ")
        # print(err)
        # if retries < 1:
            # raise ConnectionResetError('No more retries!')
        # driver.close()
        # time.sleep(8)
        # driver = webdriver.Chrome("D:/Desktop/capstone/UI/chromedriver")
        
        return get(url, retries - 1)
    except ProtocolError as err:
        print("error is ... ")
        print(err)
        if retries < 1:
            raise ConnectionResetError('No more retries!')
        driver.close()
        time.sleep(8)
        driver = webdriver.Chrome(path+"chromedriver")
        return get(url, retries - 1)
        
        

counter = 0
#number of times to loop

print('Fetching Content From Reuters')
link= [] 
dates= []
titles= []
timestr = time.strftime("%Y-%m-%dT%H:%M:%S")



links = []

linkss = 'https://www.reuters.com/assets/searchArticleLoadMoreJson?blob=finance&bigOrSmall=big&articleWithBlog=true&sortBy=date&dateRange=pastWeek&numResultsToShow=250&pn=&callback=addMoreNewsResults'
print('Obtaining links for the past 2 days...')




r = requests.get(linkss)
sentence = str(r.content)

x = sentence
counter = 0


#fetch links
while x.find('headline') != -1 : 

    title = x.find('headline')
    if x.find(',\r\n      ') != -1 :
        continue
  
    frontquote = x.find('"',title)
    string = x.find(',',title)
    backquote = x.find('"',frontquote+1)
    title = x[frontquote+1:backquote]
    title = title.replace('mln','million')
    title = title.replace('Mln','million')
    title = title.replace('bln','billion')
    title = title.replace('Bln','billion')
    title = title.replace('<b>financing<\\/b>',' ')
    title = title.replace('<b>finance<\\/b>',' ')
    title = title.replace('eur','euros')
    title = title.replace(':',' -')
    if title.find('UPDATE') != -1 or title.find('FACTBOX') != -1 or title.find('ANALYSIS') != -1 or title.find('BRIEF') != -1 or title.find('EXCLUSIVE') != -1 :
        dash = title.find('-')
        title = title[dash+1:]

    titles.append(title)

    date = x.find('date:')
    frontquote = x.find('"',date)
    string = x.find('\\',date)
    backquote = x.find('"',frontquote+1)

    dates.append(x[frontquote+1:backquote-4])

    ref = x.find('href')
    frontquote = x.find('"',ref)
    string = x.find(',',ref)
    backquote = x.find('"',frontquote+1)

    link.append(x[frontquote+1:backquote])
    x = x[string:]

for item in link:
    links.append('reuters.com' + item)


df = pd.DataFrame({'Headline':titles,'Date':dates,'Links':links})



#removed sponsored links
df = df[df['Links'].str.find('sponsored')==-1]
url = df['Links'].tolist()
df['Date'] = pd.to_datetime(df['Date'])

#filters links to 2 day range
df = df[df['Date']>(datetime.now() - timedelta(days=2,hours = 12))]

df = df.groupby('Headline', as_index=False).max()

options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--ignore-ssl-errors')
options.add_experimental_option('excludeSwitches', ['enable-logging'])


driver = webdriver.Chrome(path+"chromedriver", options=options)

dates_1 = []
title_1 = []
author_1 = []
contents_1 = []
url = []

print('Scraping all links...')
links = df['Links']



for i in range(0,links.shape[0]):
    
    driver.get("https://" + links.iloc[i])

    
    content = driver.page_source
    soup = BeautifulSoup(content, "html.parser")


    
    article_date = (soup.find('div', attrs={'class':'ArticleHeader_date'}).text).split('/')[0][:-1].split(' ')
    article_date = article_date[1][:-1]+ ' ' + article_date[0] + ' ' + article_date[2]
    article_date = datetime.strptime(article_date,'%d %B %Y').date()



    dates_1.append(article_date)


    title_1.append(soup.find('div', attrs={'class':'ArticleHeader_content-container'}).h1.text)
    #author_1.append(soup.find('div', attrs={'class':'BylineBar_byline'}).text)

    #for article only
    article = soup.find('div', attrs={'class':'StandardArticleBody_body'})
    resultset = article.find_all("p")
    fr = [element for result in resultset for element in result if isinstance(element, NavigableString)]
    spanset = [e.text for e in soup.find_all("span",{"itemprop":True})]
    setA = ["".join(z) for z in zip(fr,spanset)]
    final = setA + fr[len(spanset):]
    contents_1.append("".join(final[0:len(final)-1]))
    url.append("https://" + links.iloc[i])    
    
    printProgressBar(i+1, links.shape[0], prefix = 'Progress:', suffix = 'Complete', length = 50)

    time.sleep(5)    

print('Done')
print('Updating Scraping Data ...')


#pd.DataFrame({'Headline':title,'Date':date,'Author':author,'Content':contents}).to_csv('x.csv', index=False, encoding='utf-8')
#print(pd.DataFrame({'Headline':title_1,'Date':date_1,'Author':author_1,'Content':contents_1,'URL':url}))
final_scrape_data = pd.DataFrame({'Headline':title_1,'Date':dates_1,'Content':contents_1})

lists = []
for index, row in final_scrape_data.iterrows():
    lists.append(get_sentiment(row['Content']))
    
final_scrape_data['scores'] = lists
compound_scores = []
for i in range(0,final_scrape_data['scores'].shape[0]):
    compound_scores.append(final_scrape_data['scores'].iloc[i]['compound'])

final_scrape_data['compound_scores'] = compound_scores


final_scrape_data['Date'] = pd.to_datetime(final_scrape_data['Date'],format =  '%Y-%m-%d')
final_scrape_data["Date"] = final_scrape_data["Date"].dt.strftime( '%d/%m/%Y')


timestr = time.strftime("%Y-%m-%d %H-%M-%S")
initial_scrape_data = pd.read_csv(path+'data/Scrape_Data/scored_data.csv', index_col=False)

#create backup
shutil.copy(path+'data/Scrape_Data/scored_data.csv',path+'data/Backup_Data/Scrape_Data(Backup)/scored_data('+timestr+').csv')


initial_scrape_data = pd.concat([initial_scrape_data,final_scrape_data]) 
initial_scrape_data = initial_scrape_data.reset_index(drop=True)
initial_scrape_data = initial_scrape_data.drop_duplicates(['Content'])
initial_scrape_data = initial_scrape_data.drop_duplicates(['Headline'])

initial_scrape_data.to_csv(path+'data/Scrape_Data/scored_data.csv', index=False, date_format= '%d/%m/%Y')






print('Done')