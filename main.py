import re 
import tweepy 
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from tweepy import OAuthHandler 
from textblob import TextBlob

consumer_key = 'ZSRCFcNIZo5upxar07gR7pnAk'
consumer_secret = 'I79AcEsE6I0LCjdpd62sRkAxPxevzIPuD0xkrGjoGEU8EoUw0U'
access_token = '1423364971912564736-Ton4Gh5LWwstICX0KpEmtLH5toLLQY'
access_token_secret = '7umKAIcMhws4q6CaKsCJf95QVV5UZmoO1yoDcgLRcPbFD'

hashtag = "#metaverse"
tweetsPerQry = 100
maxTweets = 100000

authentication = tweepy.OAuthHandler(consumer_key, consumer_secret)
authentication.set_access_token(access_token, access_token_secret)
api = tweepy.API(authentication, wait_on_rate_limit=True)
maxId = -1
tweetCount = 0

result_path = 'metaverse'
num = 0
iteration = []
tweets_collected = []
id_info = []
time_posted_month = []
time_posted_day = []
time_posted_hour = []
time_posted_year = []
tweet_source = []
tweet_friends = []

while tweetCount < maxTweets:
    if(maxId <= 0):
        newTweets = api.search_tweets(q=hashtag, count=tweetsPerQry, result_type="recent", tweet_mode="extended")
    else:
        newTweets = api.search_tweets(q=hashtag, count=tweetsPerQry, max_id=str(maxId - 1), result_type="recent", tweet_mode="extended")

    if not newTweets:
        print("Tweet Habis")
        break
	
    for tweet in newTweets:
        print('------')
        print('num',num)
        iteration.append(num)
        num += 1

        print(tweet.full_text.encode('utf-8'))
        tweets_collected.append(tweet.full_text.encode('utf-8'))
        id_info.append(tweet.id)
        time_posted_month.append(tweet.user.created_at.month)
        time_posted_day.append(tweet.user.created_at.day)
        time_posted_hour.append(tweet.user.created_at.hour)
        time_posted_year.append(tweet.user.created_at.year)

        tweet_source.append(tweet.source)
        tweet_friends.append(tweet.user.friends_count)

        print('saving to excel')
        data_w = {'tweet count': iteration, 'id':id_info, 'tweet content': tweets_collected,
        'month':time_posted_month,'day':time_posted_day,'hour':time_posted_hour,'year':time_posted_year,
        'source':tweet_source, 'friends':tweet_friends}  
        my_csv = pd.DataFrame(data_w)
        name_save = result_path + '.csv' 
        my_csv.to_csv( name_save, index=False)
    
    
    
    
    
## ANALYSIS
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from datetime import datetime
  
def date(day,month,year):
    i = str(day)+'/'+str(month)+'/'+str(year)+' 00:00:00'
      
    return datetime.strptime(
      i, '%d/%m/%y %H:%M:%S')
    
    
    
############# Data analysis #############
# Read data
recent=[2020,2021,2022]
all_info = pd.read_csv('metaverse.csv')
all_info=all_info[all_info['year'].isin(recent) ]
all_info['date']=10000*all_info['year']+100*all_info['month']+all_info['day']
tweet_content = all_info['tweet content']
month_posted = all_info['month']
day_posted = all_info['day']
hour_posted = all_info['hour']
year_posted = all_info['year']
date_posted = all_info['date']
all_info['temp']= all_info['day'].apply(lambda x: str(x) if x>=10 else '0'+str(x))+ '/'+all_info['month'].apply(lambda x: str(x))+'/'+all_info['year'].apply(lambda x: str(x)[-2:])+' 00:00:00'

all_info['datetime'] = all_info['temp'].apply(lambda x: datetime.strptime(
      x, '%d/%m/%y %H:%M:%S'))

tweet_source = all_info['source']
tweet_friends = all_info['friends']

# Check the unique of the data 
tweet_content_u = tweet_content.unique().tolist()
month_posted_u = month_posted.unique().tolist()
day_posted_u = day_posted.unique().tolist()
date_posted_u = date_posted.unique().tolist()
hour_posted_u = hour_posted.unique().tolist()
tweet_source_u = tweet_source.unique().tolist()
tweet_friends_u = tweet_friends.unique().tolist()

# Histogram of week wise usage
plt.hist(week_posted, bins=148, edgecolor='white', linewidth=1.2)
plt.ylabel('Number of tweets')
plt.xlabel('Date')
plt.show()

# Histogram of monthly usage
plt.hist(month_posted, bins=12, edgecolor='white', linewidth=1.2)
plt.ylabel('Number of tweets')
plt.xlabel('Month')
plt.show()

# Histogram of daily usage
plt.hist(day_posted, bins=31, edgecolor='white', linewidth=1.2)
plt.ylabel('Number of tweets')
plt.xlabel('Day')
plt.show()

# Histogram of hourly usage
plt.hist(hour_posted, bins=24, edgecolor='white', linewidth=1.2)
plt.ylabel('Number of tweets')
plt.xlabel('Hour')
plt.show()



############# Tweet content analysis #############
def remove_content(text):
    text = re.sub(r"http\S+", "", text) #remove urls
    text=re.sub(r'\S+\.com\S+','',text) #remove urls
    text=re.sub(r'\@\w+','',text) #remove mentions
    text =re.sub(r'\#\w+','',text) #remove hashtags
    return text

def process_text(text, stop_words, stem=False): #clean text
    text=remove_content(text)
    text = re.sub('[^A-Za-z]', ' ', text.lower()) #remove non-alphabets
    tokenized_text = word_tokenize(text) #tokenize
    clean_text = [
         word for word in tokenized_text
         if word not in stop_words]
    if stem:
        clean_text=[stemmer.stem(word) for word in clean_text]
    return ' '.join(clean_text)

stopw = stopwords.words('english')
tmp_tweets = tweet_content.apply(lambda x: process_text(x,stopw))

filter_list = ['b','x','xa', 'rt', 'xe', 'xc', 'https','xb','xf','xbb','xef','de','xd','xac','ed','xce','xcf']
cleaned_tweets = []
for i in range(len(tmp_tweets)):
    tmp = tmp_tweets[i].split()
    tmp2 = [j for j in tmp if j not in filter_list]
    tmp2 = ' '.join(tmp2)
    cleaned_tweets.append(tmp2)

from wordcloud import WordCloud, STOPWORDS
wc_tweets = ' '.join(cleaned_tweets)
wordcloud = WordCloud(width = 800, height = 500, 
                background_color ='white', 
                min_font_size = 10).generate(wc_tweets)
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0) 
plt.show()

from sklearn.feature_extraction.text import CountVectorizer
def plot_ngram(sentences, ngram_range=(1,3), top=20,firstword=''):
    c=CountVectorizer(ngram_range=ngram_range)
    X=c.fit_transform(sentences)
    words=pd.DataFrame(X.sum(axis=0),columns=c.get_feature_names()).T.sort_values(0,ascending=False).reset_index()
    res=words[words['index'].apply(lambda x: firstword in x)].head(top)
    t1 = [i for i in res['index'] ]
    t2 = [i for i in res[0] ]
    plt.bar(t1,t2)
    plt.xticks(rotation=45)
    plt.ylabel('count')
    plt.show()

plot_ngram(cleaned_tweets, ngram_range=(1,1))
plot_ngram(cleaned_tweets, ngram_range=(2,2))

from textblob import TextBlob
sentiment = tweet_content.apply(lambda x:TextBlob(x).sentiment[0])
subject = tweet_content.apply(lambda x: TextBlob(x).sentiment[1])
polarity = sentiment.apply(lambda x: 'pos' if x>=0 else 'neg')
subject_2 = sentiment[subject>0.5]
polarity_withSubject = subject_2.apply(lambda x: 'pos' if x>=0 else 'neg')

fig, ax = plt.subplots()
N, bins, patches = ax.hist(polarity, bins=2, edgecolor='white', linewidth=1)
patches[0].set_facecolor('b')
patches[1].set_facecolor('r')
plt.ylabel('count')
plt.xlabel('polarity')
plt.show()

fig, ax = plt.subplots()
N, bins, patches = ax.hist(polarity_withSubject, bins=2, edgecolor='white', linewidth=1)
patches[0].set_facecolor('b')
patches[1].set_facecolor('r')
plt.ylabel('count')
plt.xlabel('subjectivity')
plt.show()

print('Well Done!')

import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
sb.distplot(week_posted,kde = False)
plt.show()



def CountFrequency(my_list):
     
   # Creating an empty dictionary
   count = {}
   for i in my_list:
    count[i] = count.get(i, 0) + 1
   return count

d=CountFrequency(week_posted)
"""
df=pd.DataFrame(d, index=range(len(d.keys()))).transpose()[0]
df.columns=['idx','week','#tweets']
plt.plot(d)

df=pd.DataFrame()
df['week_posted']=d.keys()
df['count']=df['week_posted'].apply(lambda x: d[x])

plt.plot(df)
"""


a=0
for i in d:
    a=max(a,d[i])

l=d.keys()
l=sorted(l)
y=[]
for k in l:
    print(k,": ",d[k])
    y.append(d[k])
    
j=0
for i in range(len(l)):
    if l[i]>20210000:
        j=i
        break

plt.plot(range(len(y)),y)

q=[]
j=0
qx=[]
for i in range(2019,2023):
    t2020=all_info[all_info['year']==i]
    t2020['week']=t2020['datetime'].apply(lambda x: x.week)
    sorted(t2020['week'].unique())
    a=range(53)
    a=[0 for i in a]
    
    for t in t2020['datetime']:
        a[t.week-1]+=1
    q+=a
    
    j+=1
    

plt.plot(l[:-35])

a=[]
tw=[]
for i in range(2019,2023):
    t2022=all_info[all_info['year']==i]
    t2022['week']=t2022['datetime'].apply(lambda x: x.week)
    c=range(53)
    c=[0 for i in c]
    bt=range(53)
    bt=[[] for i in bt]
    i=0
    for t in t2022['datetime']:
        c[t.week-1]+=1
        bt[t.week-1].append(t2022.iloc[i])
        i+=1
    tw+=bt
    a+=c
    
    
for i in range(len(tw)):
    tw[i]=pd.DataFrame(tw[i])
    
plt.plot(a)

from scipy.signal import find_peaks
plt.rcParams["figure.figsize"] = (15,4)
y=a
plt.plot(range(len(y)), y)

roots=[]
for i in range(1,len(y)):
    if y[i]-y[i-1]>=100 or y[i]-y[i-1]>=50 and y[i]-y[i-1]>=50:
        roots.append(i)

vals=range(len(y))



plt.plot(roots,[y[i] for i in roots], ls="", marker="o", label="points")

change in sentiment over last 2 and coming 2 days
most frequent ngrams [1,2] for current 2


#######################################
all_info = tw[0]
tweet_content = all_info['tweet content']
month_posted = all_info['month']
day_posted = all_info['day']
hour_posted = all_info['hour']
tweet_source = all_info['source']
tweet_friends = all_info['friends']

# Check the unique of the data 
tweet_content_u = tweet_content.unique().tolist()
month_posted_u = month_posted.unique().tolist()
day_posted_u = day_posted.unique().tolist()
hour_posted_u = hour_posted.unique().tolist()
tweet_source_u = tweet_source.unique().tolist()
tweet_friends_u = tweet_friends.unique().tolist()


############# Tweet content analysis #############
def remove_content(text):
    text = re.sub(r"http\S+", "", text) #remove urls
    text=re.sub(r'\S+\.com\S+','',text) #remove urls
    text=re.sub(r'\@\w+','',text) #remove mentions
    text =re.sub(r'\#\w+','',text) #remove hashtags
    return text

def process_text(text, stop_words, stem=False): #clean text
    text=remove_content(text)
    text = re.sub('[^A-Za-z]', ' ', text.lower()) #remove non-alphabets
    tokenized_text = word_tokenize(text) #tokenize
    clean_text = [
         word for word in tokenized_text
         if word not in stop_words]
    if stem:
        clean_text=[stemmer.stem(word) for word in clean_text]
    return ' '.join(clean_text)

stopw = stopwords.words('english')
tmp_tweets = tweet_content.apply(lambda x: process_text(x,stopw))

filter_list = ['b','x','xa', 'rt', 'xe', 'xc', 'https','xb','xf','xbb','xef','de','xd','xac','ed','xce','xcf']
cleaned_tweets = []
for i in range(len(tmp_tweets)):
    tmp = tmp_tweets.iloc[i].split()
    tmp2 = [j for j in tmp if j not in filter_list]
    tmp2 = ' '.join(tmp2)
    cleaned_tweets.append(tmp2)

from wordcloud import WordCloud, STOPWORDS
wc_tweets = ' '.join(cleaned_tweets)
wordcloud = WordCloud(width = 800, height = 500, 
                background_color ='white', 
                min_font_size = 10).generate(wc_tweets)
#plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0) 
plt.show()

from sklearn.feature_extraction.text import CountVectorizer
def plot_ngram(sentences, ngram_range=(1,3), top=20,firstword=''):
    c=CountVectorizer(ngram_range=ngram_range)
    X=c.fit_transform(sentences)
    words=pd.DataFrame(X.sum(axis=0),columns=c.get_feature_names()).T.sort_values(0,ascending=False).reset_index()
    res=words[words['index'].apply(lambda x: firstword in x)].head(top)
    t1 = [i for i in res['index'] ]
    t2 = [i for i in res[0] ]
    plt.figure(1)
    plt.bar(t1,t2)
    plt.xticks(rotation=45)
    plt.ylabel('count')
    plt.show()

plot_ngram(cleaned_tweets, ngram_range=(1,1))
plot_ngram(cleaned_tweets, ngram_range=(2,2))

from textblob import TextBlob
sentiment = tweet_content.apply(lambda x:TextBlob(x).sentiment[0])
subject = tweet_content.apply(lambda x: TextBlob(x).sentiment[1])
polarity = sentiment.apply(lambda x: 'pos' if x>=0 else 'neg')
subject_2 = sentiment[subject>0.5]
polarity_withSubject = subject_2.apply(lambda x: 'pos' if x>=0 else 'neg')

fig, ax = plt.subplots()
N, bins, patches = ax.hist(polarity, bins=2, edgecolor='white', linewidth=1)
patches[0].set_facecolor('b')
patches[1].set_facecolor('r')
plt.ylabel('count')
plt.xlabel('polarity')
plt.show()

fig, ax = plt.subplots()
N, bins, patches = ax.hist(polarity_withSubject, bins=2, edgecolor='white', linewidth=1)
patches[0].set_facecolor('b')
patches[1].set_facecolor('r')
plt.ylabel('count')
plt.xlabel('subjectivity')
plt.show()

print('Well Done!')



df = pd.DataFrame(
    columns=['idx', 'x', ''],
    index=['Jane', 'Melissa', 'John', 'Matt'])
print(df))

sentiments=[]
subjectivities1=[]
subjectivities2=[]
for y in range(2019, 2023):
    for w in range(0,54):
        f=all_info[all_info['year']==y]
        f=f[f['week']==w]
        if(len(f)>0):
            tweet_content=f['tweet content']
            sentiment = tweet_content.apply(lambda x:TextBlob(x).sentiment[0])
            subject = tweet_content.apply(lambda x: TextBlob(x).sentiment[1])
            polarity = sentiment.apply(lambda x: 'pos' if x>=0 else 'neg')
            sentiments.append(sentiment_ratio(polarity))
            
            subject_1 = sentiment[subject<=0.5]
            subject_2 = sentiment[subject>0.5]
            polarity_withSubject1 = subject_1.apply(lambda x: 'pos' if x>=0 else 'neg')
            polarity_withSubject2 = subject_2.apply(lambda x: 'pos' if x>=0 else 'neg')
            subjectivities1.append(sentiment_ratio(polarity_withSubject1))
            subjectivities2.append(sentiment_ratio(polarity_withSubject2))
        else:
            sentiments.append(0)
            subjectivities1.append(0)
            subjectivities1.append(0)

all_info['week']=all_info['datetime'].apply(lambda x: x.week)
len(polarity)/len(polarity[polarity['tweet content']='pos']])
        
def sentiment_ratio(df):
    x=0
    if len(df)>0:
        for i in range(len(df)):
            if df.iloc[i]=='pos':
                x+=1
        return x/len(df)
    else:
        return 0
    
for i in range(len(sentiments)):
    if sentiments[i]<0.1:
        print(i)

plt.figure(0)
t2=177
plt.plot(sentiments[:t2], label='Fraction of Positive sentiment tweets')

plt.plot(subjectivities1[:t2], label='sentiments, low subjectivity')

plt.plot(subjectivities2[:t2], label='sentiments, high subjectivity')
    
#plt.plot([i/max(l) for i in q[:t2]], label='volume')
plt.legend()
#don't touch l

qx=[]
for i in range(2019,2023):
    for w in range(1,54):
        qx.append("Y:"+str(i)+", W:"+str(w))

for i in range(52,163):
    if p[i]<0.2:
        p.pop(i)

qxx=[]        
qxx.append("Y:2019 W:0")
for i in range(len(qx[:177])):
    if i%20==0:
        qxx.append(qx[i])
        
plt.xticks(range(len(qxx)), qxx)
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(20))
plt.xlabel("Year: Week")
plt.ylabel("Fraction of positive tweets")
plt.plot(sentiments[1:177])


import numpy as np
import math
math.sqrt(np.var(q[:177]))/mean(q[:177])

for i in range(10,len(sentiments[:t2])):
    if sentiments[i]==0:
        sentiments[i]=(sentiments[i-2]+sentiments[i+2])/2
        print(i)

    
math.sqrt(np.var(sentiments[:177]))/mean(sentiments[:177])



volatile events: 0.3 change in subjective sentiment, high volume


plt.plot(subjectivities1[:t2], label='sentiments, low subjectivity')

plt.plot(subjectivities2[:t2], label='sentiments, high subjectivity')
plt.legend()
        
plt.xticks(range(len(qxx)), qxx)
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(20))
plt.xlabel("Year: Week")
plt.ylabel("Fraction of positive tweets")


x=0
for i in range(len(subjectivities1[:t2])):
    x+=abs(subjectivities1[i]-subjectivities2[i])
    
    
    
# n-gram analysis
    
# cleaning
def remove_content(text):
    text = re.sub(r"http\S+", "", text) #remove urls
    text=re.sub(r'\S+\.com\S+','',text) #remove urls
    text=re.sub(r'\@\w+','',text) #remove mentions
    text =re.sub(r'\#\w+','',text) #remove hashtags
    return text

def process_text(text, stop_words, stem=False): #clean text
    text=remove_content(text)
    text = re.sub('[^A-Za-z]', ' ', text.lower()) #remove non-alphabets
    tokenized_text = word_tokenize(text) #tokenize
    clean_text = [
         word for word in tokenized_text
         if word not in stop_words]
    if stem:
        clean_text=[stemmer.stem(word) for word in clean_text]
    return ' '.join(clean_text)

stopw = stopwords.words('english')
tmp_tweets = tweet_content.apply(lambda x: process_text(x,stopw))

filter_list = ['b','x','xa', 'rt', 'xe', 'xc', 'https','xb','xf','xbb','xef','de','xd','xac','ed','xce','xcf']
cleaned_tweets = []
for i in range(len(tmp_tweets)):
    tmp = tmp_tweets.iloc[i].split()
    tmp2 = [j for j in tmp if j not in filter_list]
    tmp2 = ' '.join(tmp2)
    cleaned_tweets.append(tmp2)


sentiments=[]
subjectivities1=[]
subjectivities2=[]
for y in range(2019, 2023):
    for w in range(0,54):
        f=all_info[all_info['year']==y]
        f=f[f['week']==w]
        if(len(f)>0):
            tweet_content=f['tweet content']
            sentiment = tweet_content.apply(lambda x:TextBlob(x).sentiment[0])
            subject = tweet_content.apply(lambda x: TextBlob(x).sentiment[1])
            polarity = sentiment.apply(lambda x: 'pos' if x>=0 else 'neg')
            sentiments.append(sentiment_ratio(polarity))
            
            subject_1 = sentiment[subject<=0.5]
            subject_2 = sentiment[subject>0.5]
            polarity_withSubject1 = subject_1.apply(lambda x: 'pos' if x>=0 else 'neg')
            polarity_withSubject2 = subject_2.apply(lambda x: 'pos' if x>=0 else 'neg')
            subjectivities1.append(sentiment_ratio(polarity_withSubject1))
            subjectivities2.append(sentiment_ratio(polarity_withSubject2))
            
            tmp_tweets = tweet_content.apply(lambda x: process_text(x,stopw))
            cleaned_tweets = []
            for i in range(len(tmp_tweets)):
                tmp = tmp_tweets.iloc[i].split()
                tmp2 = [j for j in tmp if j not in filter_list]
                tmp2 = ' '.join(tmp2)
                cleaned_tweets.append(tmp2)
                
            
        else:
            sentiments.append(0)
            subjectivities1.append(0)
            subjectivities1.append(0)

plt.figure(0)            
plot_ngram(cleaned_tweets, ngram_range=(1,1))
plt.figure(1)            
plot_ngram(cleaned_tweets, ngram_range=(2,2))

j=0
for i in all_info.columns:
    j+=1
    print(str(j)+". "+i)
