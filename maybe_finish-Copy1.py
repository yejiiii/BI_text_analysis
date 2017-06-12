
# coding: utf-8

# ### 데이터 받아오기

# In[1]:

import json
import numpy as np
import random
import nltk


# In[2]:

# for i in range(9):
#     i = "00"+str(i+1)
#     j = "Books_5_"+i+".json"
with open("Books_5.json","r") as data_file:
    raw_data=data_file.readlines()


# In[3]:

json_list = []
for i in range(len(raw_data)):
    try:
        json_data = json.loads(raw_data[i])
        json_list.append(json_data)
    except:
        print(raw_data[i],i)


# ### 데이터 분리하기

# In[4]:

review_data = []

for i in range(len(json_list)):
    review_dict={}
    review_dict["reviewText"] = json_list[i]["reviewText"]
    review_dict["overall"] = json_list[i]["overall"]
    review_data.append(review_dict)


# In[5]:

reviewText = []
for i in range(len(review_data)):
    reviewText.append(review_data[i]["reviewText"])


# In[6]:

reviewText[9]


# In[7]:

y_data = []
for i in range(len(review_data)):
    y_data.append(review_data[i]["overall"])


# In[8]:

from collections import Counter
Counter(y_data)


# <h3>데이터 전처리</h3>

# In[9]:

from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re


# In[ ]:

def review_to_words(raw_review):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    
   
    cleaned_text_list=[]
    for i in range(len(raw_review)):
        
        review_text = BeautifulSoup(raw_review[i]).get_text()
        letters_only = re.sub("[^a-zA-Z]", " ", review_text)
        words = letters_only.lower().split()
        cleaned_text = " ".join(words)
        cleaned_text_list.append(cleaned_text)

    return cleaned_text_list 


# In[ ]:

pre_data=review_to_words(reviewText)


# In[ ]:

print(len(reviewText), len(pre_data), len(y_data))


# ### 데이터 셔플

# In[ ]:

c = list(zip(pre_data, y_data))
random.shuffle(c)


# In[ ]:

a, b = zip(*c)


# In[ ]:

pre_text_data=list(a)
overall_data=np.array(list(b))


# In[ ]:

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer='word', sublinear_tf=True,lowercase=True)

tfidf = vectorizer.fit_transform(pre_text_data)


# ### 가장 작은 평점으로 갯수 맞추기

# In[ ]:

from collections import Counter
length=sorted(Counter(overall_data).most_common())[0][1]


# In[ ]:

# pre_data=[]
# overall_data=[]
# for j in range(len(reviewText)):
#     try:
#         if 50<=len(reviewText[j]) or len(reviewText[j])<=1500:
#             pre_data.append(reviewText[j])
#             overall_data.append(y_data[j])
#     except:
#         print(j)


# In[ ]:

search=[1.0,2.0,3.0,4.0,5.0]
count=0
text=[]
y=[]

for i in search:
    count=0
    for index,value in enumerate(overall_data):
        if (count==length): break
        elif (i==value):
            text.append(pre_text_data[index])
            y.append(value)
            count=count+1


# In[ ]:

Counter(y)


# In[ ]:

print(len(text), len(y))


# <h3>데이터 셔플 다시</h3>

# In[ ]:

c = list(zip(text, y))
random.shuffle(c)


# In[ ]:

a, b = zip(*c)


# In[ ]:

text_data=list(a)
overall=np.array(list(b))


# In[ ]:

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer='word', sublinear_tf=True,lowercase=True, stop_words='english', ngram_range=(1,2),
                            max_df = 0.2, min_df =2)


tfidf = vectorizer.fit_transform(text_data)


# In[ ]:

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(tfidf , overall , test_size=0.33, random_state=100)


# In[ ]:

x_train.shape, y_train.shape, x_test.shape, y_test.shape


# ### 예측

# ### LogisticRegression

# In[ ]:

from sklearn import linear_model
logreg = linear_model.LogisticRegression(C=0.8, random_state=500, solver='sag', multi_class='multinomial', warm_start=True)

logreg.fit(x_train, y_train)
logreg.predict(x_test)==y_test

# from sklearn import linear_model
# #logreg = linear_model.LogisticRegression()
# logreg = linear_model.LogisticRegression(random_state=42,solver='sag',multi_class='multinomial',warm_start=True)
# logreg.fit(x_train, y_train)


# In[ ]:

print(y_test[:10], logreg.predict(x_test)[:10])


# In[ ]:

from sklearn.metrics import accuracy_score
accuracy_score(logreg.predict(x_test), y_test)


# In[ ]:

from sklearn.metrics import precision_score
precision_score(logreg.predict(x_test), y_test,average=None) 


# In[ ]:

from sklearn import metrics
for i in range(1,6):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, logreg.predict(x_test), pos_label=i)
#pos_label Label considered as positive and others are considered negative.
    print(i,"점 = ",metrics.auc(fpr, tpr))


# In[ ]:

result=logreg.predict(x_test)


# In[ ]:

from sklearn.metrics import classification_report
print(classification_report(y_test, result))


# ### INPUT 

# In[ ]:

texts = ["""I enjoyed this book, as long as I kept in mind that the book was written for young adults.  
         The characters were believable and it was easy to identify with their pain.""",
        "I think the author gets lost in his own thoughts and trys to make storys of something that's not there.",
        """I was very excited to start reading this book and although I did like it, 
        I thought that a lot of the words to describe body parts was kind of high-schoolish.  
        Didn't care for the cliff hanging ending.""",
        """I don't know if you are familiar with FANFIC on the internet but that what this book reminded me of.  
        I love the series and thought I would like the books...not so much.""",
        """As I love western era romances, this one was wonderful.  An easy read and ended the way we always hope....happily!"""]
overall=[4,1,3,2,5]
vecs = vectorizer.transform(review_to_words(texts)) 


# In[ ]:

print(overall,logreg.predict(vecs), "\n", y_test[:20], logreg.predict(x_test)[:20])


# In[ ]:




# In[ ]:



