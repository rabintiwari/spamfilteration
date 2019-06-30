
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


emails=pd.read_csv("D:\Rabin\Videos\spam_or_not_spam.csv")
emails.head()


# In[3]:


X=emails["email"]
y=emails["label"]
len(X)


# In[4]:


from sklearn.model_selection import train_test_split


# In[5]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[6]:


a=len(X_train)
print(a)
b=len(y_train)
print(b)
c=len(X_test)
print(c)
d=len(y_test)
print(d)


# In[7]:


from sklearn.naive_bayes import MultinomialNB


# In[8]:


model=GaussianNB()


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


cv=CountVectorizer()


# In[ ]:


X_traincv=cv.fit_transform(X_train.values.astype('U'))


# In[ ]:


rabin=X_traincv.toarray()
rabin


# In[ ]:


a=cv.get_feature_names()
a


# In[ ]:


len(rabin[0])


# In[ ]:


cv.inverse_transform(rabin[0])


# In[ ]:


X_train.iloc[0]


# In[10]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[11]:


cv=TfidfVectorizer(stop_words="english")


# In[12]:


X_traincv=cv.fit_transform(X_train.values.astype('U'))
#X_testcv=cv.transform(X_test.values.astype('U'))
X_testcv=cv.transform(["you are selected for the grand prize of $1000"])


# In[13]:


X_traincv.toarray()


# In[14]:


len(X_traincv.toarray())


# In[15]:


a=cv.get_feature_names()
a


# In[16]:


len(a)


# In[17]:


cv.inverse_transform(X_traincv.toarray())


# In[9]:


model=MultinomialNB()


# In[18]:


model.fit(X_traincv,y_train)


# In[20]:


pre=model.predict(X_testcv)
pre


# In[21]:


pre


# In[22]:


model


# In[23]:


actual=np.array(y_test)
actual
    


# In[24]:


count = 0
for i in range(len(pre)):
    if pre[i]==actual[i]:
        count+=1


# In[25]:


count

