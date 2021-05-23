#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[5]:


# import libraries
import nltk

import pandas as pd
from sqlalchemy import create_engine, text
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfTransformer
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# In[6]:


# load data from database
engine = create_engine('sqlite:///Message.db')
df = pd.read_sql_table('Message', engine) 
df=df.dropna()
X=df[['message']]
Y=df.drop(['message','original','genre','id'],axis=1)


# In[7]:


Y.shape


# ### 2. Write a tokenization function to process your text data

# In[16]:


from nltk.stem.porter import PorterStemmer
def tokenize(text):
    """Normalize, tokenize and stem text string
    
    Args:
    text: string. String containing message for processing
       
    Returns:
    stemmed: list of strings. List containing normalized and stemmed word tokens
    """
    # Convert text to lowercase and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize words
    tokens = word_tokenize(text)
    
    # Stem word tokens and remove stop words
    stemmer = PorterStemmer()
    stop_words = stopwords.words("english")
    
    stemmed = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    return stemmed


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[17]:


from sklearn.multioutput import MultiOutputClassifier
pipeline = Pipeline([('vectorizer',CountVectorizer(tokenizer=tokenize)),
                     ('tfid', TfidfTransformer()),
                     ('clf',MultiOutputClassifier(estimator=DecisionTreeClassifier()))                    
     
                    ])

pipeline.get_params()


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y)
#print(X_train.shape,y_train.shape)
#X_train.shape[0] != y_train.shape[0]
#X_train.shape[1]==y_train.shape[1]
#y_train.shape

vectorizer=CountVectorizer()
tfid= TfidfTransformer()
clf=MultiOutputClassifier(estimator=DecisionTreeClassifier())
                      
    
X_new=vectorizer.fit_transform(X_train)
X_tfid=tfid.fit_transform(X_new)
#clf.fit(X_tfid,y_train)


# In[19]:


print(X_train.shape)
print(y_train.shape)
pipeline.fit(X_train,y_train)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[ ]:





# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[ ]:


parameters = 

cv = 


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[ ]:





# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[ ]:





# ### 9. Export your model as a pickle file

# In[ ]:





# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




