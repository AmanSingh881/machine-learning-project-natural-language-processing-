import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#data cleaning and preprocessing
messages = pd.read_csv('SMSSpamCollection',sep='\t',names=["label","message"])
lemmatize = WordNetLemmatizer()
corpus=[]
for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]'," ",messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [lemmatize.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review = " ".join(review)
    corpus.append(review)

cv = CountVectorizer(max_features=3000)
X = cv.fit_transform(corpus).toarray()
y = pd.get_dummies(messages['label'])
y= y.iloc[:,1].values
#train test split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,y, test_size= 0.20, random_state = 0)

#traing data using NAive bayes classifier
spam_detect_model = MultinomialNB().fit(Xtrain,Ytrain)
y_pred = spam_detect_model.predict(Xtest)

confusion_m=confusion_matrix(Ytest,y_pred)

accuracy = accuracy_score(Ytest,y_pred)
print('accuracy of model',accuracy*100,"%")
