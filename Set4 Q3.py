#!/usr/bin/env python
# coding: utf-8

# In[3]:


#i)
from sklearn.datasets import load_digits
digits = load_digits()
import matplotlib.pyplot as plt
import numpy as np

#ii)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size=0.2)

#iii)
indexList=np.arange(len(x_train))
print(indexList)


# In[4]:


#iv)
x_labelled=x_train[indexList[:300]]
y_labelled=y_train[indexList[:300]]
nonLabelledIndices=indexList[300:]
y_train_nonlabel = np.copy(y_train)
y_train_nonlabel[nonLabelledIndices] = -1


# In[8]:


#v)
from sklearn.semi_supervised import LabelPropagation
lp=LabelPropagation(gamma=0)
lp.fit(x_train, y_train_nonlabel)
#vi)
lp.score(x_test,y_test)


# In[9]:


#vii)
from sklearn.semi_supervised import LabelSpreading
ls=LabelSpreading(gamma=0.4)
ls.fit(x_train, y_train_nonlabel)
#viii)
ls.score(x_test,y_test)


# In[10]:


#ix)
y_labelled=lp.transduction_[nonLabelledIndices]


# In[11]:


#x)
from sklearn.metrics import confusion_matrix
conf = confusion_matrix(y_train[nonLabelledIndices], y_labelled, labels=lp.classes_)
conf


# In[ ]:




