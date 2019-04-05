
# coding: utf-8

# In[ ]:


import pandas as pd
import scipy.stats as s
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


data=pd.read_csv("data.csv")


# In[ ]:


maindata=data.iloc[:,:]


# In[ ]:


bdata=maindata[maindata['diagnosis']=='B']
bdata.shape[0]


# In[ ]:


mdata=maindata[maindata['diagnosis']=='M']
mdata.shape[0]


# In[ ]:


bdatatraining=bdata.iloc[0:200,2:3]
bdatatraining


# In[ ]:


mdatatraining=mdata.iloc[0:200,2:3]


# In[ ]:


trainingdata=pd.concat([bdatatraining,mdatatraining],axis=0)


# In[ ]:


pcapb=bdatatraining.shape[0]/400
pcapm=mdatatraining.shape[0]/400


# In[ ]:


mcapb=bdatatraining['radius_mean'].mean()
mcapm=mdatatraining['radius_mean'].mean()


# In[ ]:


scapb=bdatatraining['radius_mean'].std()
scapm=mdatatraining['radius_mean'].std()


# In[ ]:


bdatatesting=bdata.iloc[200:,:]
mdatatesting=mdata.iloc[200:,:]
bdatatesting


# In[ ]:


#this function is for only one parameter that is mean_radius here
def Univariatenaivebayesclassifier(oneexample):
    #(p(stest(radius)/benign))=normal distribution pdf (gaussian formulae)
    pstesttumorisbenign=s.norm.pdf(oneexample,mcapb,scapb)
    #prior probability(P(A))=(p(benign))=pcapb
    num=pstesttumorisbenign*pcapb
    #P(~A)=(p(stest(radius)/malignant))
    pstesttumorismalignant=s.norm.pdf(oneexample,mcapm,scapm)
    #p(B)=p(benign)+p(malignant)
    den2=pstesttumorismalignant*pcapm
    pfinalbenign=num/(num+den2)
    return pfinalbenign


# In[ ]:


truepositivecount=0
falsepositivecount=0
truenegativecount=0
falsenegativecount=0

#benign- negative
#malignant- positive
#format-(right/wrong,benign(-)/malignant(+)count....bta rha h)
for j in range(0,len(bdatatesting)):
    pisbenign=Univariatenaivebayesclassifier(bdatatesting.iloc[j,2])
    if pisbenign > 0.5:
        #benign tha or benign hi bta rha h
        truenegativecount+=1
    else:
        #benign tha, galat bta rha h isliye false, and tumor malignant bta rha h
        falsepositivecount+=1
print(truenegativecount)   
print(falsepositivecount)
        


# In[ ]:


bdatatesting.shape[0]


# In[ ]:


for i in range(0,len(mdatatesting)):
    pisbenign=Univariatenaivebayesclassifier(mdatatesting.iloc[i,2])
    if pisbenign < 0.5:
        truepositivecount+=1#benign tha or benign hi bta rha h
    else:
        falsenegativecount+=1#benign tha, galat bta rha h isliye false, and tumor malignant bta rha h
print(truepositivecount) 
print(falsenegativecount)


# In[ ]:


precision=(truenegativecount+truepositivecount)/(bdatatesting.shape[0]+mdatatesting.shape[0])*100
precision


# In[ ]:


recall=truepositivecount/(mdatatesting.shape[0])*100
recall

