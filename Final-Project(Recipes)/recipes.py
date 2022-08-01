#!/usr/bin/env python
# coding: utf-8

# In[2]:


# from google.colab import drive
# drive.mount('/content/drive')
# fpath='/content/drive/MyDrive/recipes/'


# In[3]:


get_ipython().system('pip install tomotopy')


# In[2]:


import matplotlib.pyplot as plt
import wordcloud as wc
import pandas as pd
import spacy
import tomotopy as tp
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.phrases import Phrases, Phraser
ldamodel = tp.LDAModel
spacy.cli.download("en_core_web_md")
nlp = spacy.load('en_core_web_md')


# In[3]:


# frec = open(fpath+'RecipeNLG_dataset.csv')
# rec=frec.readlines()
number_of_documents=10000
# receitas=pd.read_csv(fpath+'RecipeNLG_dataset.csv',nrows=number_of_documents)
receitas=pd.read_csv('RecipeNLG_dataset.csv',nrows=number_of_documents)


# In[4]:


# for i in range(len(rec['title'])):
#   if 'bible' in str(rec['title'][i]).lower():
#     print(rec['title'][i])
#     print(rec['ingredients'][i])
#     print(rec['directions'][i])
#     print()
#     print()
receitas.head()


# In[7]:


receitas.columns


# In[8]:


receitas.drop(['Unnamed: 0','link','source','NER'],axis=1,inplace=True)
receitas.head()


# In[9]:


receitas.info()


# In[10]:


docs=[]
for i in range(number_of_documents):
  aux=""
  aux+=receitas['title'][i].lower().replace("("," ").replace(")"," ")+" "
  aux+=receitas['ingredients'][i].lower().replace("("," ").replace(")"," ").replace("[", "").replace("]", "").replace("\"", "").replace("tsp","tea_spoon").replace("tbsp","soup_spoon").replace("c.","cup").replace("oz","ounce").replace("lb","pound").replace("pkg","package")+" "
  aux+=receitas['directions'][i].lower().replace("("," ").replace(")"," ").replace("[", "").replace("]", "").replace("\"", "")
  docs.append(aux)


# In[11]:


docslemma=[]
len_raw=[]
print('Building lemmas...')
for i,d in enumerate(docs):
  print(i,end='')
  len_raw.append(len(d))
  tdoc=nlp(d)
  lm=" ".join([token.lemma_ for token in tdoc  if not(token.is_stop == True or token.is_digit == True or token.is_punct == True or '\\' in token.lemma_ or '/' in token.lemma_)])
  docslemma.append(lm)
  print('\r\r\r\r\r\r\r\r',end='')
len_lemma=[len(d) for d in docslemma]
print('# of characters (raw,pre): (%d,%d)'%(sum(len_raw),sum(len_lemma)))
print('Average # of characters (raw,pre): (%.2f,%.2f)'%(sum(len_raw)/len(len_raw),sum(len_lemma)/len(len_lemma)))


# In[12]:


for i in docslemma[:10]:
  print(i)


# In[13]:


k=[]
for i in docslemma:
  k.append(i.split())
lk=len(k)
shortest=9999
longest =0
average =0
for i in k:
  test=len(i)
  if test > longest:
    longest = test
  if test < shortest:
    shortest = test
  average+=test
average/=lk
print("""number of documents: {}
shortest doc: {}
longest doc : {}
average doc : {}
""".format(lk,shortest,longest,average))


# In[14]:


for i in k[:10]:
  print(i)


# In[15]:


plt.figure(figsize=(15,15))

st=""
for i in k:
  for j in i:
    st+=j+" "
mycloud = wc.WordCloud().generate(st)
plt.imshow(mycloud)


# In[16]:


sts=st.split()
distinct=set(sts)
print("number of words: {}\nnumber of unique words: {}".format(len(sts),len(distinct)))


# In[17]:


dtoken=[simple_preprocess(d, deacc= True, min_len=3) for d in docslemma]
phrases  = Phrases(dtoken, min_count = 2,threshold=9)
bigram=Phraser(phrases)
bdocs=[bigram[d] for d in dtoken]
bdc=[]
for i in bdocs:
  st=""
  for j in i:
    st+=j+" "
  bdc.append(st)


# In[18]:


from sklearn.feature_extraction.text import TfidfVectorizer
vect=TfidfVectorizer()
colTFIDF=vect.fit_transform(bdc)
print(colTFIDF.toarray())
print(vect.get_feature_names_out())


# In[19]:


col_tokenized=bdocs
dict=corpora.Dictionary()
BoW=[dict.doc2bow(doc, allow_update=True) for doc in col_tokenized]
print(BoW[:20])
[print([(dict[id], count) for id, count in line]) for line in BoW[:20]]


# In[20]:


plt.figure(figsize=(15,15))

st=""
for i in bdocs:
  for j in i:
    st+=j+" "
mycloud = wc.WordCloud().generate(st)
plt.imshow(mycloud)

