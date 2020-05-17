#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[37]:


import pandas as pd
import numpy as np


# In[38]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[39]:


black_friday.head()


# In[40]:


black_friday.shape


# In[41]:


black_friday['Age'].unique()


# In[42]:


len(black_friday[ (black_friday['Gender'] == 'F') & (black_friday['Age'] == '26-35')])


# In[43]:


black_friday['User_ID'].nunique()


# In[44]:


np.unique(black_friday.dtypes.values).shape[0]


# In[45]:


black_friday.isna().sum().max()


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[46]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return black_friday.shape


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[47]:


def q2():
    resposta = len(black_friday[ (black_friday['Gender'] == 'F') & (black_friday['Age'] == '26-35')])
    return resposta


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[48]:


def q3():
    resposta = black_friday['User_ID'].nunique()
    return resposta


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[49]:


def q4():
    resposta = np.unique(black_friday.dtypes.values).shape[0]
    return resposta


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[50]:


def q5():
    return float(black_friday.isna().any(axis=1).sum() / black_friday.shape[0])


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[51]:


def q6():
    resposta = int(black_friday.isna().sum().max())
    return resposta


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[52]:


def q7():
    resposta = black_friday['Product_Category_3'].value_counts().index[0]
    return resposta


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[59]:


maximo = black_friday['Purchase'].max()
minimo = black_friday['Purchase'].min()
black_friday['Purchase_norm'] = black_friday['Purchase'].apply(lambda x: (x-minimo)/(maximo-minimo))


# In[60]:


def q8():
    resposta = float(black_friday['Purchase_norm'].mean())
    return resposta


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[63]:


media = black_friday['Purchase'].mean()
desvpad = black_friday['Purchase'].std()
black_friday['Purchase_zscore'] = black_friday['Purchase'].apply(lambda x: (x-media)/desvpad)


# In[56]:


def q9():
    return int(((black_friday['Purchase_zscore']>=-1) & (black_friday['Purchase_zscore']<=1)).sum())


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[57]:


def q10():
    black_friday[black_friday['Product_Category_2'].isna()]['Product_Category_3'].isna().value_counts()
    return True

