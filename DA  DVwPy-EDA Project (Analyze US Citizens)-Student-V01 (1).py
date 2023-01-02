#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <p style="text-align: center;"><img src="https://docs.google.com/uc?id=1lY0Uj5R04yMY3-ZppPWxqCr5pvBLYPnV" class="img-fluid" 
# alt="CLRSWY"></p>
# 
# ## <p style="background-color:#FDFEFE; font-family:newtimeroman; color:#9d4f8c; font-size:100%; text-align:center; border-radius:10px 10px;">WAY TO REINVENT YOURSELF</p>
# 
# ## <p style="background-color:#FDFEFE; font-family:newtimeroman; color:#060108; font-size:200%; text-align:center; border-radius:10px 10px;">Data Analysis & Visualization with Python</p>
# 
# ## <p style="background-color:#FDFEFE; font-family:newtimeroman; color:#060108; font-size:200%; text-align:center; border-radius:10px 10px;">Project Solution</p>
# 
# ![image.png](https://i.ibb.co/mT1GG7j/US-citizen.jpg)
# 
# ## <p style="background-color:#FDFEFE; font-family:newtimeroman; color:#060108; font-size:200%; text-align:center; border-radius:10px 10px;">Analysis of US Citizens by Income Levels</p>

# <a id="toc"></a>
# 
# ## <p style="background-color:#9d4f8c; font-family:newtimeroman; color:#FFF9ED; font-size:175%; text-align:center; border-radius:10px 10px;">Content</p>
# 
# * [Introduction](#0)
# * [Dataset Info](#1)
# * [Importing Related Libraries](#2)
# * [Recognizing & Understanding Data](#3)
# * [Univariate & Multivariate Analysis](#4)    
# * [Other Specific Analysis Questions](#5)
# * [Dropping Similar & Unneccessary Features](#6)
# * [Handling with Missing Values](#7)
# * [Handling with Outliers](#8)    
# * [Final Step to make ready dataset for ML Models](#9)
# * [The End of the Project](#10)

# ## <p style="background-color:#9d4f8c; font-family:newtimeroman; color:#FFF9ED; font-size:175%; text-align:center; border-radius:10px 10px;">Introduction</p>
# 
# <a id="0"></a>
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>
# 
# One of the most important components to any data science experiment that doesn’t get as much importance as it should is **``Exploratory Data Analysis (EDA)``**. In short, EDA is **``"A first look at the data"``**. It is a critical step in analyzing the data from an experiment. It is used to understand and summarize the content of the dataset to ensure that the features which we feed to our machine learning algorithms are refined and we get valid, correctly interpreted results.
# In general, looking at a column of numbers or a whole spreadsheet and determining the important characteristics of the data can be very tedious and boring. Moreover, it is good practice to understand the problem statement and the data before you get your hands dirty, which in view, helps to gain a lot of insights. I will try to explain the concept using the Adult dataset/Census Income dataset available on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Adult). The problem statement here is to predict whether the income exceeds 50k a year or not based on the census data.
# 
# # Aim of the Project
# 
# Applying Exploratory Data Analysis (EDA) and preparing the data to implement the Machine Learning Algorithms;
# 1. Analyzing the characteristics of individuals according to income groups
# 2. Preparing data to create a model that will predict the income levels of people according to their characteristics (So the "salary" feature is the target feature)

# ## <p style="background-color:#9d4f8c; font-family:newtimeroman; color:#FFF9ED; font-size:175%; text-align:center; border-radius:10px 10px;">Dataset Info</p>
# 
# <a id="1"></a>
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>
# 
# The Census Income dataset has 32561 entries. Each entry contains the following information about an individual:
# 
# - **salary (target feature/label):** whether or not an individual makes more than $50,000 annually. (<= 50K, >50K)
# - **age:** the age of an individual. (Integer greater than 0)
# - **workclass:** a general term to represent the employment status of an individual. (Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked)
# - **fnlwgt:** this is the number of people the census believes the entry represents. People with similar demographic characteristics should have similar weights.  There is one important caveat to remember about this statement. That is that since the CPS sample is actually a collection of 51 state samples, each with its own probability of selection, the statement only applies within state.(Integer greater than 0)
# - **education:** the highest level of education achieved by an individual. (Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.)
# - **education-num:** the highest level of education achieved in numerical form. (Integer greater than 0)
# - **marital-status:** marital status of an individual. Married-civ-spouse corresponds to a civilian spouse while Married-AF-spouse is a spouse in the Armed Forces. Married-spouse-absent includes married people living apart because either the husband or wife was employed and living at a considerable distance from home (Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse)
# - **occupation:** the general type of occupation of an individual. (Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces)
# - **relationship:** represents what this individual is relative to others. For example an individual could be a Husband. Each entry only has one relationship attribute. (Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried)
# - **race:** Descriptions of an individual’s race. (White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black)
# - **sex:** the biological sex of the individual. (Male, female)
# - **capital-gain:** capital gains for an individual. (Integer greater than or equal to 0)
# - **capital-loss:** capital loss for an individual. (Integer greater than or equal to 0)
# - **hours-per-week:** the hours an individual has reported to work per week. (continuous)
# - **native-country:** country of origin for an individual (United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands)

# ## <p style="background-color:#9d4f8c; font-family:newtimeroman; color:#FFF9ED; font-size:175%; text-align:center; border-radius:10px 10px;">How to Installing/Enabling Intellisense or Autocomplete in Jupyter Notebook</p>
# 
# ### Installing [jupyter_contrib_nbextensions](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/install.html)
# 
# **To install the current version from The Python Package Index (PyPI), which is a repository of software for the Python programming language, simply type:**
# 
# !pip install jupyter_contrib_nbextensions
# 
# **Alternatively, you can install directly from the current master branch of the repository:**
# 
# !pip install https://github.com/ipython-contrib/jupyter_contrib_nbextensions/tarball/master
# 
# ### Enabling [Intellisense or Autocomplete in Jupyter Notebook](https://botbark.com/2019/12/18/how-to-enable-intellisense-or-autocomplete-in-jupyter-notebook/)
# 
# 
# ### Installing hinterland for jupyter without anaconda
# 
# **``STEP 1:``** ``Open cmd prompt and run the following commands``
#  1) pip install jupyter_contrib_nbextensions<br>
#  2) pip install jupyter_nbextensions_configurator<br>
#  3) jupyter contrib nbextension install --user<br> 
#  4) jupyter nbextensions_configurator enable --user<br>
# 
# **``STEP 2:``** ``Open jupyter notebook``
#  - click on nbextensions tab<br>
#  - unckeck disable configuration for nbextensions without explicit compatibility<br>
#  - put a check on Hinterland<br>
# 
# **``Step 3:``** ``Open new python file and check autocomplete feature``
# 
# [VIDEO SOURCE](https://www.youtube.com/watch?v=DKE8hED0fow)
# 
# ![Image_Assignment](https://i.ibb.co/RbmDmD6/E8-EED4-F3-B3-F4-4571-B6-A0-1-B3224-AAB060-4-5005-c.jpg)

# ## <p style="background-color:#9d4f8c; font-family:newtimeroman; color:#FFF9ED; font-size:175%; text-align:center; border-radius:10px 10px;">Importing Related Libraries</p>
# 
# <a id="2"></a>
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>
# 
# Once you've installed NumPy & Pandas you can import them as a library:

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")

plt.rcParams["figure.figsize"] = (10, 6)

sns.set_style("whitegrid")
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Set it None to display all rows in the dataframe
# pd.set_option('display.max_rows', None)

# Set it to None to display all columns in the dataframe
pd.set_option('display.max_columns', None)


# ### <p style="background-color:#9d4f8c; font-family:newtimeroman; color:#FFF9ED; font-size:150%; text-align:left; border-radius:10px 10px;">Reading the data from file</p>

# In[2]:


df0= pd.read_csv("adult_eda.csv")
df = df0.copy()


# In[3]:


df0.shape


# ## <p style="background-color:#9d4f8c; font-family:newtimeroman; color:#FFF9ED; font-size:175%; text-align:center; border-radius:10px 10px;">Recognizing and Understanding Data</p>
# 
# <a id="3"></a>
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>
# 
# ### 1.Try to understand what the data looks like
# - Check the head, shape, data-types of the features.
# - Check if there are some dublicate rows or not. If there are, then drop them. 
# - Check the statistical values of features.
# - If needed, rename the columns' names for easy use. 
# - Basically check the missing values.

# In[4]:


# Your Code is Here

df.head()


# Desired Output:
# 
# ![image.png](https://i.ibb.co/qFn8RZs/US-Citicens1.png)

# In[5]:


# Your Code is Here

df.shape

Desired Output:

(32561, 15)
# In[6]:


# Your Code is Here

df.info()

Desired Output:

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 32561 entries, 0 to 32560
Data columns (total 15 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   age             32561 non-null  int64  
 1   workclass       32561 non-null  object 
 2   fnlwgt          32561 non-null  int64  
 3   education       32561 non-null  object 
 4   education-num   31759 non-null  float64
 5   marital-status  32561 non-null  object 
 6   occupation      32561 non-null  object 
 7   relationship    27493 non-null  object 
 8   race            32561 non-null  object 
 9   sex             32561 non-null  object 
 10  capital-gain    32561 non-null  int64  
 11  capital-loss    32561 non-null  int64  
 12  hours-per-week  32561 non-null  int64  
 13  native-country  32561 non-null  object 
 14  salary          32561 non-null  object 
dtypes: float64(1), int64(5), object(9)
memory usage: 3.7+ MB
# In[7]:


# Check if the Dataset have any Duplicate

# Your Code is Here
df.duplicated().value_counts()

Desired Output:

False    32537
True        24
dtype: int64
# In[8]:


# Drop Duplicates

# Your Code is Here

df.drop_duplicates(inplace=True)


# In[9]:


# Check the shape of the Dataset

# Your Code is Here

df.shape

Desired Output:

(32537, 15)
# In[10]:


# Your Code is Here
df.describe().T


# Desired Output:
# 
# ![image.png](https://i.ibb.co/HnG6Xdn/US-Citicens2.png)

# **Rename the features of;**<br>
# **``"education-num"``**, **``"marital-status"``**, **``"capital-gain"``**, **``"capital-loss"``**, **``"hours-per-week"``**, **``"native-country"``** **as**<br>
# **``"education_num"``**, **``"marital_status"``**, **``"capital_gain"``**, **``"capital_loss"``**, **``"hours_per_week"``**, **``"native_country"``**, **respectively and permanently.**

# In[11]:


# Your Code is Here

df.rename(columns={"education-num" : "education_num",
                   "marital-status" : "marital_status",
                   "capital-gain" : "capital_gain",
                   "capital-loss": "capital_loss",
                   "hours-per-week" : "hours_per_week",
                   "native-country" : "native_country",
                   },
          inplace = True)


# In[12]:


# Check the sum of Missing Values per column

# Your Code is Here
df.isnull().sum()

age                  0
workclass            0
fnlwgt               0
education            0
education_num      802
marital_status       0
occupation           0
relationship      5064
race                 0
gender               0
capital_gain         0
capital_loss         0
hours_per_week       0
native_country       0
salary               0
dtype: int64
# In[13]:


# Check the Percentage of Missing Values

# Your Code is Here
df.isnull().sum() / df.shape[0]*100

Desired Output:

age               0.000
workclass         0.000
fnlwgt            0.000
education         0.000
education_num     2.465
marital_status    0.000
occupation        0.000
relationship     15.564
race              0.000
gender            0.000
capital_gain      0.000
capital_loss      0.000
hours_per_week    0.000
native_country    0.000
salary            0.000
dtype: float64
# ### 2.Look at the value counts of columns that have object datatype and detect strange values apart from the NaN Values

# In[14]:


# Your Code is Here
df.columns

Desired Output:

Index(['age', 'workclass', 'fnlwgt', 'education', 'education_num',
       'marital_status', 'occupation', 'relationship', 'race', 'gender',
       'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
       'salary'],
      dtype='object')
# In[15]:


# Your Code is Here
df.describe(include="object").T


# Desired Output:
# 
# ![image.png](https://i.ibb.co/WspBGfZ/US-Citicens3.png)

# **Assign the Columns (Features) of object data type as** **``"object_col"``**

# In[16]:


# Your Code is Here

object_col = df.select_dtypes(include="object").columns
object_col

Desired Output:

Index(['workclass', 'education', 'marital_status', 'occupation',
       'relationship', 'race', 'gender', 'native_country', 'salary'],
      dtype='object')
# In[17]:


for col in object_col:
    print(col)
    print("--"*8)
    print(df[col].value_counts(dropna=False))
    print("--"*20)


# **Check if the Dataset has any Question Mark** **``"?"``**

# In[18]:


# Your Code is Here

df[df.isin(['?'])].any()

Desired Output:

age               False
workclass          True
fnlwgt            False
education         False
education_num     False
marital_status    False
occupation         True
relationship      False
race              False
gender            False
capital_gain      False
capital_loss      False
hours_per_week    False
native_country     True
salary            False
dtype: bool
# ## <p style="background-color:#9d4f8c; font-family:newtimeroman; color:#FFF9ED; font-size:175%; text-align:center; border-radius:10px 10px;">Univariate & Multivariate Analysis</p>
# 
# <a id="4"></a>
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>
# 
# Examine all features (first target feature("salary"), then numeric ones, lastly categoric ones) separetly from different aspects according to target feature.
# 
# **to do list for numeric features:**
# 1. Check the boxplot to see extreme values 
# 2. Check the histplot/kdeplot to see distribution of feature
# 3. Check the statistical values
# 4. Check the boxplot and histplot/kdeplot by "salary" levels
# 5. Check the statistical values by "salary" levels
# 6. Write down the conclusions you draw from your analysis
# 
# **to do list for categoric features:**
# 1. Find the features which contains similar values, examine the similarities and analyze them together 
# 2. Check the count/percentage of person in each categories and visualize it with a suitable plot
# 3. If need, decrease the number of categories by combining similar categories
# 4. Check the count of person in each "salary" levels by categories and visualize it with a suitable plot
# 5. Check the percentage distribution of person in each "salary" levels by categories and visualize it with suitable plot
# 6. Check the count of person in each categories by "salary" levels and visualize it with a suitable plot
# 7. Check the percentage distribution of person in each categories by "salary" levels and visualize it with suitable plot
# 8. Write down the conclusions you draw from your analysis
# 
# **Note :** Instruction/direction for each feature is available under the corresponding feature in detail, as well.

# ## Salary (Target Feature)

# **Check the count of person in each "salary" levels and visualize it with a countplot**

# In[19]:


df.sample(5)


# In[20]:


# Your Code is Here
df.salary.value_counts(dropna=False)

Desired Output:

<=50K    24698
>50K      7839
Name: salary, dtype: int64
# In[21]:


# Your Code is Here
ax = sns.countplot(x="salary",data=df);
ax.bar_label(ax.containers[0], size = 13);


# Desired Output:
# 
# ![image.png](https://i.ibb.co/9qwrtB1/US-Citicens4.png)

# **Check the percentage of person in each "salary" levels and visualize it with a pieplot**

# In[22]:


# Your Code is Here
df.salary.value_counts(dropna=False) / df.salary.value_counts(dropna=False).sum()

Desired Output:

<=50K   0.759
>50K    0.241
Name: salary, dtype: float64
# In[23]:


# Your Code is Here
y = df.salary.value_counts(dropna=False) / df.salary.value_counts(dropna=False).sum()
mylabels = ["<=50K",">50K"]
myexplode = [0, 0.1]
colors_list = ["lightskyblue", "gold"]
plt.pie(y, labels=mylabels, labeldistance=1.1,
        colors=colors_list,explode=myexplode,
        shadow=True, autopct='%.1f%%');
plt.title("Percentage of Income-Levels", fontdict = {'fontsize': 14})


# Desired Output:
# 
# ![image.png](https://i.ibb.co/8YFvBrq/US-Citices5.png)

# **Write down the conclusions you draw from your analysis**

# **Result :** .................

# ## Numeric Features

# ## age

# **Check the boxplot to see extreme values**

# In[24]:


# Your Code is Here

sns.boxplot(x="age", data=df);


# Desired Output:
# 
# ![image.png](https://i.ibb.co/JKKwy5K/US-Citizens6.png)

# **Check the histplot/kdeplot to see distribution of feature**

# In[25]:


# Your Code is Here

sns.displot(x="age", data=df, kde=True,bins=20);


# Desired Output:
# 
# ![image.png](https://i.ibb.co/JcJ9cyp/US-Citizens7.png)

# **Check the statistical values**

# In[26]:


# Your Code is Here
df.describe()["age"]

Desired Output:

count   32537.000
mean       38.586
std        13.638
min        17.000
25%        28.000
50%        37.000
75%        48.000
max        90.000
Name: age, dtype: float64
# **Check the boxplot and histplot/kdeplot by "salary" levels**

# In[27]:


# Your Code is Here

sns.boxplot(x='salary',y='age',data=df);


# Desired Output:
# 
# ![image.png](https://i.ibb.co/64tBVNT/US-Citizens8.png)

# In[28]:


# Your Code is Here
sns.histplot(x="age", data=df,hue="salary",bins=20,kde=True);


# Desired Output:
# 
# ![image.png](https://i.ibb.co/q5P0sVf/US-Citizens9.png)

# In[29]:


# Your Code is Here

sns.kdeplot(x="age", data=df,hue="salary",fill=True);


# Desired Output:
# 
# ![image.png](https://i.ibb.co/7Y2HkxB/US-Citizens10.png)

# **Check the statistical values by "salary" levels**

# In[30]:


# Your Code is Here
df.groupby("salary")["age"].describe()


# Desired Output:
# 
# ![image.png](https://i.ibb.co/xYYZcZZ/US-Citizens11.png)

# **Write down the conclusions you draw from your analysis**

# **Result :** ................

# ## fnlwgt

# **Check the boxplot to see extreme values**

# In[31]:


# Your Code is Here

sns.boxplot(data=df, x="fnlwgt")


# Desired Output:
# 
# ![image.png](https://i.ibb.co/x2TtkzH/US-Citizens12.png)

# **Check the histplot/kdeplot to see distribution of feature**

# In[32]:


# Your Code is Here

sns.kdeplot(x="fnlwgt", data=df,fill=True);


# Desired Output:
# 
# ![image.png](https://i.ibb.co/ZmMV8nv/US-Citizens13.png)

# **Check the statistical values**

# In[33]:


# Your Code is Here

df.fnlwgt.describe()

Desired Output:

count     32537.000
mean     189780.849
std      105556.471
min       12285.000
25%      117827.000
50%      178356.000
75%      236993.000
max     1484705.000
Name: fnlwgt, dtype: float64
# **Check the boxplot and histplot/kdeplot by "salary" levels**

# In[34]:


# Your Code is Here

sns.boxplot(x="salary", y="fnlwgt",data=df);


# Desired Output:
# 
# ![image.png](https://i.ibb.co/ZxJS7JW/US-Citizens14.png)

# In[35]:


# Your Code is Here

sns.kdeplot(x="fnlwgt", data=df,hue="salary",fill=True);


# Desired Output:
# 
# ![image.png](https://i.ibb.co/TgygLrz/US-Citizens15.png)

# **Check the statistical values by "salary" levels**

# In[36]:


# Your Code is Here

df.groupby("salary")["fnlwgt"].describe()


# Desired Output:
# 
# ![image.png](https://i.ibb.co/LzWqdBf/US-Citizens16.png)

# **Write down the conclusions you draw from your analysis**

# **Result :** ...............

# ## capital_gain

# **Check the boxplot to see extreme values**

# In[37]:


# Your Code is Here

sns.boxplot(x="capital_gain", data=df)


# Desired Output:
# 
# ![image.png](https://i.ibb.co/6Xj1TCz/US-Citizens17.png)

# **Check the histplot/kdeplot to see distribution of feature**

# In[38]:


# Your Code is Here
sns.kdeplot(x="capital_gain",data=df,fill=True)


# Desired Output:
# 
# ![image.png](https://i.ibb.co/X3nW72Q/US-Citizens18.png)

# **Check the statistical values**

# In[39]:


# Your Code is Here

df.capital_gain.describe()

Desired Output:

count   32537.000
mean     1078.444
std      7387.957
min         0.000
25%         0.000
50%         0.000
75%         0.000
max     99999.000
Name: capital_gain, dtype: float64
# **Check the boxplot and histplot/kdeplot by "salary" levels**

# In[40]:


# Your Code is Here
sns.boxplot(x="salary", y="capital_gain",data=df);


# Desired Output:
# 
# ![image.png](https://i.ibb.co/CM3cTgt/19.png)

# In[41]:


# Your Code is Here

sns.kdeplot(x="capital_gain",data=df, hue="salary")


# Desired Output:
# 
# ![image.png](https://i.ibb.co/h7DKvLY/20.png)

# **Check the statistical values by "salary" levels**

# In[42]:


# Your Code is Here

df.groupby("salary")["capital_gain"].describe()


# Desired Output:
# 
# ![image.png](https://i.ibb.co/mzYxTD4/21.png)

# **Check the statistical values by "salary" levels for capital_gain not equal the zero**

# In[43]:


# Your Code is Here

df[df.capital_gain != 0].groupby("salary").capital_gain.describe()


# Desired Output:
# 
# ![image.png](https://i.ibb.co/r3mdBkK/22.png)

# **Write down the conclusions you draw from your analysis**

# **Result :** ...........................

# ## capital_loss

# **Check the boxplot to see extreme values**

# In[44]:


# Your Code is Here

sns.boxplot(data=df,x="capital_loss")


# Desired Output:
# 
# ![image.png](https://i.ibb.co/Db3XHKz/23.png)

# **Check the histplot/kdeplot to see distribution of feature**

# In[45]:


# Your Code is Here

sns.kdeplot(data=df,x="capital_loss",fill=True);


# Desired Output:
# 
# ![image.png](https://i.ibb.co/z7P15zX/24.png)

# **Check the statistical values**

# In[46]:


# Your Code is Here

df.capital_loss.describe()

Desired Output:

count   32537.000
mean       87.368
std       403.102
min         0.000
25%         0.000
50%         0.000
75%         0.000
max      4356.000
Name: capital_loss, dtype: float64
# **Check the boxplot and histplot/kdeplot by "salary" levels**

# In[47]:


# Your Code is Here

sns.boxplot(x="salary",y="capital_loss",data=df)


# Desired Output:
# 
# ![image.png](https://i.ibb.co/Dr7Bv9V/25.png)

# In[48]:


# Your Code is Here

sns.kdeplot(x="capital_loss",fill=True,hue="salary",data=df)


# Desired Output:
# 
# ![image.png](https://i.ibb.co/4Vg5Zyy/26.png)

# **Check the statistical values by "salary" levels**

# In[49]:


# Your Code is Here

df.groupby("salary")["capital_loss"].describe()


# Desired Output:
# 
# ![image.png](https://i.ibb.co/h9DTKNW/27.png)

# **Check the statistical values by "salary" levels for capital_loss not equel the zero**

# In[50]:


df[df.capital_loss != 0].groupby("salary").capital_gain.describe()


# In[51]:


# Your Code is Here
df[df.capital_loss != 0].groupby("salary").capital_loss.describe()


# Desired Output:
# 
# ![image.png](https://i.ibb.co/gJzQvmD/28.png)

# **Write down the conclusions you draw from your analysis**

# **Result :** ..................

# ## hours_per_week

# **Check the boxplot to see extreme values**

# In[52]:


# Your Code is Here

sns.boxplot(x="hours_per_week",data=df)


# Desired Output:
# 
# ![image.png](https://i.ibb.co/TkNCRYY/29.png)

# **Check the histplot/kdeplot to see distribution of feature**

# In[53]:


# Your Code is Here

sns.kdeplot(data=df,x="hours_per_week",fill=True);


# Desired Output:
# 
# ![image.png](https://i.ibb.co/tsp5GXb/30.png)

# **Check the statistical values**

# In[54]:


# Your Code is Here

df.hours_per_week.describe()

Desired Output:

count   32537.000
mean       40.440
std        12.347
min         1.000
25%        40.000
50%        40.000
75%        45.000
max        99.000
Name: hours_per_week, dtype: float64
# **Check the boxplot and histplot/kdeplot by "salary" levels**

# In[55]:


# Your Code is Here

sns.boxplot(data=df,x="salary",y="hours_per_week");


# Desired Output:
# 
# ![image.png](https://i.ibb.co/4RhSct7/31.png)

# In[56]:


# Your Code is Here

sns.kdeplot(data=df,x="hours_per_week",hue="salary",fill=True)


# Desired Output:
# 
# ![image.png](https://i.ibb.co/pbbVnMG/32.png)

# **Check the statistical values by "salary" levels**

# In[57]:


# Your Code is Here

df.groupby("salary")["hours_per_week"].describe()


# Desired Output:
# 
# ![image.png](https://i.ibb.co/6NbWfzz/33.png)

# **Write down the conclusions you draw from your analysis**

# **Result :** .....................

# ### See the relationship between each numeric features by target feature (salary) in one plot basically

# In[58]:


# Your Code is Here

sns.pairplot(df,hue="salary",corner=True,palette="viridis");


# Desired Output:
# 
# ![image.png](https://i.ibb.co/N7Fz4hg/34.png)

# ## Categorical Features

# ## education & education_num

# **Detect the similarities between these features by comparing unique values**

# In[59]:


# Your Code is Here
df.education.value_counts(dropna=False)

Desired Output:

HS-grad         10494
Some-college     7282
Bachelors        5353
Masters          1722
Assoc-voc        1382
11th             1175
Assoc-acdm       1067
10th              933
7th-8th           645
Prof-school       576
9th               514
12th              433
Doctorate         413
5th-6th           332
1st-4th           166
Preschool          50
Name: education, dtype: int64
# In[60]:


# Your Code is Here
df.education_num.value_counts(dropna=False)

Desired Output:

9.000     10208
10.000     7089
13.000     5245
14.000     1686
11.000     1343
7.000      1146
12.000     1044
6.000       916
NaN         802
4.000       630
15.000      559
5.000       503
8.000       424
16.000      405
3.000       329
2.000       159
1.000        49
Name: education_num, dtype: int64
# In[61]:


# Your Code is Here

df.groupby("education")["education_num"].value_counts(dropna=False)

Desired Output:

education     education_num
10th          6.000              916
              NaN                 17
11th          7.000             1146
              NaN                 29
12th          8.000              424
              NaN                  9
1st-4th       2.000              159
              NaN                  7
5th-6th       3.000              329
              NaN                  3
7th-8th       4.000              630
              NaN                 15
9th           5.000              503
              NaN                 11
Assoc-acdm    12.000            1044
              NaN                 23
Assoc-voc     11.000            1343
              NaN                 39
Bachelors     13.000            5245
              NaN                108
Doctorate     16.000             405
              NaN                  8
HS-grad       9.000            10208
              NaN                286
Masters       14.000            1686
              NaN                 36
Preschool     1.000               49
              NaN                  1
Prof-school   15.000             559
              NaN                 17
Some-college  10.000            7089
              NaN                193
Name: education_num, dtype: int64
# **Visualize the count of person in each categories for these features (education, education_num) separately**

# In[62]:


# Your Code is Here
sns.countplot(x = "education", data= df)
plt.xticks(rotation=90);


# Desired Output:
# 
# ![image.png](https://i.ibb.co/5xc31HR/35.png)

# In[63]:


# Your Code is Here
sns.countplot(x = 'education_num', data= df)


# Desired Output:
# 
# ![image.png](https://i.ibb.co/6HWtNN6/36.png)

# **Check the count of person in each "salary" levels by these features (education and education_num) separately and visualize them with countplot**

# In[64]:


# Your Code is Here

df.groupby("education")["salary"].value_counts()

Desired Output:

education     salary
10th          <=50K      871
              >50K        62
11th          <=50K     1115
              >50K        60
12th          <=50K      400
              >50K        33
1st-4th       <=50K      160
              >50K         6
5th-6th       <=50K      316
              >50K        16
7th-8th       <=50K      605
              >50K        40
9th           <=50K      487
              >50K        27
Assoc-acdm    <=50K      802
              >50K       265
Assoc-voc     <=50K     1021
              >50K       361
Bachelors     <=50K     3132
              >50K      2221
Doctorate     >50K       306
              <=50K      107
HS-grad       <=50K     8820
              >50K      1674
Masters       >50K       959
              <=50K      763
Preschool     <=50K       50
Prof-school   >50K       423
              <=50K      153
Some-college  <=50K     5896
              >50K      1386
Name: salary, dtype: int64
# In[65]:


# Your Code is Here

ax= sns.countplot(x="education", data=df,hue="salary");
ax.legend(loc=(0.5,0.9))
plt.xticks(rotation=90);


# Desired Output:
# 
# ![image.png](https://i.ibb.co/qxZXX1y/37.png)

# In[66]:


# Your Code is Here

df.groupby("education_num")["salary"].value_counts()

Desired Output:

education_num  salary
1.000          <=50K       49
2.000          <=50K      153
               >50K         6
3.000          <=50K      313
               >50K        16
4.000          <=50K      592
               >50K        38
5.000          <=50K      477
               >50K        26
6.000          <=50K      854
               >50K        62
7.000          <=50K     1088
               >50K        58
8.000          <=50K      391
               >50K        33
9.000          <=50K     8579
               >50K      1629
10.000         <=50K     5746
               >50K      1343
11.000         <=50K      994
               >50K       349
12.000         <=50K      787
               >50K       257
13.000         <=50K     3078
               >50K      2167
14.000         >50K       935
               <=50K      751
15.000         >50K       410
               <=50K      149
16.000         >50K       302
               <=50K      103
Name: salary, dtype: int64
# In[67]:


# Your Code is Here
sns.countplot(data=df,x="education_num",hue="salary");


# Desired Output:
# 
# ![image.png](https://i.ibb.co/2M0BYyk/38.png)

# **Visualize the boxplot of "education_num" feature by "salary" levels**

# In[68]:


# Your Code is Here

sns.boxplot(data=df,x="salary",y="education_num")


# Desired Output:
# 
# ![image.png](https://i.ibb.co/mSBNzKw/39.png)

# **Decrease the number of categories in "education" feature as low, medium, and high level and create a new feature with this new categorical data.**

# In[69]:


def mapping_education(x):
    if x in ["Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th"]:
        return "low_level_grade"
    elif x in ["HS-grad", "Some-college", "Assoc-voc", "Assoc-acdm"]:
        return "medium_level_grade"
    elif x in ["Bachelors", "Masters", "Prof-school", "Doctorate"]:
        return "high_level_grade"


# In[70]:


# Your Code is Here
df.education.apply(mapping_education).value_counts(dropna=False)

Desired Output:

medium_level_grade    20225
high_level_grade       8064
low_level_grade        4248
Name: education, dtype: int64
# In[71]:


# By using "mapping_education" def function above, create a new column named "education_summary"

# Your Code is Here

df["education_summary"] = df.education.apply(mapping_education)


# **Visualize the count of person in each categories for these new education levels (high, medium, low)**

# In[72]:


# Your Code is Here

sns.countplot(data=df,x="education_summary");


# Desired Output:
# 
# ![image.png](https://i.ibb.co/cx3Dzn1/40.png)

# **Check the count of person in each "salary" levels by these new education levels(high, medium, low) and visualize it with countplot**

# In[73]:


# Your Code is Here

df.groupby("education_summary")["salary"].value_counts()

Desired Output:

education_summary   salary
high_level_grade    <=50K      4155
                    >50K       3909
low_level_grade     <=50K      4004
                    >50K        244
medium_level_grade  <=50K     16539
                    >50K       3686
Name: salary, dtype: int64
# In[74]:


# Your Code is Here

sns.countplot(data=df,x="education_summary", hue="salary");


# Desired Output:
# 
# ![image.png](https://i.ibb.co/tXk04LJ/41.png)

# **Check the percentage distribution of person in each "salary" levels by each new education levels (high, medium, low) and visualize it with pie plot separately**

# In[75]:


edu =df.groupby("education_summary")["salary"].value_counts(normalize=True)
edu

Desired Output:

education_summary   salary
high_level_grade    <=50K    0.515
                    >50K     0.485
low_level_grade     <=50K    0.943
                    >50K     0.057
medium_level_grade  <=50K    0.818
                    >50K     0.182
Name: salary, dtype: float64
# In[76]:


plt.figure(figsize = (18, 6))
index = 1
for i in [0, 2, 4]:
    plt.subplot(1,3,index)
    edu[i:i+2].plot.pie(subplots=True,
                        labels=["<=50K", ">50K"],
                        autopct="%.2f%%",
                        textprops={'fontsize': 12},
                        colors=['pink', 'lightskyblue'],
                        )
    plt.title(edu.index[i][0], fontdict = {'fontsize': 14})
    index += 1


# Desired Output:
# 
# ![image.png](https://i.ibb.co/9W6kXc6/42.png)

# **Check the count of person in each these new education levels(high, medium, low) by "salary" levels and visualize it with countplot**

# In[77]:


# Your Code is Here

df.groupby("salary")["education_summary"].value_counts()

Desired Output:

salary  education_summary 
<=50K   medium_level_grade    16539
        high_level_grade       4155
        low_level_grade        4004
>50K    high_level_grade       3909
        medium_level_grade     3686
        low_level_grade         244
Name: education_summary, dtype: int64
# In[78]:


# Your Code is Here
sns.countplot(data=df,x="salary", hue="education_summary")


# Desired Output:
# 
# ![image.png](https://i.ibb.co/K9xLxvF/43.png)

# **Check the the percentage distribution of person in each these new education levels(high, medium, low) by "salary" levels and visualize it with pie plot separately**

# In[79]:


# Your Code is Here

educ = df.groupby("salary")["education_summary"].value_counts(normalize=True)
educ

Desired Output:

salary  education_summary 
<=50K   medium_level_grade   0.670
        high_level_grade     0.168
        low_level_grade      0.162
>50K    high_level_grade     0.499
        medium_level_grade   0.470
        low_level_grade      0.031
Name: education_summary, dtype: float64
# In[80]:


# Your Code is Here
plt.figure(figsize = (18, 6))
index = 1
for i in [0, 3]:
    plt.subplot(1,2,index)
    educ[i:i+3].plot.pie(subplots=True,
                        autopct="%.2f%%",
                        textprops={'fontsize': 12},
                        colors=['pink', 'lightskyblue', 'lightgreen'],
                        )
    plt.title(educ.index[i][0], fontdict = {'fontsize': 14})
    index += 1


# Desired Output:
# 
# ![image.png](https://i.ibb.co/42pnNPc/44.png)

# In[81]:


# Your Code is Here
edu_df = pd.DataFrame(educ)
edu_df.rename(columns = {"education_summary" : "percentage"}, inplace = True)
edu_df.reset_index(inplace = True)
edu_df.sort_values(by = ["salary", "education_summary"], inplace=True)
edu_df


# Desired Output:
# 
# ![image.png](https://i.ibb.co/jHYrhz8/45.png)

# In[82]:


# Your Code is Here
plt.figure(figsize = (18, 6))
index = 1
for i in [0, 3]:
    plt.subplot(1,2,index)
    edu_df["percentage"][i:i+3].plot.pie(subplots=True,
                                         labels=["high", "low", "medium"],
                                         autopct="%.2f%%",
                                         textprops={'fontsize': 12},
                                         colors=['pink', 'lightskyblue', 'lightgreen'],
                                         )
    plt.title(edu_df.salary[i], fontdict = {'fontsize': 14})
    index += 1


# Desired Output:
# 
# ![image.png](https://i.ibb.co/5BnYV6h/46.png)

# **Write down the conclusions you draw from your analysis**

# **Result :** ......................

# ## marital_status & relationship

# **Detect the similarities between these features by comparing unique values**

# In[83]:


# Your Code is Here

df.marital_status.value_counts(dropna=False)

Desired Output:

Married-civ-spouse       14970
Never-married            10667
Divorced                  4441
Separated                 1025
Widowed                    993
Married-spouse-absent      418
Married-AF-spouse           23
Name: marital_status, dtype: int64
# In[84]:


# Your Code is Here

df.relationship.value_counts(dropna=False)

Desired Output:

Husband           13187
Not-in-family      8292
NaN                5064
Unmarried          3445
Wife               1568
Other-relative      981
Name: relationship, dtype: int64
# In[85]:


# Fill missing values with "Unknown" in the column of "relationship"

# Your Code is Here
df.relationship.fillna("Unknown", inplace=True)


# In[86]:


# Your Code is Here
df.groupby("relationship")["marital_status"].value_counts(dropna=False)

Desired Output:

relationship    marital_status       
Husband         Married-civ-spouse       13178
                Married-AF-spouse            9
Not-in-family   Never-married             4694
                Divorced                  2403
                Widowed                    547
                Separated                  420
                Married-spouse-absent      211
                Married-civ-spouse          17
Other-relative  Never-married              611
                Married-civ-spouse         124
                Divorced                   110
                Separated                   55
                Widowed                     48
                Married-spouse-absent       32
                Married-AF-spouse            1
Unknown         Never-married             4481
                Divorced                   328
                Separated                   99
                Married-civ-spouse          95
                Married-spouse-absent       45
                Widowed                     15
                Married-AF-spouse            1
Unmarried       Divorced                  1600
                Never-married              881
                Separated                  451
                Widowed                    383
                Married-spouse-absent      130
Wife            Married-civ-spouse        1556
                Married-AF-spouse           12
Name: marital_status, dtype: int64
# **Assessment :** ........

# **Visualize the count of person in each categories**

# In[87]:


# Your Code is Here
sns.countplot(data=df,x="marital_status");


# Desired Output:
# 
# ![image.png](https://i.ibb.co/1RNHVvj/47.png)

# **Check the count of person in each "salary" levels by categories and visualize it with countplot**

# In[88]:


# Your Code is Here

df.groupby("marital_status")["salary"].value_counts(dropna=False)

Desired Output:

marital_status         salary
Divorced               <=50K      3978
                       >50K        463
Married-AF-spouse      <=50K        13
                       >50K         10
Married-civ-spouse     <=50K      8280
                       >50K       6690
Married-spouse-absent  <=50K       384
                       >50K         34
Never-married          <=50K     10176
                       >50K        491
Separated              <=50K       959
                       >50K         66
Widowed                <=50K       908
                       >50K         85
Name: salary, dtype: int64
# In[89]:


# Your Code is Here
sns.countplot(data=df,x="marital_status",hue="salary")
plt.xticks(rotation=45);


# Desired Output:
# 
# ![image.png](https://i.ibb.co/qjNhW9h/48.png)

# **Decrease the number of categories in "marital_status" feature as married, and unmarried and create a new feature with this new categorical data**

# In[90]:


def mapping_marital_status(x):
    if x in ["Never-married", "Divorced", "Separated", "Widowed"]:
        return "unmarried"
    elif x in ["Married-civ-spouse", "Married-AF-spouse", "Married-spouse-absent"]:
        return "married"


# In[91]:


# Your Code is Here
df.marital_status.apply(mapping_marital_status).value_counts(dropna=False)

Desired Output:

unmarried    17126
married      15411
Name: marital_status, dtype: int64
# In[92]:


# By using "mapping_marital_status" def function above, create a new column named "marital_status_summary"

# Your Code is Here

df["marital_status_summary"]= df.marital_status.apply(mapping_marital_status)


# **Visualize the count of person in each categories for these new marital status (married, unmarried)**

# In[93]:


# Your Code is Here
sns.countplot(data=df,x="marital_status_summary")


# Desired Output:
# 
# ![image.png](https://i.ibb.co/wRjj6Bx/49.png)

# **Check the count of person in each "salary" levels by these new marital status (married, unmarried) and visualize it with countplot**

# In[94]:


# Your Code is Here

df.groupby("marital_status_summary")["salary"].value_counts()

Desired Output:

marital_status_summary  salary
married                 <=50K      8677
                        >50K       6734
unmarried               <=50K     16021
                        >50K       1105
Name: salary, dtype: int64
# In[95]:


# Your Code is Here

sns.countplot(data=df,x="marital_status_summary",hue="salary")


# Desired Output:
# 
# ![image.png](https://i.ibb.co/0JtYnFb/50.png)

# **Check the percentage distribution of person in each "salary" levels by each new marital status (married, unmarried) and visualize it with pie plot separately**

# In[96]:


# Your Code is Here

marital =df.groupby("marital_status_summary")["salary"].value_counts(normalize=True)
marital

Desired Output:

marital_status_summary  salary
married                 <=50K    0.563
                        >50K     0.437
unmarried               <=50K    0.935
                        >50K     0.065
Name: salary, dtype: float64
# In[97]:


# Your Code is Here
plt.figure(figsize = (18, 6))
index = 1
for i in [0, 2]:
    plt.subplot(1,2,index)
    marital[i:i+2].plot.pie(subplots=True,
                            labels=["<=50K", ">50K"],
                            autopct="%.2f%%",
                            textprops={'fontsize': 12},
                            colors=['pink', 'lightskyblue'],
                            )
    plt.title(marital.index[i][0], fontdict = {'fontsize': 14})
    index += 1


# Desired Output:
# 
# ![image.png](https://i.ibb.co/TYxT5Zz/51.png)

# **Check the count of person in each these new marital status (married, unmarried) by "salary" levels and visualize it with countplot**

# In[98]:


# Your Code is Here
df.groupby("salary")["marital_status_summary"].value_counts()

Desired Output:

salary  marital_status_summary
<=50K   unmarried                 16021
        married                    8677
>50K    married                    6734
        unmarried                  1105
Name: marital_status_summary, dtype: int64
# In[99]:


# Your Code is Here

sns.countplot(data=df,x="salary",hue="marital_status_summary")


# Desired Output:
# 
# ![image.png](https://i.ibb.co/YWjjsZP/52.png)

# **Check the the percentage distribution of person in each these new marital status (married, unmarried) by "salary" levels and visualize it with pie plot separately**

# In[100]:


# Your Code is Here

mar_per = df.groupby("salary")["marital_status_summary"].value_counts(normalize=True)
mar_per

Desired Output:

salary  marital_status_summary
<=50K   unmarried                0.649
        married                  0.351
>50K    married                  0.859
        unmarried                0.141
Name: marital_status_summary, dtype: float64
# In[101]:


# Your Code is Here
marital_df = pd.DataFrame(mar_per)
marital_df.rename(columns = {"marital_status_summary" : "percentage"}, inplace = True)
marital_df.reset_index(inplace = True)
marital_df.sort_values(by = ["salary", "marital_status_summary"], inplace = True)
marital_df


# Desired Output:
# 
# ![image.png](https://i.ibb.co/Swb4rb7/v53.png)

# In[102]:


# Your Code is Here

plt.figure(figsize = (18, 6))
index = 1
for i in [0, 2]:
    plt.subplot(1,2,index)
    marital_df["percentage"][i:i+2].plot.pie(subplots=True,
                                             labels=["married", "unmarried"],
                                             autopct="%.2f%%",
                                             textprops={'fontsize': 12},
                                             colors=['pink', 'lightskyblue'],
                                             )
    plt.title(marital_df.salary[i], fontdict = {'fontsize': 14})
    index += 1


# Desired Output:
# 
# ![image.png](https://i.ibb.co/cJxmqwG/54.png)

# **Write down the conclusions you draw from your analysis**

# **Result :** .................

# ## workclass

# **Check the count of person in each categories and visualize it with countplot**

# In[103]:


# Your Code is Here

df.workclass.value_counts(dropna=False)

Desired Output:

Private             22673
Self-emp-not-inc     2540
Local-gov            2093
?                    1836
State-gov            1298
Self-emp-inc         1116
Federal-gov           960
Without-pay            14
Never-worked            7
Name: workclass, dtype: int64
# In[104]:


# Your Code is Here
sns.countplot(data=df, x="workclass")
plt.xticks(rotation=90);


# Desired Output:
# 
# ![image.png](https://i.ibb.co/NmKTp84/55.png)

# **Replace the value "?" to the value "Unknown"** 

# In[105]:


# Replace "?" values with "Unkown"

# Your Code is Here
df.workclass.replace("?", "Unknown", inplace=True)


# **Check the count of person in each "salary" levels by workclass groups and visualize it with countplot**

# In[106]:


# Your Code is Here
df.groupby("workclass")["salary"].value_counts(dropna=False)

Desired Output:

workclass         salary
Federal-gov       <=50K       589
                  >50K        371
Local-gov         <=50K      1476
                  >50K        617
Never-worked      <=50K         7
Private           <=50K     17712
                  >50K       4961
Self-emp-inc      >50K        622
                  <=50K       494
Self-emp-not-inc  <=50K      1816
                  >50K        724
State-gov         <=50K       945
                  >50K        353
Unknown           <=50K      1645
                  >50K        191
Without-pay       <=50K        14
Name: salary, dtype: int64
# In[107]:


# Your Code is Here

sns.countplot(data=df,x="workclass",hue="salary");
plt.xticks(rotation=90)
plt.legend(loc="upper left");


# Desired Output:
# 
# ![image.png](https://i.ibb.co/bPnNvsn/56.png)

# **Check the percentage distribution of person in each "salary" levels by each workclass groups and visualize it with bar plot**

# In[108]:


# Your Code is Here

workcls = df.groupby("workclass")["salary"].value_counts(normalize=True)
workcls

Desired Output:

workclass         salary
Federal-gov       <=50K    0.614
                  >50K     0.386
Local-gov         <=50K    0.705
                  >50K     0.295
Never-worked      <=50K    1.000
Private           <=50K    0.781
                  >50K     0.219
Self-emp-inc      >50K     0.557
                  <=50K    0.443
Self-emp-not-inc  <=50K    0.715
                  >50K     0.285
State-gov         <=50K    0.728
                  >50K     0.272
Unknown           <=50K    0.896
                  >50K     0.104
Without-pay       <=50K    1.000
Name: salary, dtype: float64
# In[109]:


# Your Code is Here
workclass_df = pd.DataFrame(workcls)
workclass_df.rename(columns = {"salary" : "percentage"}, inplace = True)
workclass_df.reset_index(inplace = True)
workclass_df.sort_values(by = ["workclass", "salary"], inplace=True)
workclass_df


# Desired Output:
# 
# ![image.png](https://i.ibb.co/8YvM14M/57.png)

# In[110]:


# Your Code is Here

fig, ax = plt.subplots()

ax = sns.barplot(data = workclass_df, x = "workclass", y = "percentage", hue = "salary")

plt.xticks(rotation=90);


# Desired Output:
# 
# ![image.png](https://i.ibb.co/NFN5q04/58.png)

# **Check the count of person in each workclass groups by "salary" levels and visualize it with countplot**

# In[111]:


# Your Code is Here
df.groupby("salary")["workclass"].value_counts(dropna=False)

Desired Output:

salary  workclass       
<=50K   Private             17712
        Self-emp-not-inc     1816
        Unknown              1645
        Local-gov            1476
        State-gov             945
        Federal-gov           589
        Self-emp-inc          494
        Without-pay            14
        Never-worked            7
>50K    Private              4961
        Self-emp-not-inc      724
        Self-emp-inc          622
        Local-gov             617
        Federal-gov           371
        State-gov             353
        Unknown               191
Name: workclass, dtype: int64
# In[112]:


# Your Code is Here

sns.countplot(data=df,x="salary",hue="workclass")
plt.legend(loc="upper center")


# Desired Output:
# 
# ![image.png](https://i.ibb.co/98V8zkN/59.png)

# **Check the the percentage distribution of person in each workclass groups by "salary" levels and visualize it with countplot**

# In[113]:


# Your Code is Here
workclass = df.groupby("salary")["workclass"].value_counts(normalize=True)
workclass

Desired Output:

salary  workclass       
<=50K   Private            0.717
        Self-emp-not-inc   0.074
        Unknown            0.067
        Local-gov          0.060
        State-gov          0.038
        Federal-gov        0.024
        Self-emp-inc       0.020
        Without-pay        0.001
        Never-worked       0.000
>50K    Private            0.633
        Self-emp-not-inc   0.092
        Self-emp-inc       0.079
        Local-gov          0.079
        Federal-gov        0.047
        State-gov          0.045
        Unknown            0.024
Name: workclass, dtype: float64
# In[114]:


# Your Code is Here
workclass_df = pd.DataFrame(workclass)
workclass_df.rename(columns = {"workclass" : "percentage"}, inplace = True)
workclass_df.reset_index(inplace = True)
workclass_df.sort_values(by = ["salary", "workclass"], inplace=True)
workclass_df


# Desired Output:
# 
# ![image.png](https://i.ibb.co/QcdnXpk/60.png)

# In[115]:


# Your Code is Here

fig, ax = plt.subplots(figsize=(14, 6))

ax = sns.barplot(data=workclass_df, x="salary", y="percentage", hue="workclass")

for container in ax.containers:
    ax.bar_label(container,fmt="%.2f");


# Desired Output:
# 
# ![image.png](https://i.ibb.co/Kz5BDBj/61.png)

# **Write down the conclusions you draw from your analysis**

# **Result :** ..................

# ## occupation

# **Check the count of person in each categories and visualize it with countplot**

# In[116]:


# Your Code is Here

df.occupation.value_counts(dropna=False)

Desired Output:

Prof-specialty       4136
Craft-repair         4094
Exec-managerial      4065
Adm-clerical         3768
Sales                3650
Other-service        3291
Machine-op-inspct    2000
?                    1843
Transport-moving     1597
Handlers-cleaners    1369
Farming-fishing       992
Tech-support          927
Protective-serv       649
Priv-house-serv       147
Armed-Forces            9
Name: occupation, dtype: int64
# In[117]:


# Your Code is Here
sns.countplot(data=df,x="occupation")
plt.xticks(rotation=90);


# Desired Output:
# 
# ![image.png](https://i.ibb.co/F3qqLjS/62.png)

# **Replace the value "?" to the value "Unknown"**

# In[118]:


# Replace "?" values with "Unknown"

# Your Code is Here

df.occupation.replace("?", "Unknow",inplace=True)


# **Check the count of person in each "salary" levels by occupation groups and visualize it with countplot**

# In[119]:


# Your Code is Here

df.groupby("occupation")["salary"].value_counts(dropna=False)

Desired Output:

occupation         salary
Adm-clerical       <=50K     3261
                   >50K       507
Armed-Forces       <=50K        8
                   >50K         1
Craft-repair       <=50K     3165
                   >50K       929
Exec-managerial    <=50K     2097
                   >50K      1968
Farming-fishing    <=50K      877
                   >50K       115
Handlers-cleaners  <=50K     1283
                   >50K        86
Machine-op-inspct  <=50K     1751
                   >50K       249
Other-service      <=50K     3154
                   >50K       137
Priv-house-serv    <=50K      146
                   >50K         1
Prof-specialty     <=50K     2278
                   >50K      1858
Protective-serv    <=50K      438
                   >50K       211
Sales              <=50K     2667
                   >50K       983
Tech-support       <=50K      644
                   >50K       283
Transport-moving   <=50K     1277
                   >50K       320
Unknown            <=50K     1652
                   >50K       191
Name: salary, dtype: int64
# In[120]:


# Your Code is Here

sns.countplot(data=df,x="occupation",hue="salary")
plt.xticks(rotation=90);


# Desired Output:
# 
# ![image.png](https://i.ibb.co/RhkhQCW/63.png)

# **Check the percentage distribution of person in each "salary" levels by each occupation groups and visualize it with bar plot**

# In[121]:


# Your Code is Here

occupation = df.groupby("occupation")["salary"].value_counts(normalize=True)
occupation

Desired Output:

occupation         salary
Adm-clerical       <=50K    0.865
                   >50K     0.135
Armed-Forces       <=50K    0.889
                   >50K     0.111
Craft-repair       <=50K    0.773
                   >50K     0.227
Exec-managerial    <=50K    0.516
                   >50K     0.484
Farming-fishing    <=50K    0.884
                   >50K     0.116
Handlers-cleaners  <=50K    0.937
                   >50K     0.063
Machine-op-inspct  <=50K    0.875
                   >50K     0.124
Other-service      <=50K    0.958
                   >50K     0.042
Priv-house-serv    <=50K    0.993
                   >50K     0.007
Prof-specialty     <=50K    0.551
                   >50K     0.449
Protective-serv    <=50K    0.675
                   >50K     0.325
Sales              <=50K    0.731
                   >50K     0.269
Tech-support       <=50K    0.695
                   >50K     0.305
Transport-moving   <=50K    0.800
                   >50K     0.200
Unknown            <=50K    0.896
                   >50K     0.104
Name: salary, dtype: float64
# In[122]:


# Your Code is Here

occupation_df = pd.DataFrame(occupation)
occupation_df.rename(columns = {"salary" : "percentage"}, inplace = True)
occupation_df.reset_index(inplace = True)
occupation_df.sort_values(by = ["occupation", "salary"], inplace=True)
occupation_df


# Desired Output:
# 
# ![image.png](https://i.ibb.co/mb7JS3n/64.png)

# In[123]:


# Your Code is Here

fig, ax = plt.subplots(figsize=(14, 6))

ax = sns.barplot(data = occupation_df, x = "occupation", y = "percentage", hue = "salary")

plt.xticks(rotation=90)

for container in ax.containers:
    ax.bar_label(container,fmt="%.2f");


# Desired Output:
# 
# ![image.png](https://i.ibb.co/sW2b8wL/65.png)

# **Check the count of person in each occupation groups by "salary" levels and visualize it with countplot**

# In[124]:


# Your Code is Here

df.groupby("salary")["occupation"].value_counts()

Desired Output:

salary  occupation       
<=50K   Adm-clerical         3261
        Craft-repair         3165
        Other-service        3154
        Sales                2667
        Prof-specialty       2278
        Exec-managerial      2097
        Machine-op-inspct    1751
        Unknown              1652
        Handlers-cleaners    1283
        Transport-moving     1277
        Farming-fishing       877
        Tech-support          644
        Protective-serv       438
        Priv-house-serv       146
        Armed-Forces            8
>50K    Exec-managerial      1968
        Prof-specialty       1858
        Sales                 983
        Craft-repair          929
        Adm-clerical          507
        Transport-moving      320
        Tech-support          283
        Machine-op-inspct     249
        Protective-serv       211
        Unknown               191
        Other-service         137
        Farming-fishing       115
        Handlers-cleaners      86
        Armed-Forces            1
        Priv-house-serv         1
Name: occupation, dtype: int64
# In[125]:


# Your Code is Here
sns.countplot(data=df, x="salary",hue="occupation")


# Desired Output:
# 
# ![image.png](https://i.ibb.co/cvHS3FH/66.png)

# **Check the the percentage distribution of person in each occupation groups by "salary" levels and visualize it with bar plot**

# In[126]:


# Your Code is Here

occupation  =df.groupby("salary")["occupation"].value_counts(normalize=True)
occupation

Desired Output:

salary  occupation       
<=50K   Adm-clerical        0.132
        Craft-repair        0.128
        Other-service       0.128
        Sales               0.108
        Prof-specialty      0.092
        Exec-managerial     0.085
        Machine-op-inspct   0.071
        Unknown             0.067
        Handlers-cleaners   0.052
        Transport-moving    0.052
        Farming-fishing     0.036
        Tech-support        0.026
        Protective-serv     0.018
        Priv-house-serv     0.006
        Armed-Forces        0.000
>50K    Exec-managerial     0.251
        Prof-specialty      0.237
        Sales               0.125
        Craft-repair        0.119
        Adm-clerical        0.065
        Transport-moving    0.041
        Tech-support        0.036
        Machine-op-inspct   0.032
        Protective-serv     0.027
        Unknown             0.024
        Other-service       0.017
        Farming-fishing     0.015
        Handlers-cleaners   0.011
        Armed-Forces        0.000
        Priv-house-serv     0.000
Name: occupation, dtype: float64
# In[127]:


# Your Code is Here
occupation_df = pd.DataFrame(occupation)
occupation_df.rename(columns = {"occupation" : "percentage"}, inplace = True)
occupation_df.reset_index(inplace = True)
occupation_df.sort_values(by = ["salary", "occupation"], inplace=True)
occupation_df


# Desired Output:
# 
# ![image.png](https://i.ibb.co/7tK0PqX/67.png)

# In[128]:


# Your Code is Here
fig, ax = plt.subplots(figsize=(20, 6))
ax= sns.barplot(data=occupation_df,x="salary",y="percentage",hue="occupation")

for container in ax.containers:
    ax.bar_label(container,fmt="%.2f");


# Desired Output:
# 
# ![image.png](https://i.ibb.co/7brj34F/68.png)

# **Write down the conclusions you draw from your analysis**

# **Result :** ................

# ## race

# **Check the count of person in each categories and visualize it with countplot**

# In[129]:


# Your Code is Here

df.race.value_counts(dropna=False)

Desired Output:

White                 27795
Black                  3122
Asian-Pac-Islander     1038
Amer-Indian-Eskimo      311
Other                   271
Name: race, dtype: int64
# In[130]:


# Your Code is Here

sns.countplot(data=df,x="race")


# Desired Output:
# 
# ![image.png](https://i.ibb.co/LdKct3G/69.png)

# **Check the count of person in each "salary" levels by races and visualize it with countplot**

# In[131]:


# Your Code is Here

df.groupby("race")["salary"].value_counts(dropna=False)

Desired Output:

race                salary
Amer-Indian-Eskimo  <=50K       275
                    >50K         36
Asian-Pac-Islander  <=50K       762
                    >50K        276
Black               <=50K      2735
                    >50K        387
Other               <=50K       246
                    >50K         25
White               <=50K     20680
                    >50K       7115
Name: salary, dtype: int64
# In[132]:


# Your Code is Here

sns.countplot(data=df,x="race",hue="salary");
plt.xticks(rotation=60);


# Desired Output:
# 
# ![image.png](https://i.ibb.co/Qb4n8Y5/70.png)

# **Check the percentage distribution of person in each "salary" levels by each races and visualize it with pie plot**

# In[133]:


# Your Code is Here

race = df.groupby("race")["salary"].value_counts(normalize=True)
race

Desired Output:

race                salary
Amer-Indian-Eskimo  <=50K    0.884
                    >50K     0.116
Asian-Pac-Islander  <=50K    0.734
                    >50K     0.266
Black               <=50K    0.876
                    >50K     0.124
Other               <=50K    0.908
                    >50K     0.092
White               <=50K    0.744
                    >50K     0.256
Name: salary, dtype: float64
# In[134]:


# Your Code is Here
plt.figure(figsize = (18, 12))
index = 1
for i in [0, 2, 4, 6, 8]:
    plt.subplot(2,3,index)
    race[i:i+2].plot.pie(subplots=True,
                         labels=["<=50K", ">50K"],
                         autopct="%.2f%%",
                         textprops={'fontsize': 12},
                         colors=['pink', 'lightskyblue'],
                         )
    plt.title(race.index[i][0], fontdict = {'fontsize': 14})
#    plt.legend()
    index += 1


# Desired Output:
# 
# ![image.png](https://i.ibb.co/xsJWXp4/71.png)

# **Check the count of person in each races by "salary" levels and visualize it with countplot**

# In[135]:


# Your Code is Here

df.groupby("salary")["race"].value_counts(dropna=False)

Desired Output:

salary  race              
<=50K   White                 20680
        Black                  2735
        Asian-Pac-Islander      762
        Amer-Indian-Eskimo      275
        Other                   246
>50K    White                  7115
        Black                   387
        Asian-Pac-Islander      276
        Amer-Indian-Eskimo       36
        Other                    25
Name: race, dtype: int64
# In[136]:


# Your Code is Here

sns.countplot(data=df,x="salary",hue="race");


# Desired Output:
# 
# ![image.png](https://i.ibb.co/RBpPR38/72.png)

# **Check the the percentage distribution of person in each races by "salary" levels and visualize it with bar plot**

# In[137]:


# Your Code is Here
race = df.groupby("salary")["race"].value_counts(normalize=True)
race

Desired Output:

salary  race              
<=50K   White                0.837
        Black                0.111
        Asian-Pac-Islander   0.031
        Amer-Indian-Eskimo   0.011
        Other                0.010
>50K    White                0.908
        Black                0.049
        Asian-Pac-Islander   0.035
        Amer-Indian-Eskimo   0.005
        Other                0.003
Name: race, dtype: float64
# In[138]:


# Your Code is Here
race_df = pd.DataFrame(race)
race_df.rename(columns = {"race" : "percentage"}, inplace = True)
race_df.reset_index(inplace = True)
race_df.sort_values(by = ["salary", "race"], inplace=True)
race_df


# Desired Output:
# 
# ![image.png](https://i.ibb.co/Xy9sYCY/73.png)

# In[139]:


# Your Code is Here

ax = sns.barplot(data=race_df,x="salary",y="percentage",hue="race")


# Desired Output:
# 
# ![image.png](https://i.ibb.co/X8kf9NZ/74.png)

# **Write down the conclusions you draw from your analysis**

# **Result :** ................

# ## gender

# **Check the count of person in each gender and visualize it with countplot**

# In[140]:


df.head()


# In[141]:


# Your Code is Here
df.sex.value_counts(dropna=False)

Desired Output:

Male      21775
Female    10762
Name: gender, dtype: int64
# In[142]:


# Your Code is Here

sns.countplot(data=df, x="sex");


# Desired Output:
# 
# ![image.png](https://i.ibb.co/GVTRbrb/75.png)

# **Check the count of person in each "salary" levels by gender and visualize it with countplot**

# In[143]:


# Your Code is Here

df.groupby("sex")["salary"].value_counts(dropna=False)

Desired Output:

gender  salary
Female  <=50K      9583
        >50K       1179
Male    <=50K     15115
        >50K       6660
Name: salary, dtype: int64
# In[144]:


# Your Code is Here

sns.countplot(data=df,x="sex",hue="salary");


# Desired Output:
# 
# ![image.png](https://i.ibb.co/Nr8HRPk/76.png)

# **Check the percentage distribution of person in each "salary" levels by each gender and visualize it with pie plot**

# In[145]:


# Your Code is Here

sex = df.groupby("sex")["salary"].value_counts(normalize=True)
sex

Desired Output:

gender  salary
Female  <=50K    0.890
        >50K     0.110
Male    <=50K    0.694
        >50K     0.306
Name: salary, dtype: float64
# In[146]:


# Your Code is Here
plt.figure(figsize = (18, 6))
index = 1
for i in [0, 2]:
    plt.subplot(1,2,index)
    sex[i:i+2].plot.pie(subplots=True,
                         labels=["<=50K", ">50K"],
                         autopct="%.2f%%",
                         textprops={'fontsize': 12},
                         colors=['pink', 'lightskyblue'],
                         )
    plt.title(sex.index[i][0], fontdict = {'fontsize': 14})

    index += 1


# Desired Output:
# 
# ![image.png](https://i.ibb.co/nrHj2jk/77.png)

# **Check the count of person in each gender by "salary" levels and visualize it with countplot**

# In[147]:


# Your Code is Here

df.groupby("salary")["sex"].value_counts(dropna=False)

Desired Output:

salary  gender
<=50K   Male      15115
        Female     9583
>50K    Male       6660
        Female     1179
Name: gender, dtype: int64
# In[148]:


# Your Code is Here

sns.countplot(data=df,x="salary",hue="sex");


# Desired Output:
# 
# ![image.png](https://i.ibb.co/9sfsw11/78.png)

# **Check the the percentage distribution of person in each gender by "salary" levels and visualize it with pie plot**

# In[149]:


# Your Code is Here

sex = df.groupby("salary")["sex"].value_counts(normalize=True)
sex

Desired Output:

salary  gender
<=50K   Male     0.612
        Female   0.388
>50K    Male     0.850
        Female   0.150
Name: gender, dtype: float64
# In[150]:


# Your Code is Here
index = 1
for i in [0,2]:
    plt.subplot(1,2,index)
    sex[i:i+2].plot.pie(subplots=True,
                         labels=["Male", "Female"],
                         autopct="%.2f%%",
                         textprops={'fontsize': 12},
                         colors=['pink', 'lightskyblue'],
                         )
    plt.title(sex.index[i][0], fontdict = {'fontsize': 14})
    
    index += 1


# Desired Output:
# 
# ![image.png](https://i.ibb.co/0DzhNgG/79.png)

# **Write down the conclusions you draw from your analysis**

# **Result :** ..............

# ## native_country

# **Check the count of person in each categories and visualize it with countplot**

# In[151]:


# Your Code is Here

df.native_country.value_counts(dropna=False)

Desired Output:

United-States                 29153
Mexico                          639
?                               582
Philippines                     198
Germany                         137
Canada                          121
Puerto-Rico                     114
El-Salvador                     106
India                           100
Cuba                             95
England                          90
Jamaica                          81
South                            80
China                            75
Italy                            73
Dominican-Republic               70
Vietnam                          67
Japan                            62
Guatemala                        62
Poland                           60
Columbia                         59
Taiwan                           51
Haiti                            44
Iran                             43
Portugal                         37
Nicaragua                        34
Peru                             31
France                           29
Greece                           29
Ecuador                          28
Ireland                          24
Hong                             20
Cambodia                         19
Trinadad&Tobago                  19
Laos                             18
Thailand                         18
Yugoslavia                       16
Outlying-US(Guam-USVI-etc)       14
Honduras                         13
Hungary                          13
Scotland                         12
Holand-Netherlands                1
Name: native_country, dtype: int64
# In[152]:


# Your Code is Here
sns.countplot(data=df, x="native_country")
plt.xticks(rotation=90);


# Desired Output:
# 
# ![image.png](https://i.ibb.co/x3TNT7B/80.png)

# **Replace the value "?" to the value "Unknown"** 

# In[153]:


# Replace "?" values with "Unknown"

# Your Code is Here
df.native_country.replace("?", "Unknown", inplace = True)


# **Decrease the number of categories in "native_country" feature as US, and Others and create a new feature with this new categorical data**

# In[154]:


def mapping_native_country(x):
    if x == "United-States":
        return "US"
    else:
        return "Others"


# In[155]:


# Your Code is Here

df.native_country.apply(mapping_native_country).value_counts(dropna=False)

Desired Output:

US        29153
Others     3384
Name: native_country, dtype: int64
# In[156]:


# By using "mapping_native_country" def function above, create a new column named "native_country_summary"

# Your Code is Here
df["native_country_summary"] = df.native_country.apply(mapping_native_country)
df["native_country_summary"]

Desired Output:

0            US
1            US
2            US
3            US
4        Others
          ...  
32556        US
32557        US
32558        US
32559        US
32560        US
Name: native_country_summary, Length: 32537, dtype: object
# **Visualize the count of person in each new categories (US, Others)**

# In[157]:


# Your Code is Here
sns.countplot(data=df, x="native_country_summary");


# Desired Output:
# 
# ![image.png](https://i.ibb.co/wwDhVGd/81.png)

# **Check the count of person in each "salary" levels by these new native countries (US, Others) and visualize it with countplot**

# In[158]:


# Your Code is Here

df.groupby("native_country_summary").salary.value_counts()

Desired Output:

native_country_summary  salary
Others                  <=50K      2714
                        >50K        670
US                      <=50K     21984
                        >50K       7169
Name: salary, dtype: int64
# In[159]:


# Your Code is Here

sns.countplot(data=df, x="native_country_summary", hue="salary");


# Desired Output:
# 
# ![image.png](https://i.ibb.co/SVnKp4k/82.png)

# **Check the percentage distribution of person in each "salary" levels by each new native countries (US, Others) and visualize it with pie plot separately**

# In[160]:


# Your Code is Here
country = df.groupby(["native_country_summary"]).salary.value_counts(normalize=True)
country

Desired Output:

native_country_summary  salary
Others                  <=50K    0.802
                        >50K     0.198
US                      <=50K    0.754
                        >50K     0.246
Name: salary, dtype: float64
# In[161]:


# Your Code is Here
plt.figure(figsize = (18, 6))
index = 1
for i in [0, 2]:
    plt.subplot(1,2,index)
    country[i:i+2].plot.pie(subplots=True,
                            labels=["<=50K", ">50K"],
                            autopct="%.2f%%",
                            textprops={'fontsize': 12},
                            colors=['pink', 'lightskyblue'],
                            )
    plt.title(country.index[i][0], fontdict = {'fontsize': 14})

    index += 1


# Desired Output:
# 
# ![image.png](https://i.ibb.co/4NQ5b1b/83.png)

# **Check the count of person in each these new native countries (US, Others) by "salary" levels and visualize it with countplot**

# In[162]:


# Your Code is Here

df.groupby("salary").native_country_summary.value_counts()

Desired Output:

salary  native_country_summary
<=50K   US                        21984
        Others                     2714
>50K    US                         7169
        Others                      670
Name: native_country_summary, dtype: int64
# In[163]:


# Your Code is Here


sns.countplot(data=df, x="salary", hue="native_country_summary");


# Desired Output:
# 
# ![image.png](https://i.ibb.co/c1gQfcg/84.png)

# **Check the the percentage distribution of person in each these new native countries (US, Others) by "salary" levels and visualize it with pie plot separately**

# In[164]:


# Your Code is Here
country = df.groupby(["salary"]).native_country_summary.value_counts(normalize=True)
country

Desired Output:

salary  native_country_summary
<=50K   US                       0.890
        Others                   0.110
>50K    US                       0.915
        Others                   0.085
Name: native_country_summary, dtype: float64
# In[165]:


# Your Code is Here
plt.figure(figsize = (18, 6))
index = 1
for i in [0, 2]:
    plt.subplot(1,2,index)
    country[i:i+2].plot.pie(subplots=True,
                            labels=["US", "Others"],
                            autopct="%.2f%%",
                            textprops={'fontsize': 12},
                            colors=['pink', 'lightskyblue'],
                            )
    plt.title(country.index[i][0], fontdict = {'fontsize': 14})

    index += 1


# Desired Output:
# 
# ![image.png](https://i.ibb.co/QHc8m0x/85.png)

# **Write down the conclusions you draw from your analysis**

# **Result :** .................

# ## <p style="background-color:#9d4f8c; font-family:newtimeroman; color:#FFF9ED; font-size:175%; text-align:center; border-radius:10px 10px;">Other Specific Analysis Questions</p>
# 
# <a id="5"></a>
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>

# ### 1. What is the average age of males and females by income level?

# In[166]:


# Your Code is Here

df.groupby(["salary", "sex"]).age.mean()

Desired Output:

salary  gender
<=50K   Female   36.217
        Male     37.149
>50K    Female   42.126
        Male     44.627
Name: age, dtype: float64
# In[167]:


# Your Code is Here

fig, ax = plt.subplots()

ax = df.groupby(["salary", "sex"]).age.mean().plot.bar()

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f");


# Desired Output:
# 
# ![image.png](https://i.ibb.co/BBDy081/86.png)

# In[168]:


# Your Code is Here
age = df.groupby(["salary", "sex"])[["age"]].mean().reset_index()
age


# Desired Output:
# 
# ![image.png](https://i.ibb.co/4PD1208/87.png)

# In[169]:


# Your Code is Here
fig, ax = plt.subplots()

ax = sns.barplot(data=age, x="salary", y="age", hue="sex")

for container in ax.containers:
    ax.bar_label(container,fmt="%.2f");


# Desired Output:
# 
# ![image.png](https://i.ibb.co/2n0yGt7/88.png)

# ### 2. What is the workclass percentages of Americans in high-level income group?

# In[170]:


# Your Code is Here
workclass_US = df[(df.salary == ">50K") & (df.native_country_summary == "US")].workclass.value_counts(dropna = False, normalize = True) * 100
workclass_US

Desired Output:

Private            63.314
Self-emp-not-inc    9.192
Local-gov           8.021
Self-emp-inc        7.784
Federal-gov         4.687
State-gov           4.547
Unknown             2.455
Name: workclass, dtype: float64
# In[171]:


# Your Code is Here
fig, ax = plt.subplots()

ax = sns.barplot(x = workclass_US.index, y = workclass_US.values)
plt.xticks(rotation=45)

for container in ax.containers:
    ax.bar_label(container,fmt="%.2f");


# Desired Output:
# 
# ![image.png](https://i.ibb.co/gMHzLgH/89.png)

# ### 3. What is the occupation percentages of Americans who work as "Private" workclass in high-level income group?

# In[172]:


df.head()


# In[173]:


# Your Code is Here
occupation_US = df[(df.salary == ">50K") & (df.native_country_summary == "US") & (df.workclass == "Private")]                  .occupation.value_counts(dropna = False, normalize = True) * 100 
occupation_US

Desired Output:

Exec-managerial     26.438
Prof-specialty      19.476
Craft-repair        14.695
Sales               14.475
Adm-clerical         6.389
Transport-moving     5.442
Tech-support         4.428
Machine-op-inspct    4.428
Other-service        1.674
Handlers-cleaners    1.344
Farming-fishing      0.595
Protective-serv      0.595
Priv-house-serv      0.022
Name: occupation, dtype: float64
# In[174]:


# Your Code is Here

fig, ax = plt.subplots()

ax = sns.barplot(x = occupation_US.index, y = occupation_US.values)
plt.xticks(rotation=90)

for container in ax.containers:
    ax.bar_label(container,fmt="%.2f");


# Desired Output:
# 
# ![image.png](https://i.ibb.co/s3Kd7VS/90.png)

# ### 4. What is the education level percentages of Asian-Pac-Islander race group in high-level income group?

# In[175]:


# Your Code is Here
Asian_Pac_Islander = df[(df.salary == ">50K") & (df.race == "Asian-Pac-Islander")]                     .education.value_counts(dropna = False, normalize = True) * 100 
Asian_Pac_Islander

Desired Output:

Bachelors      35.145
Masters        15.580
HS-grad        12.319
Some-college   11.957
Prof-school     9.783
Doctorate       6.522
Assoc-voc       3.261
Assoc-acdm      2.899
5th-6th         1.087
9th             0.362
11th            0.362
10th            0.362
12th            0.362
Name: education, dtype: float64
# In[176]:


# Your Code is Here
fig, ax = plt.subplots()

ax = sns.barplot(x = Asian_Pac_Islander.index, y = Asian_Pac_Islander.values)
plt.xticks(rotation=90)

for container in ax.containers:
    ax.bar_label(container,fmt="%.2f");


# Desired Output:
# 
# ![image.png](https://i.ibb.co/rZnSFBX/91.png)

# ### 5. What is the occupation percentages of Asian-Pac-Islander race group who has a Bachelors degree in high-level income group?

# In[177]:


# Your Code is Here
Asian_Pac_Islander = df[(df.salary == ">50K") & (df.race == "Asian-Pac-Islander") & (df.education == "Bachelors")]                     .occupation.value_counts(dropna = False, normalize = True) * 100 
Asian_Pac_Islander

Desired Output:

Exec-managerial     27.835
Prof-specialty      25.773
Adm-clerical        12.371
Sales                9.278
Other-service        9.278
Craft-repair         7.216
Tech-support         3.093
Protective-serv      2.062
Transport-moving     1.031
Machine-op-inspct    1.031
Farming-fishing      1.031
Name: occupation, dtype: float64
# In[178]:


# Your Code is Here
fig, ax = plt.subplots()

ax = sns.barplot(x = Asian_Pac_Islander.index, y = Asian_Pac_Islander.values)
plt.xticks(rotation=90)

for container in ax.containers:
    ax.bar_label(container,fmt="%.2f");


# Desired Output:
# 
# ![image.png](https://i.ibb.co/zZVsbJf/92.png)

# ### 6. What is the mean of working hours per week by gender for education level, workclass and marital status? Try to plot all required in one figure.

# In[179]:


# Your Code is Here
g = sns.catplot(x="education_summary",
                y="hours_per_week",
                data=df,
                kind="bar",
                estimator= np.mean,
                hue="sex",
                col="marital_status_summary",
                row="native_country_summary",
                ci=None,
                palette=sns.color_palette(['pink', 'skyblue']));

g.fig.set_size_inches(15, 8)
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Working Hours Per Week by Gender for Education, Workclass, Marital Status')

for ax in g.axes.ravel():

    for container in ax.containers:
        ax.bar_label(container,fmt="%.2f");
    
    ax.margins(y=0.2)

plt.show()


# Desired Output:
# 
# ![image.png](https://i.ibb.co/G5KY8nf/93.png)

# ## <p style="background-color:#9d4f8c; font-family:newtimeroman; color:#FFF9ED; font-size:175%; text-align:center; border-radius:10px 10px;">Dropping Similar & Unneccessary Features</p>
# 
# <a id="6"></a>
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>

# In[180]:


# Your Code is Here
df.info()

Desired Output:

<class 'pandas.core.frame.DataFrame'>
Int64Index: 32537 entries, 0 to 32560
Data columns (total 18 columns):
 #   Column                  Non-Null Count  Dtype  
---  ------                  --------------  -----  
 0   age                     32537 non-null  int64  
 1   workclass               32537 non-null  object 
 2   fnlwgt                  32537 non-null  int64  
 3   education               32537 non-null  object 
 4   education_num           31735 non-null  float64
 5   marital_status          32537 non-null  object 
 6   occupation              32537 non-null  object 
 7   relationship            32537 non-null  object 
 8   race                    32537 non-null  object 
 9   gender                  32537 non-null  object 
 10  capital_gain            32537 non-null  int64  
 11  capital_loss            32537 non-null  int64  
 12  hours_per_week          32537 non-null  int64  
 13  native_country          32537 non-null  object 
 14  salary                  32537 non-null  object 
 15  education_summary       32537 non-null  object 
 16  marital_status_summary  32537 non-null  object 
 17  native_country_summary  32537 non-null  object 
dtypes: float64(1), int64(5), object(12)
memory usage: 5.7+ MB
# In[181]:


# Drop the columns of "education", "education_num", "relationship", "marital_status", "native_country" permanently

# Your Code is Here

df.drop(columns = ["education", "education_num", "relationship", "marital_status", "native_country"], inplace = True)


# ## <p style="background-color:#9d4f8c; font-family:newtimeroman; color:#FFF9ED; font-size:175%; text-align:center; border-radius:10px 10px;">Handling with Missing Value</p>
# 
# <a id="7"></a>
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>

# **Check the missing values for all features basically**

# In[182]:


# Your Code is Here

df.isnull().sum()

Desired Output:

age                       0
workclass                 0
fnlwgt                    0
occupation                0
race                      0
gender                    0
capital_gain              0
capital_loss              0
hours_per_week            0
salary                    0
education_summary         0
marital_status_summary    0
native_country_summary    0
dtype: int64
# **1. It seems that there is no missing value. But we know that "workclass", and "occupation" features have missing values as the "Unknown" string values. Examine these features in more detail.**
# 
# **2. Decide if drop these "Unknown" string values or not**

# In[183]:


# Your Code is Here

df.workclass.value_counts()

Desired Output:

Private             22673
Self-emp-not-inc     2540
Local-gov            2093
Unknown              1836
State-gov            1298
Self-emp-inc         1116
Federal-gov           960
Without-pay            14
Never-worked            7
Name: workclass, dtype: int64
# In[184]:


# Your Code is Here


df.occupation.value_counts()

Desired Output:

Prof-specialty       4136
Craft-repair         4094
Exec-managerial      4065
Adm-clerical         3768
Sales                3650
Other-service        3291
Machine-op-inspct    2000
Unknown              1843
Transport-moving     1597
Handlers-cleaners    1369
Farming-fishing       992
Tech-support          927
Protective-serv       649
Priv-house-serv       147
Armed-Forces            9
Name: occupation, dtype: int64
# In[185]:


# Your Code is Here
df[df.occupation == "Unknown"].workclass.value_counts()

Desired Output:

Unknown         1836
Never-worked       7
Name: workclass, dtype: int64
# In[186]:


# Replace "Unknown" values with NaN using numpy library

# Your Code is Here

df.replace("Unknown", np.nan, inplace = True)


# In[187]:


# Your Code is Here
df.isnull().sum()

Desired Output:

age                          0
workclass                 1836
fnlwgt                       0
occupation                1843
race                         0
gender                       0
capital_gain                 0
capital_loss                 0
hours_per_week               0
salary                       0
education_summary            0
marital_status_summary       0
native_country_summary       0
dtype: int64
# In[188]:


# Drop missing values in df permanently

# Your Code is Here
df.dropna(inplace = True)


# In[189]:


# Your Code is Here

df.isnull().sum()

Desired Output:

age                       0
workclass                 0
fnlwgt                    0
occupation                0
race                      0
gender                    0
capital_gain              0
capital_loss              0
hours_per_week            0
salary                    0
education_summary         0
marital_status_summary    0
native_country_summary    0
dtype: int64
# In[190]:


# Your Code is Here

df.info()

Desired Output:

<class 'pandas.core.frame.DataFrame'>
Int64Index: 30694 entries, 0 to 32560
Data columns (total 13 columns):
 #   Column                  Non-Null Count  Dtype 
---  ------                  --------------  ----- 
 0   age                     30694 non-null  int64 
 1   workclass               30694 non-null  object
 2   fnlwgt                  30694 non-null  int64 
 3   occupation              30694 non-null  object
 4   race                    30694 non-null  object
 5   gender                  30694 non-null  object
 6   capital_gain            30694 non-null  int64 
 7   capital_loss            30694 non-null  int64 
 8   hours_per_week          30694 non-null  int64 
 9   salary                  30694 non-null  object
 10  education_summary       30694 non-null  object
 11  marital_status_summary  30694 non-null  object
 12  native_country_summary  30694 non-null  object
dtypes: int64(5), object(8)
memory usage: 3.3+ MB
# ## <p style="background-color:#9d4f8c; font-family:newtimeroman; color:#FFF9ED; font-size:175%; text-align:center; border-radius:10px 10px;">Handling with Outliers</p>
# 
# <a id="8"></a>
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>

# ### Boxplot and Histplot for all numeric features
# 
# **Plot boxplots for each numeric features at the same figure as subplots**

# In[191]:


# Your Code is Here
plt.boxplot((df[df.select_dtypes('number').columns]), 
            labels=df.select_dtypes('number').columns,
            meanprops={"markerfacecolor":"white", 
                       "markeredgecolor":"black",
                       "markersize":"10"});
plt.show()


# Desired Output:
# 
# ![image.png](https://i.ibb.co/DKMSBDk/94.png)

# In[192]:


# Your Code is Here
index = 0
plt.figure(figsize=(20, 10))
for feature in df.select_dtypes('number').columns:
    index += 1
    plt.subplot(2,3,index)
    sns.boxplot(x=feature, 
                data=df, 
                whis=1.5,
                meanprops={"markerfacecolor":"white", 
                           "markeredgecolor":"black",
                           "markersize":"10"})


# Desired Output:
# 
# ![image.png](https://i.ibb.co/JKtcs9S/95.png)

# **Plot both boxplots and histograms for each numeric features at the same figure as subplots**

# In[193]:


# Your Code is Here
index = 0
plt.figure(figsize=(20, 40))
for feature in df.select_dtypes('number').columns:
    index += 1
    plt.subplot(6,2, index)
    sns.boxplot(x=feature, data=df, whis=1.5)
    index += 1
    plt.subplot(6,2,index)
    sns.histplot(x=feature, data=df)


# Desired Output:
# 
# ![image.png](https://i.ibb.co/fMpP3yR/96.png)

# **Check the statistical values for all numeric features**

# In[194]:


# Your Code is Here


df.describe().T


# Desired Output:
# 
# ![image.png](https://i.ibb.co/t3MJHDr/97.png)

# **1. After analyzing all features, we have decided that we can't evaluate extreme values in "fnlwgt, capital_gain, capital_loss" features in the scope of outliers.**
# 
# **2. So let's examine "age and hours_per_week" features and detect extreme values which could be outliers by using IQR Rule.**

# ### age

# In[195]:


# Your Code is Here
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
sns.boxplot(data=df.age,
            meanprops={"markerfacecolor":"white", 
                       "markeredgecolor":"black",
                       "markersize":"10"})

plt.subplot(1, 2, 2)
sns.histplot(data=df.age);


# Desired Output:
# 
# ![image.png](https://i.ibb.co/SnzH5Nz/98.png)

# In[196]:


# Find IQR defining quantile 0.25 for low level and 0.75 for high level 

# Your Code is Here

low = df.age.quantile(0.25)
high = df.age.quantile(0.75)
IQR = high - low
low, high, IQR

Desired Output:

(28.0, 47.0, 19.0)
# In[197]:


# Find lower and upper limit using IQR

# Your Code is Here

lower_lim = low - (1.5 * IQR)
upper_lim = high + (1.5 * IQR)
lower_lim, upper_lim

Desired Output:

(-0.5, 75.5)
# In[198]:


# Your Code is Here


df[df.age > upper_lim].age.value_counts()

Desired Output:

90    35
76    30
77    20
80    16
79    15
81    14
78    14
84     8
82     7
83     5
88     3
85     3
86     1
Name: age, dtype: int64
# In[199]:


# Define the observations whose age is greater than upper limit and sort these observations by age in descending order

# Your Code is Here

df[df.age > upper_lim].sort_values(by="age", ascending=False)


# Desired Output:
# 
# ![image.png](https://i.ibb.co/x2wDgzQ/99.png)

# ### hours_per_week

# In[200]:


# Your Code is Here
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
sns.boxplot(data=df.hours_per_week,
            meanprops={"markerfacecolor":"white", 
                       "markeredgecolor":"black",
                       "markersize":"10"})

plt.subplot(1, 2, 2)
sns.histplot(data=df.hours_per_week, bins=10);


# Desired Output:
# 
# ![image.png](https://i.ibb.co/xq53X6w/100.png)

# In[201]:


# Find IQR defining quantile 0.25 for low level and 0.75 for high level 

# Your Code is Here
low = df.hours_per_week.quantile(0.25)
high = df.hours_per_week.quantile(0.75)
IQR = high - low
low, high, IQR

Desired Output:

(40.0, 45.0, 5.0)
# In[202]:


# Find the lower and upper limit using IQR

# Your Code is Here
lower_lim = low - (1.5 * IQR)
upper_lim = high + (1.5 * IQR)
lower_lim, upper_lim

Desired Output:

(32.5, 52.5)
# In[203]:


# Your Code is Here
df[df.hours_per_week > upper_lim].hours_per_week.value_counts().sort_index(ascending=False)

Desired Output:

99      80
98      11
97       2
96       5
95       2
94       1
92       1
91       3
90      28
89       2
88       2
87       1
86       2
85      13
84      41
82       1
81       3
80     124
78       8
77       6
76       3
75      63
74       1
73       2
72      68
70     284
68      12
67       4
66      17
65     242
64      14
63      10
62      18
61       2
60    1441
59       5
58      27
57      17
56      91
55     683
54      39
53      23
Name: hours_per_week, dtype: int64
# In[204]:


# Define the observations where  hours per week are greater than upper limit and 
# sort these observations by hours per week in descending order

# Your Code is Here
df[df.hours_per_week > upper_lim].sort_values(by="hours_per_week", ascending=False)


# Desired Output:
# 
# ![image.png](https://i.ibb.co/zGCnbjz/101.png)

# In[205]:


# Your Code is Here
df[df.hours_per_week < lower_lim].hours_per_week.value_counts().sort_index()

Desired Output:

1        8
2       15
3       24
4       28
5       39
6       40
7       20
8      103
9       17
10     223
11       9
12     143
13      19
14      28
15     350
16     182
17      27
18      64
19      14
20    1066
21      23
22      39
23      20
24     220
25     582
26      30
27      28
28      74
29       6
30    1009
31       5
32     239
Name: hours_per_week, dtype: int64
# In[206]:


# Your Code is Here

df[df.hours_per_week < lower_lim].groupby("salary").hours_per_week.describe()


# Desired Output:
# 
# ![image.png](https://i.ibb.co/swYNtdM/102.png)

# In[207]:


# Your Code is Here
df[df.hours_per_week < lower_lim].groupby("salary").age.describe()


# Desired Output:
# 
# ![image.png](https://i.ibb.co/S7RWpxD/103.png)

# **Result :** As we see, there are number of extreme values in both "age and hours_per_week" features. But how can we know if these extreme values are outliers or not? At this point, **domain knowledge** comes to the fore.
# 
# **Domain Knowledge for this dataset:**
# 1. In this dataset, all values are created according to the statements of individuals. So It can be some "data entries errors".
# 2. In addition, we have aimed to create an ML model with some restrictions as getting better performance from the ML model.
# 3. In this respect, our sample space ranges for some features are as follows.
#     - **age : 17 to 80**
#     - **hours_per_week : 7 to 70**
#     - **if somebody's age is more than 60, he/she can't work more than 60 hours in a week**

# ### Dropping rows according to the domain knownledge 

# In[208]:


# Create a condition according to your domain knowledge on age stated above and 
# sort the observations meeting this condition by age in ascending order

# Your Code is Here
df[(df.age < 17) | (df.age > 80)].sort_values(by = "age", ascending = False)


# Desired Output:
# 
# ![image.png](https://i.ibb.co/pJC50ZV/104.png)

# In[209]:


# Find the shape of the dataframe created by the condition defined above for age 

# Your Code is Here

df[(df.age < 17) | (df.age > 80)].shape

Desired Output:

(76, 13)
# In[210]:


# Assign the indices of the rows defined in accordance with condition above for age

# Your Code is Here
drop_index = df[(df.age < 17) | (df.age > 80)].index
drop_index

Desired Output:

Int64Index([  222, 18832, 10545, 11512, 11996, 12975, 14159, 15892, 18277,
            18413, 18725, 19212,  8973, 19489, 19747, 20610, 22220, 24043,
            28463, 31030, 32277, 32367, 10210, 15356,  5370,  4070,  1040,
             6232,  1935,  2303,  5272,  6624,  2891,  5406,  8806,  1168,
            22895, 21835, 24027, 20463,  8381, 32459, 26731, 27795,  9471,
             6214, 14711, 11238,  7720, 15662,  7481, 24395, 23459, 19172,
            16302, 14756,  8431, 20421, 22481, 31855, 13696, 24280,  4834,
            29594, 28948, 12830,   918, 13295, 24560,  3537, 13928, 19045,
             6748,  2906, 21501, 19495],
           dtype='int64')
# In[211]:


# Drop these indices defined above for age

# Your Code is Here

df.drop(drop_index, inplace = True)


# In[212]:


# Create a condition according to your domain knowledge on hours per week stated above and 
# sort the observations meeting this condition by hours per week in descending order

# Your Code is Here
df[(df.hours_per_week < 7) | (df.hours_per_week > 70)].sort_values(by = "hours_per_week", ascending = False)


# Desired Output:
# 
# ![image.png](https://i.ibb.co/rMp7C58/105.png)

# In[213]:


# Find the shape of the dataframe created by the condition defined above for hours per week 

# Your Code is Here


df[(df.hours_per_week < 7) | (df.hours_per_week > 70)].shape

Desired Output:

(621, 13)
# In[232]:


# Assign the indices of the rows defined in accordance with condition above for hours per week

# Your Code is Here
drop_index = df[(df.hours_per_week < 7) | (df.hours_per_week > 70)].index
drop_index

Desired Output:

Int64Index([22216,  5432, 19053, 19141, 19399, 19529, 19731, 19997, 20036,
            21056,
            ...
             6180, 29867,  1036, 11451, 22960, 20909, 25078, 19750,   189,
            24284],
           dtype='int64', length=621)
# In[233]:


# Drop these indices defined above for hours per week

# Your Code is Here

df.drop(drop_index, inplace = True)


# In[234]:


# Create a condition according to your domain knowledge on both age and hours per week stated above 

# Your Code is Here

df[(df.age > 60) & (df.hours_per_week > 60)]


# Desired Output:
# 
# ![image.png](https://i.ibb.co/Ch8XSdW/106.png)

# In[235]:


# Find the shape of the dataframe created by the condition defined above for both age and hours per week


# Your Code is Here

df[(df.age > 60) & (df.hours_per_week > 60)].shape

Desired Output:

(23, 13)
# In[236]:


# Assign the indices of the rows defined in accordance with condition above for both age and hours per week

# Your Code is Here
drop_index = df[(df.age > 60) & (df.hours_per_week > 60)].index
drop_index

Desired Output:

Int64Index([ 1541,  2154,  2184,  2665,  3101,  5417,  6826,  8066,  9646,
            12624, 16634, 18367, 19584, 20125, 23399, 23585, 24903, 25910,
            26625, 27721, 28294, 31342, 32192],
           dtype='int64')
# In[237]:


# Drop these indices defined above for both age and hours per week

# Your Code is Here

df.drop(drop_index, inplace = True)


# In[238]:


# What is new shape of dataframe now

# Your Code is Here
df.shape

Desired Output:

(29974, 13)
# In[239]:


# Reset the indices and take the head of DataFrame now

# Your Code is Here
df.reset_index(drop = True, inplace = True)
df.head()


# Desired Output:
# 
# ![image.png](https://i.ibb.co/5MXPD2b/107.png)

# ## <p style="background-color:#9d4f8c; font-family:newtimeroman; color:#FFF9ED; font-size:175%; text-align:center; border-radius:10px 10px;">Final Step to Make the Dataset Ready for ML Models</p>
# 
# <a id="9"></a>
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>

# ### 1. Convert all features to numeric

# **Convert target feature (salary) to numeric (0 and 1) by using map function**

# In[240]:


# Your Code is Here
df["salary"] = df.salary.map({"<=50K" : 0, ">50K" : 1})
df["salary"]

Desired Output:

0        0
1        0
2        0
3        0
4        0
        ..
29969    0
29970    1
29971    0
29972    0
29973    1
Name: salary, Length: 29974, dtype: int64
# In[241]:


# Your Code is Here
df.salary.value_counts()

Desired Output:

0    22524
1     7450
Name: salary, dtype: int64
# **Convert all features to numeric by using get_dummies function**

# In[242]:


# Your Code is Here
df_dummy = pd.get_dummies(df, drop_first=True)
df_dummy


# Desired Output:
# 
# ![image.png](https://i.ibb.co/0F1SHRt/108.png)

# In[245]:


# What's the shape of dataframe

# Your Code is Here
df.shape

Desired Output:

(29974, 13)
# In[244]:


# What's the shape of dataframe created by dummy operation

# Your Code is Here
df_dummy.shape

Desired Output:

(29974, 34)
# ### 2. Take a look at correlation between features by utilizing power of visualizing

# In[246]:


# Your Code is Here

df_dummy.corr()


# Desired Output:
# 
# ![image.png](https://i.ibb.co/Dgb8RYZ/109.png)

# In[247]:


# Your Code is Here
plt.figure(figsize=(20, 20))
sns.heatmap(df_dummy.corr(), annot=True, cmap="YlGnBu");


# Desired Output:
# 
# ![image.png](https://i.ibb.co/5XH3X4q/110.png)

# In[248]:


# Your Code is Here
df_dummy_corr_salary = df_dummy.corr()[["salary"]].drop("salary").sort_values(by = "salary", ascending = False)
df_dummy_corr_salary


# Desired Output:
# 
# ![image.png](https://i.ibb.co/19RytkS/111.png)

# In[249]:


# Your Code is Here
plt.figure(figsize = (3, 14))
sns.heatmap(df_dummy_corr_salary, annot = True, cmap = "YlGnBu")
plt.show()


# ![image.png](https://i.ibb.co/80GcYKr/112.png)

# In[250]:


# Your Code is Here
plt.figure(figsize = (6, 14))
df_dummy.corr()["salary"].drop("salary").sort_values().plot.barh();


# Desired Output:
# 
# ![image.png](https://i.ibb.co/0MCPc4d/113.png)

# <a id="10"></a>
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>
# 
# ## <p style="background-color:#FDFEFE; font-family:newtimeroman; color:#9d4f8c; font-size:150%; text-align:center; border-radius:10px 10px;">The End of the Project</p>
# 
# <p style="text-align: center;"><img src="https://docs.google.com/uc?id=1lY0Uj5R04yMY3-ZppPWxqCr5pvBLYPnV" class="img-fluid" 
# alt="CLRSWY"></p>
# 
# ## <p style="background-color:#FDFEFE; font-family:newtimeroman; color:#9d4f8c; font-size:100%; text-align:center; border-radius:10px 10px;">WAY TO REINVENT YOURSELF</p>
# 
# ___
# 
