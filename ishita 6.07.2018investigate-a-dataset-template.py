
# coding: utf-8

# >
# # Project: Investigate a Dataset (The Movie Database )
# 
# #To analyze and invertigate dataset of TMDb
# 
# This dataset TMDb (The Movie Database ) contains information on 10866 movies giving various data on runtime, budget, revenue , directors and genres of the films etc. I choose this dataset as I was interested to know more about the movies which is part of our life. So the curiosity to know more about the movies made me select this dataset.
# I tried to explore various variables and its co relation with other variables within the dataset. 
# like-
# ~ relationship between release_year and popularity.
# ~ relationship between runtime and release_year.
# ~ relationship between runtime and popularity
# 
# I will explore and find answers to the following questions in the dataset.
# ~ 1 What is the co realtionship between runtime and popularity? More runtime means more popularity?
# ~ 2 Which are the features associated to high popularity (ranking)?
# ~ 3 What has been the budget trends over the years? Did it increase or decrease over the years?
# ~ 4 How does popularity change over the years, how is its trend over the years?
# ~ 5 Which are the movies that got highest and lowest vote average?
# ~ 6 Which are the directors who made highest number of movies and what is the difference in their popularity ranking?
# ~ 7 More vote average means movies with more popular ranking?
# ~ 8 What is the co-relationship between Budget and popularity?
# ~ 9 Which are the top 5  movies with i.highest budget ii. with highest popularity iii.vote average 
# ~ 10 Which is the genre with highest number of movies in the dataset?
# ~ 11 How has been the trend of the movie release over the years?
# ~ 12 Which is the most popular movie in this dataset?
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# 
# ## Introduction
# 
# To get familiarize with the dataset. Understand the data to frame questions. The number of columns and rows, unique values in the dataset.

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

df= pd.read_csv('tmdb-movies.csv')
df.head()


# Here we see casts, genres, production companies are strings seperated by |. Rest other values in the dataset seems okay.

# In[11]:


df.shape


# Observation : There are 10866 movie samples and 21 columns in the dataset.

# In[12]:


df.isnull().sum()


# Observation : Here we see there are missing data in id, cast, hopepage, director, tagline, keywords, overview, production comapnies.
# Solution  : I will drop null values columns with small quantity of nulls : cast, director, and genres, in further data cleaning section.

# In[13]:


# To check the unique values in the data set
df.nunique()


# Observation : We seee unique values of each columns.

# In[14]:


# to check the data types in the dataset.
df.dtypes


# In[15]:


df= pd.read_csv('tmdb-movies.csv')
col_name = df.columns.tolist()
print (col_name)


# I get to see all the columns in the dataset. This helps to understand the columns in the dataset and know your data well.

# In[16]:


# check if there is NaN value in each column
df.isna().any()


# Observation : There are NaN values in many columns.

# In[17]:


# Check the number of NaN values in each feature
df.isna().sum()


# observation : Homepage and tagline have highest number of NaN values followed by keywords in this dataset.

# 
# ## Data Wrangling
# 
# 
# ### General Properties

# # After discussing the structure of the data, Now we will see any problems that need to be adressed in data cleaning process.

# In[18]:


# check which columns have missing values with info()
df.info()


# Observation: Here I notice few columns have null values which I will clean later in further data cleaning process.

# In[19]:


# To find duplicate rows
df.duplicated().sum()


# Observation : There is one duplicate row. Hence I will drop it in data cleaning section.

# In[10]:


# To understand the data better, summary statistics
#Maximum run count is of 900 mins, Majority of the films have runtime of 111 minutes.
#df.describe()


# we get mean and and maximum values of each columns and so can understand data better. Maximum runtime of the movies in this dataset is of 900 mins, Majority of the films have runtime of 111 minutes. Maximum of the films are of the year 2015. I even see that few movies ven have budget of 0$. 

# 
# ### Data Cleaning

# # Will perform data cleaning steps in this part of the section.

# In[20]:


# After discussing the structure of the data and any problems that need to be
# To drop or remove the duplicate rows or data in the datatset
df.drop_duplicates(inplace=True)


# I dropped the duplicate rows in the dataaset.

# In[21]:


#Lets check it one more time whether there are any more duplicate rows or data in the dataset.
df.duplicated().sum()


# Observation: Now, we can see no duplicate rows in the dataset.

# In[22]:


# To deal with missing data.
drop =['cast', 'director', 'genres']
df.dropna(subset= drop, how= 'any', inplace= True)


# I have dropped the null values in columns- cast, director, genres.

# In[23]:


#Lets again check the dataset to see if the changes are made.
df.isnull().sum()


# Observation: I can notice here that null values in homepage, tagline, keywords, and production_companies. I will not use homepage, tagline and keywords in my data analysis in future so i will drop these columns later. 

# In[24]:


# To check number of values in the budget with 0 value.
df_budget= df.groupby('budget').count()['id']
df_budget.head(5)


# In[25]:


# To check number of values in the revenue with 0 value.
df_revenue= df.groupby('revenue').count()['id']
df_revenue.head(5)


# There can be no films with 0 budget and 0 revenue.

# Note:- I will replace the values of 0 in budget and revenue with null value in budget and revenue column as they are huge in amount that is more that 50 percent of the movies in the dataset. It will be wise decision to replace the 0 values with null values.

# In[26]:


df['budget'] = df['budget'].replace(0,np.NaN)
df['revenue'] = df['revenue'].replace(0,np.NaN)
df.info()


# In[27]:


# lets see the number of movies with 0runtime and its not possible to have such movies with no that is 0 runtime.
df_runtime= df.groupby('runtime').count()['id']
df_runtime.head(5)


# Observation: There are 28 such movies with 0 runtime. 
# Solution: Its better to just drop them or remove them from the dataset as they are very few.

# In[28]:


df.query('runtime!=0', inplace=True)
df.query('runtime==0')


# Observation : So now no columns with runtime = 0

# In[29]:


df_runtime= df.groupby('runtime').count()['id']
df_runtime.head(5)


# In[30]:


# to rop columns that are not needed for data anlysis from the dataset.

df.drop(['imdb_id', 'tagline','homepage','keywords', 'overview'], axis= 1, inplace=True)


# In[31]:


# To check the changes made 
df.head(1)


# In[32]:


df.describe()


# Observation: We can see no 0 values in the budget and revenue. The statistics analysis is better with out values with 'zero'.

# # Initial analysis of the data. 
# # Exploratory Data Analysis
# 

# In[33]:


df.hist(figsize=(10,10));


# Observation: - We can understand the data better with the help of histogram. Few histograms are left skewed histograms like runtime release year. the mean of the data will be less than the median.
# 
# -Maximum of the histograms are right skewed like revenue_adj , vote_count, budget, budget_adj etc.Hence mean will be greater than the median. 
# 
# -Only vote_average has normal didtribution of the data in that column, It looks like the bell curve.

# # Relationship between the variables.

# In[34]:


# To plot relationship between release_year and popularity
df.plot(x='release_year', y='popularity', kind='scatter');


# Observation: From the above scatter plot we see that there has been a gradual increase in the popularity ranking from 1960 to 2015. Post 2012 a great rise in the popularity ranking.

# In[35]:


#plot relationship between runtime and release_year
df.plot(x='runtime', y='release_year', kind='scatter');


# Observation: From the above scatter plot we can observe that there has been rise in runtime post 1990 and pre 1990 the runtime of the movies in the dataset remains almost similar till 1960. There is a mojor increase in runtime from 2000 to 2015.

# In[36]:


#plot relationship between runtime and popularity
df.plot(x='runtime', y='popularity', kind='scatter');


# Observation : Maximum of the movies are in between 100 and 180 minustes runtime range. There are very few movies whoes runtime crosses 200 minutes mark. To my surprise there are very few movies that even cross 250 minutes, even 400 minutes runtime.

# # Question 1  What is the co realtionship between runtime and popularity? More runtime means more popularity?

# In[37]:


df.runtime.median()


# In[38]:


df.popularity.median()


# Observation : Here we get the median of the runtime and the popularity ranking of the dataset

# In[39]:


# select movies with less and more than the median runtime
less_runtime = df[df.runtime <99 ]
more_runtime = df[df.runtime >=99 ]


# In[40]:


num_runtime = df.shape[0]
num_runtime == less_runtime ['popularity'].count() + more_runtime['popularity'].count() # should be True


# In[41]:


# To get mean popularity for the less runtime and more runtime groups
less_runtime.popularity.mean(), more_runtime.popularity.mean()


# Observation: The mean popularity of the less run time is lesser than the mean popularity of the more time. Hence more the runtime more is the popularity.

# In[42]:


median = df['runtime'].median()
less = df.query('runtime < {}'.format(median))
more = df.query('runtime >= {}'.format(median))

mean_popularity_less = less['popularity'].mean()
mean_popularity_more = more['popularity'].mean()


# In[43]:


locations = [1, 2]
heights = [mean_popularity_less, mean_popularity_more]
labels = ['Less', 'More']
plt.bar(locations, heights, tick_label=labels)
plt.title('Average popularity ranking by runtime')
plt.xlabel('Runtime')
plt.ylabel('Average popularity ranking');


# Observation: Less the runtime, less is the average popularity ranking and more the runtime, more is the average popularity ranking.

# In[44]:


df[df['runtime']> 99].shape[0]


# There are 5252 films which are more than the median runtime of 99 minutes

# In[45]:


df[df['runtime']<= 99].shape[0]


# Observation: There are 5614 films which are less than or equal to the median runtime of 99 minutes in this dataset. There are more movies of less than the median runtime of 99 minutes than the movies whose runtime is more than the median.

#  # Question 2 Which are the features associated to high popularity (ranking)?

# In[46]:


top_popularity = df.query('popularity > popularity.mean()')
top_popularity.describe()


# Observation:
# The above statistical information is of the movies that have been better than the movies that have average popularity ranking.
# Hence the count of such movies are of 3023.
# The maximum popyularity ranking a movie can get is 32.98. Maximum of such movies have 119 minutes of runtime.

# # Question 3 What has been the budget trends over the years? Did it increase or decrease over the years?

# In[47]:


df_budget_year = df.groupby(['release_year'])['budget'].count()
df_budget_year.plot()


# Observation: As the years passed the budget of the films increased after year 2000 there was a rise in the budget. You acn see a slight drop post 2011.

# # Question 4 How does popularity change over the years, how is its trend over the years?

# In[48]:


# to compute the mean for popularity by release year.
popularity_mean = df.groupby('release_year').mean()['popularity']
popularity_mean.tail()


# In[49]:


popularity_median = df.groupby('release_year').median()['popularity']
popularity_median.tail()


# Observation: By this we can understand the median of the popularity over the years. There was a drop in the median popularity of the films post 2011 and pre 2015.

# Let us visualize it.

# In[50]:


index_mean = popularity_mean.index
index_median = popularity_median.index


# In[51]:


sns.set_style('whitegrid')
#set x, y axis data
#x1, y1 for mean data; x2, y2 for median data
x1, y1 = index_mean, popularity_mean
x2, y2 = index_median, popularity_median
#set size
plt.figure(figsize=(10,6))
#plot line chart for mean and median
plt.plot(x1, y1, color = 'g', label = 'mean')
plt.plot(x2, y2, color = 'b', label = 'median')
#set title and labels
plt.title('Popularity Over Years')
plt.xlabel('Year')
plt.ylabel('Popularity');
#set legend
plt.legend(loc='upper left')


# Observation: Through the graph we can understand that the the mean popularity has been increasing year to year and it was highest, i.e. it's peak was in 2015. The average popularity of the movies have been on high trend since past few years.
# 

# # Question 5 Which are the movies that got highest and lowest vote average?

# In[52]:


df.describe().vote_average


# In[53]:


bin_edges = [1.50, 5.40, 6.0, 6.60, 9.20]


# In[54]:


bin_names = ['low', 'medium', 'medium-high', 'high']


# In[55]:


df['vote_rating'] = pd.cut(df['vote_average'], bin_edges, labels=bin_names)


# In[56]:


df.head()


# In[67]:


bin_names = df.groupby(['vote_rating','release_year']).median()
bin_names.tail(20)


# Observation: So by this we can see the movies as per their id to know the movies with vote rating as 'High' to understand which are those movies and other factors associated to them. The movie in 2015 got highest vote average.

# In[69]:


result_mean = df.groupby('vote_rating')['popularity'].mean()
result_mean


# In[70]:


vote = np.arange(len(result_mean))  
# the width of the bars
width = 1.0       
vote


# In[74]:


# plot bars
#set style
sns.set_style('darkgrid')
bars = plt.bar(vote, result_mean, width, color='r', alpha=.10, label='mean')

# title and labels
plt.ylabel('popularity')
plt.xlabel('vote rating')
plt.title('Popularity with vote rating')
locations = vote  # xtick locations，345...
labels = result_mean.index  
plt.xticks(locations, labels)
# legend
plt.legend()


# Observation: We can undertand by ths bar graph that more the vote rating more is the popularity. This is so obvious more votes that a movie gets through facebook or other means of survey more popular it becomes as more people go to watch it.

# In[77]:


df_high=df[df['vote_rating'] =='high']
df_high.head(10)


# Observation: We can see the list of movies that got 'high' vote rating and and even its high popularity. Mad max and star wars are among the high vote rating band and even high popularity too. We get list of movies with high vote_rating.

#  # Question 6 Which are the directors who made highest number of movies and what is the difference in their popularity ranking?

# In[78]:


df.director.value_counts()


# In[80]:


df.query('director in ["Clint Eastwood"]').original_title.nunique()


# Woody Allen and Clint Eastwood are two directors who directed highest number of films than any directors in the dataset.

# In[81]:


df_CE=df[df['director'] =='Clint Eastwood']
df_CE.head()


# In[82]:


df_WA=df[df['director'] =='Woody Allen']
df_WA.head()


# In[83]:


# More statistical information on popularity on Clint's movies.
df_CE['popularity'].describe()


# The maximum popularity that he got was of 3.

# In[44]:


# More statistical information on popularity of Woody's movies.
df_WA['popularity'].describe()


# Observation : In terms of Maximum popularity ranking Woody Allen got was 1.36, lesser than Clint Eastwood.

# Let us visualize it.

# In[84]:


fig, ax = plt.subplots(figsize=(8,6))
ax.hist(df_CE['popularity'], alpha=0.5, label='Clint Eastwood')  
ax.hist(df_WA['popularity'], alpha=0.5, label='Woody Allen')
ax.set_title('Distribution of popularity of Clint Eastwood and Woody Allen')
ax.set_xlabel('Popularity')
ax.set_ylabel('Count')
ax.legend(loc='upper right')
plt.show()


# Observation: Client Eastwood made more popular movies than Woody Allen, Though Woody Allen made more movies than Clint Eastwood.

# # Question 7 More vote average means movies with more popular ranking?

# In[85]:


df.plot(x='vote_average', y='popularity', kind='scatter');


# Observation: The movies between vote average 6 and 8 are very popular. The movies with 9 popularity ranking do not have high vote average.

# # Question 8 What is the co-relationship between Budget and popularity?

# In[86]:


formatter = lambda x: '{:,.0f}'.format(x)
df['budget_adj'] = df['budget_adj'].map(formatter)


# In[87]:


formatter = lambda x: '{:,.0f}'.format(x)
df['revenue_adj'] = df['revenue_adj'].map(formatter)


# In[88]:


df.head()


# In[89]:


plt.scatter(df['budget'], df['popularity'])
#labels, titles
plt.xlabel('budget')
plt.ylabel('Popularity')
plt.title('Budget vs Popularity');


# Observation: By the above scatter plot we notice that the theres no much co relation between the budget of the movies and its popularity, that is more the budget more is its popularity is not true. There are movies whose budget were more than the other maximum movies but they were not popular.

# # Question 9 Which are the top 5  movies with i.highest budget ii. with highest popularity iii.vote average 

# In[90]:


df.nlargest(5, 'popularity')


# observation: Here we can get top 5 movies with highest popularity. But I can see that though Jurassic world is more popular than the first runner up Mad Max their vote average is different. Mad Max has more vote_average than Jurassic world and release year is the same. Popularuity of this movie must be due to its caast or story of graphic effects. From these 5 movies 2 movies falls into the category of medium-high vote_rating.

# In[91]:


df.nlargest(5, 'budget')


# Observation: We get to see top 5 movies with the highest budget. Movie - The Warrior's Way tops the list. Superman Returns which is 5th in the list has low vote_rating.
# 
# The Warrior's Way though tops the list of high budget movies its popularity is very less than the other movies.

# In[112]:


df.nlargest(5, 'vote_average')


# Observation: 
# Movie 'The Story of Film: An Odyssey' tops the list of vote_average. 

# # Question 10 Which is the genre with highest number of movies in the dataset?

# In[93]:


def breakdata(column):
    data =df[column].str.cat(sep='|')
    data = pd.Series(data.split('|'))
    count = data.value_counts(ascending=False)
    return count


# In[94]:


count = breakdata('genres')
count.head()


# Observation: Drama has highest number of movies and it is followed by comedy.

# # Question 11 How has been the trend of the movie release over the years?

# In[108]:


movie_count = df.groupby('release_year').count()['id']
movie_count.head()


# In[109]:


#set style
sns.set_style('darkgrid')
#set x, y axis data
# x is movie release year
x = movie_count.index
# y is number of movie released
y = movie_count
#set size
plt.figure(figsize=(12, 7))
#plot line chart 
plt.plot(x, y, color = 'r', label = 'mean')
#set title and labels
plt.title('Number of Movie Released year by year')
plt.xlabel('Year')
plt.ylabel('Number of Movie Released');


# Observation: As the years passed we see there was a rise in the number of movies released each year.

# # Question 12 Which is the most popular movie in this dataset?

# In[140]:


popular_movie = df[['original_title', 'popularity']]


# In[142]:


plt.figure(figsize=[30,20])
sns.barplot(data=movie_popu.sort_values(by='popularity', ascending=False)[0:50], x='original_title', y='popularity')
plt.xticks(rotation=100)
plt.xlabel('Movie')
plt.ylabel('Popularity rating')
plt.title('The popularity rating of each movie in the dataset')


# Observation: Jurassic world , Mad Max & Interstellar' are the most popular movies in the dataset and they are way morre popular than many other movies in the dataset.

# ## Conclusions
# 

# 
# Data cleaning: In the data cleaning process I found that many values were null values in various columns. Hence I droppped null values in cast, genres and director.
# The major problem with the data set was of 'zero' values in the dataset  in the revenue and budget column. And more than 50% of the films had 'zero' value in their columns. Which is realyy impossible in reality as there can be no films that had zero budget and revenue too zero. To my surprise that it was in large numbers. Hence I decided to to retain these rows and replace zero values with null values, to keep the integrity of the data.
# The same case exists in column 'runtime' , but the zero values in that column very few so I dropped them.
# 
# Overall the data was clean and had only one duplicate row and it was dropped. The number of movies in the datset were enough to the analysis and will help us to understand the trend of the movies over the years and its co relation with other variables in better way.
# 
# Drawback in the dataset:
# It will be more informative if the actual cost of thee film was also included in the dataset because budget minus revenue is not the accurate way tto calculate the profit made by the movies.
# The zero values in the revenue and budget were in excess and hence making calualtions related to these columns would not be a good idea.
# 
# Insights:
# 
# We usually think more is the budget more is the popularity as viewers are more keen to such movies which are expensive due to its expenses spent on technicl effects and locations of shoots or extra vagant expenses to give a best view experience. By the scatter plot I notice that the theres no much co relation between the budget of the movies and its popularity, that is more the budget more is its popularity is not true. There are movies whose budget were more than the other maximum movies but they were not popular.
# 
# I was more curious to know if the popularity of the films, does it increase or decrease over the year with advent of technology and other means of entertainment in our day to day like like plays, game stations and tourism or amusemment parks etc. Through scatter plot I could undersatnd the popularity of films have increased over the years and i think that can be due to innovation in film making of social media in the recent years.
# 
# The question that I explored was that if the runtime is more how is its popularity? The anser to it was very much obvious that more runtime, more is the popularity as people prefer a film more engaging with better story and entertain them for atleast 2 hours.
# 
# Another insight that we can get is the more the popularity more is the vote avrage ranking and popular movies will get better vote average by the viewers. 
# 
# We usually think that directors who make maximum movies must be most popular but this was challenged by the graph that suggested that, client Eastwood made more popular movies than Woody Allen, Though Woody Allen made more movies than Clint Eastwood.
# 
# With more other forms of entertainement and inflation, I was curious to know do film makers make more movies or has it decreased over the years, As the years passed we see there was a rise in the number of movies released each year, but a recent drop near 2013-2015.
# 
# Ofcourse, I wanted to know the most popular movie in the dataset and Jurassic world was the most popular movie till date and it is not surprisisng to know that as it was indeed block buster movie.
#      
# It was certainly a great experience to analyze this dataset through various statistical graphs to clear pre concieved notions and know more about the dataset.
