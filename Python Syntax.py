#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Title: Python Training and Reference Syntax
# Purpose: A collection of useful Python code examples
# Author: Thomas Handscomb
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#############################
# Scheduling Python functions
#############################
# Import modules
import datetime
import apscheduler

# Create a function to be run on a schedule.
# In reality this would be the (much more) complicated function to schedule
def DateTimePrint():
    # Define the current time
    now = datetime.datetime.now()
    CurrentTime = now.strftime("%Y-%m-%d %H:%M:%S")
    print(CurrentTime)
      
DateTimePrint()

# Calling the scheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.combining import OrTrigger
from apscheduler.triggers.cron import CronTrigger
     
# Configure the background scheduler to run at a specified time
if __name__ == '__main__':
    scheduler = BackgroundScheduler()
  
    # Configure the trigger conditions
    DatePrint_Jobtrigger = OrTrigger([
    CronTrigger(day_of_week = 'mon-sun', \
    hour=21, minute=12, second = 30)
   ])
    
    # Add the DateTimePrint function as a scheduled job with trigger conditions specified above
    scheduler.add_job(DateTimePrint, DatePrint_Jobtrigger, misfire_grace_time = 30)    
           
print("Scheduler is running")
try:
    scheduler.start()
except (KeyboardInterrupt, SystemExit):
    pass

# Get the scheduler job list
#scheduler.get_jobs()
 
# To stop the scheduler
#scheduler.shutdown()


############################
# Sending Emails with Python
############################

# Import smtplib for the actual sending function
import smtplib

from_address = 'Thomas.Handscomb@<Company name>.com'
to_address = 'Jane.Colleague@<Company name>.com'

# Define parameters
smtphost = "10.32.205.151"
s = smtplib.SMTP(smtphost)

# Send the message via SMTP server.
s.sendmail(from_address, to_address, "Hello")


#############
# User Inputs
#############

original = input("Enter a word:")
if len(original)>0:
    print("Your word entered is " + str(original))
else: print ("empty")

#################
# Return vs Print
#################

# Note the above difference between return and print
def square_ret(n):
    squared = n**2
    return squared
    
def square_print(n):
    squared = n**2
    print(squared)

square_ret(3)
square_print(3)

# By returning an output you can use it downstream
# So use the output in an other square
square_print(square_ret(3))

# However this doesn't work beause square_print doesn't return anything so it's trying to print an empty string
square_print(square_print(3))

##############################
## Loops, % and .format syntax
##############################

# The python % syntax allows you to easily refer to loop variable within a loop
# %d looks for a (decimal) number
# %s looks for a string
# %f looks for a float

for x in range(0, 3):
    print ("We're on loop %d" % (x))
    print ("We're on loop %.2f" % (x))
    print ("We're on loop " + str(x))

# %s looks for a string
for i in range (1, 5):
    print("[Fund_%d]" %(i))
    print("[Fund_" + str(i) + "]")
    print("[Fund_%s]" %("TH"))


# Note also the '{}'.format(a,b) is similar to the %s
a,b = (10, 20)

if a<b:
    print('%s is less then %s' %(a,b))
    print('{} is also less than {}'.format(a, b))
else:
    print('Other')

#########################################
# The 4 inbuilt data structures in Python
#########################################
#~~~~~~
# Lists
#~~~~~~    
Test_List = [3,4,"TH"]
type(Test_List)
#dir(Test_List)

# Can reverse a list
Test_List.reverse()

# Can append to a list, i.e. lists are mutable
Test_List.append(54)

# Can refer to an item by it's position
Test_List[0]

# Look at the data types of the list - can have different data types in them
for i in range(0, len(Test_List)):
    print ('%d' %(i), Test_List[i], str(type(Test_List[i])))    

# Can also define a list as follows - however it splits each individual character as an element
L = list('34TH')
len(L)

#~~~~~~~
# Tuples
#~~~~~~~
v = (1,5,5,19)
type(v)

# Can refer to an item by it's position
v[2]

# Can have different data types as with a list
w = (3,6, 'TY')

# Tuples are not mutable (cannot append for example)
dir(w)

for i in range(0, len(w)):
    print ('%d' %(i), w[i], str(type(w[i])))

#~~~~~~~~~~~~~
# Dictionaries
#~~~~~~~~~~~~~
# Holds (key, value) pairs
stock = {
    "banana": 6,
    "apple": 0,
    "orange": 32,
    "pear": "Out of stock"}

type(stock)

# Can extract the keys and labels, put into a list and refer to by position
list(stock.keys())[0]
list(stock.values())[0]

# Dictionaries are mutable
stock.update({"grape":13})

# Can hold different data types
for i in range(0, len(stock)):
    print ('%d' %(i), list(stock.values())[i], str(type(list(stock.values())[i])))

#~~~~~
# Sets
#~~~~~
S = {2, 3, 4, 5, 5, 5, 5, 7}
type(S)
dir(S)

# Sets are mutable, can union sets for example
S
T = S.union({1,10})


############################
# if __name__ == '__main__':
############################

# Checks if the file is being run directly by python or is it importing another (python) file
print(__name__)

# First set the working directory to the same directory as this .py file
import os
os.getcwd()

os.chdir('C:\\Users\\Tom\\Desktop\\GitHub Page\\Blog Repos\\Useful-Python-Syntax')
os.getcwd()

# The if __name__ == '__main__': syntax in the NonMain.py file is detected here
import NonMain

# Compare this to running NonMain.py directly

########################
## Progress bar on loops
########################

# A useful guide to the progress of loops
from tqdm import tqdm
from time import sleep

for i in tqdm([1, 2, 3]):
    print(i)
    sleep(0.5)
    
for i in tqdm(range(100)):
    sleep(0.02)


###############
## Unit Testing
###############

def test_sum():
    assert sum([1, 2, 3]) == 6, "Should be 6"

if __name__ == "__main__":
    test_sum()
    print("Everything passed")

########################
# Subsetting a dataframe
########################
import pandas as pd

sentiment_df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv', delimiter='\t', header=None)
sentiment_df.head(10)


# Can subset just rows
sentiment_df[0:10] # Gives a dataframe of the first 10 rows, all columns
sentiment_df[:10] # Gives the same first 10 rows

# Can use loc (location) to specify the rows by the index
# As a series
sentiment_df.loc[0]
# More commonly as a DataFrame slice
sentiment_df.loc[[0]]

# using iloc syntax is df.iloc[row range, column range]
sentiment_df.iloc[0:10, :] # Gives first 10 rows, all columns
sentiment_df.iloc[:10, :] # Gives first 10 rows, all columns
sentiment_df.iloc[0:10, 0:1] # Gives first 10 rows, first column (as right ] is a ')' )
sentiment_df.iloc[0:10, 1:2] # Gives first 10 rows, second column (as right ] is a ')' )

# Check object types are all DataFrames
type(sentiment_df[0:10])
type(sentiment_df.iloc[0:10, :])
type(sentiment_df.iloc[0:2000, 0:1])

# Select rows with an AND statement (AND = &)
sentiment_df.columns = ['Text', 'Flag']
sentiment_df_and = sentiment_df[(sentiment_df['Text'] == 'the film is strictly routine') & (sentiment_df['Flag'] == 0)]

# Select rows with an OR statement (OR = |)
sentiment_df_or = sentiment_df[(sentiment_df['Text'] == 'the film is strictly routine') | (sentiment_df['Flag'] == 0)]

##################################
# Pandas dataframe size and naming
##################################
import numpy as np
import pandas as pd

# Create test dataframe
# Note the np.random seed defines a fixed sequence of 'random numbers'
np.random.seed(123)
df_Test = pd.DataFrame(np.random.rand(8, 4))

df_Test

# Rename columns
df_Test.columns = ['a', 'b', 'c', 'd']
df_Test

# Add prefixes and/or suffixes to column names
df_Test = df_Test.add_prefix('X_').add_suffix('_Y')
df_Test

# pandas DataFrames are designed to fit into memory, and so sometimes you need to reduce the DataFrame size in order to work with it on your system.

# Here's the size of the test DataFrame:
df_Test.info(memory_usage='deep')

##################################################
# Importing multiple files into a single DataFrame
##################################################

from glob import glob
import pandas as pd
# Select all files into a list using *wildcard
files = glob('C:\\Users\\Tom\\Desktop\\GitHub Page\\Blog Repos\\Useful-Python-Syntax\\File*.xlsx')
#files = glob('https://github.com/ThomasHandscomb/Useful-Python-Syntax/blob/master/File*.xlsx')

# prints the filenames and the data
for f in files:
    print (f)
    print (pd.read_excel(f))

# Concatenate dataframes underneath each other using a loop
df_null = pd.DataFrame()
for f in files:
    df_append = df_null.append(pd.read_excel(f))
    df_null = df_append

df_append

# More efficient is to concatenate around a loop, use axis parameter to concatenate on rows
df_append2 = pd.concat((pd.read_excel(f) for f in files), axis='rows')
df_append2

df_append2_Index = pd.concat((pd.read_excel(f) for f in files), axis='rows', ignore_index=True)
df_append2_Index

# Or can concatenate on columns
df_append3 = pd.concat((pd.read_excel(f) for f in files), axis='columns')
df_append3

#######################################################
# Timing benefits of list comprehension vs. Loop append
#######################################################

# Continuing from the above example, wrap the start and end of your function in datetime.now()
# and calculate the different
import datetime

time_start = datetime.datetime.now()
df_null = pd.DataFrame()
for f in files:
    df_append = df_null.append(pd.read_excel(f))
    df_null = df_append
time_end = datetime.datetime.now()

time_end - time_start # microseconds=48869

# Compare to the more efficient list comprehension approach
time_list_start = datetime.datetime.now()
df_append2 = pd.concat((pd.read_excel(f) for f in files), axis='rows')
time_list_end = datetime.datetime.now()

time_list_end - time_list_start # microseconds=21913

#####################################
# Dealing with large data 1: Chunking
#####################################

# Python stores data in memory so importing large data sets can be problematic if your data is too large to fit into RAM
# Furthermore it's not enough to simply store data, you need excess RAM to actually do things with it, like processing, once imported
# I've heard a rule of thumb is you need 2 to 3 times the RAM of the size of your data you're working with, or even 5x if using (the excellent) pandas module

# A useful method is break the data set up into smaller pieces using chunksize, (usecols can also be useful to only bring in the columns that you need)
Df_large_chunk = pd.read_csv('https://github.com/ThomasHandscomb/Useful-Python-Syntax/raw/master/ChunkFile.csv'                       
                       , encoding = "ISO-8859-1"
                       , header = 0
                       , usecols = ['ID', 'Variable1']
                       , chunksize=5)

for chunk in Df_large_chunk:
    # perform data filtering 
    print(chunk)    


# Define a function to process each chunk.
# This example simply returns the first column but in practise this will be more complicated processing
def chunk_preprocessing(dframe):
    return pd.DataFrame(dframe.iloc[:,0])

# Then iterate the 'processing' function over each chunk and stitch together. Note that the chunk file is not
# stored in memory and so needs to be run again here along with the Chunk_Colsol line
Df_large_chunk = pd.read_csv('https://github.com/ThomasHandscomb/Useful-Python-Syntax/raw/master/ChunkFile.csv'                       
                       , encoding = "ISO-8859-1"
                       , header = 0
                       , usecols = ['ID', 'Variable1']
                       , chunksize=5)
Chunk_Colsol = pd.concat((chunk_preprocessing(chunk) for chunk in Df_large_chunk), axis='rows')

# View the final dataframe, constructed as the consolidation of the processed chunks
Chunk_Colsol

############################################
# Dealing with large data 2: Random sampling
############################################

# The idea here is to import in only a random sample of the data

import random

# The data to load
file = 'C:\\Users\\Tom\\Desktop\\GitHub Page\\Blog Repos\\Useful-Python-Syntax\\ChunkFile.csv'

# Count the lines
num_lines = sum(1 for l in open(file))

# Sample size - in this case ~25%
size = int(num_lines / 4)

# The row indices to skip - make sure 0 is not included to keep the header!
skip_idx = random.sample(range(1, num_lines), num_lines - size)

# Read the randomly sampled data
sampledata = pd.read_csv(file, skiprows=skip_idx, encoding = "ISO-8859-1")

sampledata

##############
# Scaling Data
##############

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer

# Create data
Sample = {'Price': [2,3,3,3,3, 4, 4, 4, 5,5, 6, 7, 8, 9, 15]}
Sample_df = pd.DataFrame(Sample)

# View Distribution
Sample_df[['Price']].hist(bins = 10, edgecolor = 'black')

# Initialise scalers
MinMaxScaler = MinMaxScaler()
Z_scaler = StandardScaler()
PTileScaler = QuantileTransformer(n_quantiles=100, output_distribution='uniform')
NormScaler = QuantileTransformer(n_quantiles=100, output_distribution='normal')

# Min/Max Scaling: Preserves distribution, converts feature to a range (0,1)
# Z-score scaling: Preserves distribution, converts to a distribution with mean 0 and std. dev of 1
# PTileScaling: Alters distribution, converts to uniform range between 0 and 1
# Norm scaling: Alters distribution, converts to normal distribution

Sample_df['Price_MinMaxScaled'] = MinMaxScaler.fit_transform(Sample_df[['Price']])
Sample_df['Price_ZScaled'] = Z_scaler.fit_transform(Sample_df[['Price']])
Sample_df['Price_PTileScaled'] = PTileScaler.fit_transform(Sample_df[['Price']])
Sample_df['Price_NormScaled'] = NormScaler.fit_transform(Sample_df[['Price']])

Sample_df['Price_MinMaxScaled'].describe()
Sample_df['Price_ZScaled'].describe()
Sample_df['Price_PTileScaled'].describe()
Sample_df['Price_NormScaled'].describe()

# Compare distributions
Sample_df[['Price']].hist(bins = 10, edgecolor = 'black')

Sample_df[['Price_MinMaxScaled']].hist(bins = 10, edgecolor = 'black')
Sample_df[['Price_ZScaled']].hist(bins = 10, edgecolor = 'black')
Sample_df[['Price_PTileScaled']].hist(bins = 10, edgecolor = 'black')
Sample_df[['Price_NormScaled']].hist(bins = 10, edgecolor = 'black')

# Can also preserve distribution and convert to a specific min/max range
MinMaxScaler_Range = MinMaxScaler(feature_range=(0.25, 1))
Sample_df['Price_Range_MinMaxScaled'] = MinMaxScaler_Range.fit_transform(Sample_df[['Price']])
Sample_df['Price_Range_MinMaxScaled'].describe()

Sample_df[['Price_Range_MinMaxScaled']].hist(bins = 10, edgecolor = 'black')

#################
# API via Python
#################
# Instead of a brower asking a website server for a webpage, an API asks the remote server for data
# which it then usually returns in json format
import requests

# Make a get request to get the latest position of the international space station from the opennotify api.
response = requests.get("http://api.open-notify.org/iss-now.json")
# Print the status code of the response.
#dir(response)
print(response.status_code)
print(response.text)

def api_call():
    if response.status_code == 200:
        return response.text
        #print(people.text)
    else:
        print("Error")

api_call()

# Research Python JSON formatter here

#######################
# Lambda apply function
#######################

# Allows you to apply user defined functioned to multiple dataframe columns
import pandas as pd
# Create a simple data frame
Sample_df = pd.DataFrame({'Price': [2,3,3,3,3, 4, 4, 4, 5,5, 6, 7, 8, 9, 15]})
Sample_df

Sample_df.dtypes

# Create a single column with list comprehension
Sample_df['New_Col_Simple'] = ['Above' if x >=5 else 'Below' for x in Sample_df['Price']]
Sample_df

# A single column with lambda
# First define the custom logic function
def custom_col(col):
    if col>= 5:
        outcome = 'Above'        
    else:
        outcome = 'Below'
    return outcome

custom_col(Sample_df['Price'])

# Then apply the custom logic function to the columns needed        
Sample_df['New_Col_Lambda'] = Sample_df.apply(lambda x: custom_col(x['Price']), axis = 1)
Sample_df

# Or can specify the column to apply to in front of the 'apply' function
Sample_df['New_Col_Lambda_2'] = Sample_df['Price'].apply(lambda x: custom_col(x))
Sample_df

# Can define a custom function on multiple columns
def custom_col_multiple(col1, col2):
    if col1 < 4:
        return 'Small'
    elif col1 >= 4 and col2 == 'Below':
        return 'Medium'
    else:
        return 'Big'

# And apply lambda x to the multiple columns
Sample_df['Second_Col_Lambda'] = Sample_df.apply(lambda x: custom_col_multiple(x['Price'], x['New_Col_Simple']), axis = 1)
Sample_df

#######################
# Pip install directory
#######################
# Running <pip show pip> in the Anaconda prompt (or <pip show <any module name>>) will return the file location of the site-packages folder
# where modules are imported to

# Running the below will show the current Python site-packages directory
import site
site.getsitepackages()[1]

# If these are different this may explain why pip install <module name> works however import <module name> in python throws a 'module not found' error
# You should use the version of Anaconda Prompt in the same Anaconda distribution as the Spyder that you're using

######################
# Run Jupyter Notebook
######################

# within cmd, navigate to the folder where the jupyter notebook is saved
# and run <jupyter notebook>
# then open http://localhost:8888/ in a browser  


###############################################################
# Using 'Yield' within a function and loop to create dataframes
###############################################################

sampledata_df = pd.DataFrame({'state': ['Ohio','Ohio','Ohio','Nevada','Nevada'],
           'year': [2000,2001,2002,2001,2002],
           'pop': [1.5,1.7,3.6,2.4,2.9]})

# Say you wanted to create 3 dataframes fromt the above, each containing a single year
# For example:
for year in sampledata_df['year'].unique():
    df_name = 'df_%i' %(year)
    print(df_name)
    df_name = sampledata_df[sampledata_df['year'] == year]
    print(df_name)

df_2000 # NameError: name 'df_2000' is not defined
    
# Using yield in a loop can store the named dataframes
def create_df():
    for year in sampledata_df['year'].unique():
        df_name = 'df_%i' %(year)
        df = sampledata_df[sampledata_df['year'] == year]
        yield df_name, df
    
create_df() # is a generator object

# The list contains the dataframes
for i in range(0, len(list(create_df()))):
    name = list(create_df())[i][0]
    data = list(create_df())[i][1]
    print(name)
    print(data)

# Using dictionaries
d = {}
for i in range(10):
    d['x_%s' %(i)] = i

print(d)
type(d)
d['x_0']

##########
# Plotting
##########
import matplotlib.pyplot as plt

# Create dataframe
simulation_df = pd.DataFrame(np.random.randint(4, size = 100)).reset_index()
simulation_df.columns = ('Simulation', 'Interaction_Count')

pivot = simulation_df.groupby(['Interaction_Count'], as_index=False).count()

# Define a visually appealing plot style
plt.style.use('ggplot')

#~~~~~~~~~~~~~~~
# Plot bar chart
# (x, height)
bars = plt.bar(x=pivot['Interaction_Count'], height=pivot['Simulation'], width = 0.4)
plt.title('Distribution of scores')
plt.ylabel('Count')
plt.xlabel('Scores')
#plt.show()
number_pixel_width = 0.05
# This writes the text (3rd parameter) at the (x,y,text) coordinates
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2 - number_pixel_width, bar.get_height() + 0.5, bar.get_height())

#~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot histogram with labels
arr = plt.hist(simulation_df['Interaction_Count'], bins = 8)
for i in range(8):
    plt.text(arr[1][i]+0.1
             , arr[0][i]+0.15
             , str("") if arr[0][i] == 0 else str(arr[0][i].astype(int)))

##########
# DateTime
##########
from dateutil.relativedelta import relativedelta

df_TimeSeries = pd.read_csv('C:/Users/Tom/Desktop/GitHub Page/Blog Repos/Real-World-Time-Series-Analysis/SectorSales.csv'
                     , encoding = "ISO-8859-1", header=0
                     , usecols = ['Date','Sector', 'Universe Gross Sales EUR M', 'Universe Net Sales EUR M']                      
                     #, parse_dates=["Date"]
                     , index_col = ["Date"])

# Care needs to be shown with indexes when working with Time Series, can set the Date column to be the index
# when reading above however cast as datetime as separate line below
df_TimeSeries.index
df_TimeSeries.index = pd.to_datetime(df_TimeSeries.index, format="%d/%m/%Y")

row = df_TimeSeries[-1:]

# Increment a date by a month
row.index = pd.to_datetime(row.index.date + relativedelta(months=+1) , format="%Y-%m-%d")

# Extract time components
df_TimeSeries_exrt = pd.read_csv('C:/Users/Tom/Desktop/GitHub Page/Blog Repos/Real-World-Time-Series-Analysis/SectorSales.csv'
                     , encoding = "ISO-8859-1", header=0
                     , usecols = ['Date','Sector', 'Universe Gross Sales EUR M', 'Universe Net Sales EUR M']                      
                     )


df_TimeSeries_exrt['Date'] = pd.to_datetime(df_TimeSeries_exrt['Date'], format = "%d/%m/%Y") 

df_TimeSeries_exrt['day'] = df_TimeSeries_exrt['Date'].dt.day
df_TimeSeries_exrt['week'] = df_TimeSeries_exrt['Date'].dt.week
df_TimeSeries_exrt['month'] = df_TimeSeries_exrt['Date'].dt.month
df_TimeSeries_exrt['year'] = df_TimeSeries_exrt['Date'].dt.year

df_TimeSeries_exrt['week_year'] = df_TimeSeries_exrt['Date'].dt.strftime('%Y-%U')

df_TimeSeries_exrt[df_TimeSeries_exrt['Date'] == '2020-02-20']


##############
# Merging data
##############

leftdata_df = pd.DataFrame({'state': ['Ohio','Ohio','Ohio','Nevada','Nevada'],
           'date': ['2016-01-01', '2016-01-02', '2016-01-03', '2016-01-04', '2016-02-26'],
           'pop': [1.5,1.7,3.6,2.4,2.9]})

# Cast date as datetime
leftdata_df['date'] = pd.to_datetime(leftdata_df['date'], format = "%Y-%m-%d")
leftdata_df.info()


rightdata_df = pd.DataFrame({'city': ['Ohio','City2','City3','Nevada'],
          'date': ['2016-01-01', '2016-01-02', '2016-01-03', '2016-01-04'],
           'flag': [1, 2, 3, 4]})

#Cast date as datetime
rightdata_df['date'] = pd.to_datetime(rightdata_df['date'], format = "%Y-%m-%d")
# Make date the index here
rightdata_df.set_index('date', inplace = True)
rightdata_df
    
# Left join on date only
merged_df_left = leftdata_df.merge(rightdata_df['flag'], on = 'date', how = 'left')
merged_df_left

# Inner join on date only
merged_df_inner = leftdata_df.merge(rightdata_df['flag'], on = 'date', how = 'inner')
merged_df_inner

# Left join on date and city
leftdata_df
rightdata_df
rightdata_df[['city', 'flag']]

second_merged_df = leftdata_df.merge(rightdata_df[['city', 'flag']]
                    , left_on = ['date', 'state']
                    , right_on = ['date', 'city']
                    , how = 'left').drop(['city'], axis = 1)

second_merged_df

second_merged_df['date'].dt.month

###########
# Pipelines
###########
from sklearn.pipeline import Pipeline

# Chain link preprocessing and model building steps together

# Start with some raw data
leftdata_df = pd.DataFrame({'state': ['Ohio','Ohio','Ohio','Nevada','Nevada'],
           'date': ['2016-01-01', '2016-01-02', '2016-01-03', '2016-01-04', '2016-02-26'],
           'pop': [1.5,1.7,3.6,2.4,2.9]})
 
leftdata_df

# Say you wanted to first scale the data and then drop some rows
temp = MinMaxScaler().fit_transform(leftdata_df[['pop']])

MinMaxScaler(leftdata_df[['pop']])

#temp[0] = 

# Define the constituent functions
preprocessor = Pipeline(
        [
            ("scaler", MinMaxScaler())
            #("pca", PCA(n_components=2, random_state=42)),
        ]
        )

# Stitch together all the components into a full pipeline
Full_Pipe = Pipeline(
    [
        ("preprocessor", preprocessor)
        #("clusterer", clusterer)
    ]
)

# Call the pipeline
p = Full_Pipe.fit(leftdata_df[['pop']])

##########################################
# Common uses of Python and useful modules
##########################################
# Use 1: 
    # Web Development 
# Modules: 
    # Requests
    # Django (web framework to make websites)
    # Flask (lighter weight web framework than Django)
    # Beautiful Soup (for HTML scraping)
    # Selenium (for website interation)
    
# Use 2:
    # Data Science
# Modules:
    # numpy (allows mathematical operations, implemented in C so much quicker than equivalent in base python)
    # pandas (reading and working with data frames)
    # matplotlib (data visualisation)
    # nltk (NLP or text preprocessing)
    # opencv (image and video data processing)

# Use 3:
    # Machine Learning
# Modules:
    # Tensorflow
    # keras (a higher level API for tensorflow, allows easier access to TensorFlow features)
    # pytorch
    # sci-kit learn (lighter weight for regression and classification problems)
    
# Use 4:
    # Building graphical user interfaces
# Modules:
    # PyQt5 (GUI/desktop application builder. This is what Spyder IDE is built with)