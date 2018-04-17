import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib.request

#
#  Parameters: dataframe containing analyst scored domains 
#  Returns: preprocessed dataframe 
#
def preprocess_analyst_scored_domains(df):
  # Drop of unecessary features
  df = df.drop(['historyDataScore', 'medianLoadTime','speedPercentile', 
                                                    'trafficDataReachRank', 'reachPerMillionValue', 'pageViewsRankValue',
                                                    'pageViewsPerMillionValue', 'usageStatisticRankValue', 'reachRankValue', 
                                                    'BellonaStatus', 'trafficDataRank'], axis=1)

  # Drop unlabeled examples
  df = df[pd.notnull(df['analystResult'])] 

  # Map private registration status 
  df['privateRegistrationStatus'] = df['privateRegistrationStatus'].replace(np.nan, 0)
  df['privateRegistrationStatus'] = df['privateRegistrationStatus'].map(lambda x: 1 if x != 0 else 0)

  # Map registrant contact country
  df['registrantContactCountry'] = df['registrantContactCountry'].map(lambda x: 1 if x == 'UNITED STATES' else 0)

  # Normalize domain age
  df['domainAge'] = (df['domainAge'] - df['domainAge'].mean())/(df['domainAge'].std())

  # Map analyst result to 0 or 1 
  df = df[pd.notnull(df['analystResult'])] 
  ar = {'TRUE': 1, 'FALSE': 0, 'TRUE ': 1, 'FALSE ': 0}
  df['analystResult'] = df['analystResult'].map(ar)
  return df

#
#  Parameters: preprocessed analyst scored domains
#  Returns: 2 Balanced Dataframes, 1 consisting of valuable domains, 1 invaluable domains
#
def class_imbalance(df):
  # Get valuable domains and then shuffle them
  valuableDomains = df[df['analystResult'] == 1]
  valuableDomains = shuffle(valuableDomains)

  # Get non valuable domains 
  nonValuableDomains = df[df['analystResult'] == 0]

  # Cut off any extra valuable domains so that there are the same number as non valuable domains
  valuableDomains = valuableDomains[:nonValuableDomains.shape[0]]

  # Assert to make sure that the valuable and nonvaluable domains are now the same size
  assert(nonValuableDomains.shape == valuableDomains.shape)
  
  return valuableDomains, nonValuableDomains

# Function to take HTML and return only the visible text
def text_from_html(body):
  soup = BeautifulSoup(body, 'html.parser')
  texts = soup.findAll(text=True)
  visible_texts = filter(tag_visible, texts)
  return u" ".join(t.strip() for t in visible_texts)


# Function that determines whether the text would be visible on the web page
def tag_visible(element):
  if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
      return False
  if isinstance(element, Comment):
      return False
  return True