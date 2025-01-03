import pandas as pd 
import os
import numpy as np
from datetime import datetime
def consistency_check():
    """Check that the different dataframes are consistent in the columns they include, if so then they can be concatenated"""
    central_2022=pd.read_csv('counts/central_counts/2022 Q1 (January-March).csv')
    central_2022['type']='central'
    strat_2022=pd.concat([pd.read_csv('counts/strat_counts/2022-Central.csv'),
                        pd.read_csv('counts/strat_counts/2022-Outer.csv'),
                        pd.read_csv('counts/strat_counts/2022-Inner-Part1.csv'), 
                        pd.read_csv('counts/strat_counts/2022-Inner-Part2.csv')])
    strat_2022['type']='strat'
    cycleway_2022=pd.concat([pd.read_csv('counts/cycleway_counts/2022-2-autumn.csv'),
                            pd.read_csv('counts/cycleway_counts/2022-1-spring.csv')])
    cycleway_2022['type']='cycleway'
    dfs=[central_2022, strat_2022, cycleway_2022]
    return all(df.columns.equals(central_2022.columns) for df in dfs)


def get_all():
    """Given the files can be safely concatenated, this function merges all datasets into a single file which is used as the basis of a sampling frame"""
    strat_files=os.listdir('strat_counts')
    strat=pd.concat([pd.read_csv('strat_counts/' + file) for file in strat_files if file[-1]== 'v'])
    cycleway_files=os.listdir('cycleway_counts')
    cycleway=pd.concat([pd.read_csv('cycleway_counts/' + file, 
                                    low_memory=False) for file in cycleway_files if file[-1]=='v'])
    central_files=os.listdir('central_counts')
    central=pd.concat([pd.read_csv('central_counts/' + file) for file in central_files if file[-1]=='v'])
    central['type']='central'
    strat['type']='strat'
    cycleway['type']='cycleway'
    df=pd.concat([strat, central, cycleway])
    return df

def process_data(df):
    """Carries out initial preprocessing of the data - removes counts that are not cycles, 
    removes some irrelevant columns, converts the time into both datetime and timestamp formats 
    (datetime primarily used but the weather API requires timestamp) and adds the location information"""
    to_remove=['Coaches', 'Buses', 'Taxis', 'Motorcycles', 'Medium goods vehicles', 'Light goods vehicles', 
           'Heavy goods vehicles', 'Cars']
    locations=pd.read_csv('Release notes/0-Count locations.csv')
    df['Datetime']=df.Date.astype(str) + " " + df.Time.astype(str)
    df['Datetime']=pd.to_datetime(df['Datetime'], format = "%d/%m/%Y %H:%M:%S")
    df=df[~df['Mode'].isin(to_remove)]
    df=df.drop(['Date', 'Time', 'Year', 'Day', 'Round', 'Path', 'Mode', 'Weather'], axis=1)
    df=df.groupby(
        ['UnqID', 'Dir','type', 'Datetime']
    , as_index=False ).Count.sum()
    df['timestamp']=int(df.Datetime.astype(np.int64)[0]/1000000000)
    df=df.merge(locations[['Site ID', 'Latitude', 'Longitude']], left_on='UnqID', 
                    right_on='Site ID').drop(['Site ID'], axis=1)
    return df

def get_hourly(df):
    "Converts the data to hourly (rather than the 15 minute intervals in the data)"
    hourly=df.groupby([pd.Grouper(key='Datetime', freq="H"), 'UnqID', 
                         'timestamp', 'Latitude', 'Longitude'], as_index=False).agg({'Count':'sum'})
    return hourly

# Sampling note
# The sampling unit is a day's worth of counting at a given site - it is expected that cycling will vary by time of day 
# (due to higher rates in rush hour) and by the day within the year (due to more cycling in warmer months).  
# Using a 'day-site' combination as the sampling unit helps to control for fluctuations during the day by definition, 
# and will control for seasonal changes if the numbers in each month are consistent.  
# To keep the file sizes manageable, aim for around 1000 counts per month. In practice this meant using all data
# from winter months where fewer counts are available, and sampling data from warmer months to get to 1000
# This sampling is achieved via the following functions

def get_profile(month_df):
    """Groups a month's worth of data by site and day and shows how many counts there are for that day-site combination. 
    A complete day is in theory 16 hours but while this is rare, 8 or more would provide a good amount of data.  
    It then profiles the spread of 'day-site' counts within that month. i.e. it returns a table showing, for each value of k 
    (the number of counts in a day at a site), how many day_site combinations match this in the given month. 
    If we multiply the first 2 columns together we get the total number of (hourly) counts this corresponds to. 
    The cumulative count then helps us decide on the value of k which is in the next function below.
"""
    df=month_df.groupby(['UnqID', pd.Grouper(key='Datetime', freq='D')], 
                                       as_index=False).Count.count()
    df=df.rename(columns={"Count": "Counts_at_site_on_day"})
    profile=pd.DataFrame(df.Counts_at_site_on_day.value_counts())
    profile=profile.rename(columns={'count': 'no_of_daysites'}).sort_index(ascending=False)
    profile['counts']=profile.index * profile.no_of_daysites
    profile['cumu_count']=profile.counts.cumsum()
    return profile

def choose_k(profile):
    """Calculate the value of k (day-site min threshold) given the monthly profile to get around 1000 total hourly counts per month.  
    The cumulative count column from the profile function helps decide i.e. want the highest value of k for which cumulative count > 1000 records 
    Want to avoid low values of k if possible as this means much less than a day of data and more of a mix of times of day. 
    However, if the data supports a very high k (e.g. lots of day_sites with the full count of 16) this may not necessarily be optimal 
    as it means fewer locations. Therefore I set a maximum k of 10 to get a decent spread of locations in all months. 
    The process is therefore:
        - is the final cumulative count for the month is < 1000? If yes k=0 i.e. all data will be used
        - would k=10 provide 1,000 or more total counts? If yes, 10 is used
        - otherwise, we go row by row through the profile until we cross the cumulative count of 1,000, giving our corresponding k. 
 """
    ten = profile.loc[profile.index==10]
    if profile.cumu_count.iloc[-1]<1000:
        k=0   # take all from months with few
    elif len(ten)>0 and ten.cumu_count.iloc[0]>1000:  
        k=10      
    else:
        i=0
        while profile.cumu_count.iloc[i]<1000:
            i+=1
        k=profile.index[i]
    return k

def get_sample(month_df):
    """once k is chosen, we may need to take a sample of day_site combinations to get a total of 1000 counts in the month. 
    If k = 0, we've already defined above that we want all data for that month as there will be < 1000.  
    For other months, the chosen k gives far more than 1000, therefore we take a sample. 
    For simplicity, we take 1000/k day_sites e.g. if k=8 we would sample 125 day_sites, 
    however since some of these will have more than 8 counts we'd get mroe than 1000 counts for that month in total.
    Once we have our sample of day_sites, we go back to the monthly data (i.e. before we grouped it by day) 
    and do an inner join with our sample on the day_site combination i.e. all hourly records from the ssampled day_sites are retained. 
    This is our monthly sample. """
    month_df['day_site']=month_df.UnqID + "_" + month_df.Datetime.dt.date.astype(str)
    profile=get_profile(month_df)
    k=choose_k(profile)
    by_day_and_site=month_df.groupby('day_site',as_index=False).Count.count()
    pop=by_day_and_site.loc[by_day_and_site.Count>=k]
    if k == 0:
        samp=pop
    else:
        n=int(1000/k)
        try: 
            samp=pop.sample(n=n, random_state=np.random.RandomState(), weights='Count')
        except:
            samp=pop
    sampled=pd.merge(left=month_df,right=samp, how='inner', on='day_site', suffixes=('', '_remove'))
    sampled.drop([i for i in sampled.columns if 'remove' in i],
                axis=1, inplace=True)
    return sampled

def build_sample(all_years):
    """To create the overall sample, we need to apply the previous 3 functions to each month of data in turn then concatenate""" 
    full_df=all_years.loc[(all_years.Datetime.dt.year <= 2022) & (
        all_years.Datetime.dt.year >=2015)]
    full_df['month_year']=full_df.Datetime.dt.to_period('M')
    full_sample=pd.DataFrame(columns=full_df.columns)
    months=full_df.month_year.unique()
    i=1
    for month in months:
        month_df=full_df.loc[full_df.month_year==month]
        sampled=get_sample(month_df)
        full_sample=pd.concat([full_sample, sampled])
        i+=1
    full_sample=full_sample.drop(['month_year', 'day_site'], axis = 1)
    full_sample['py_datetime'] = full_sample.Datetime.apply(lambda x: x.to_pydatetime())
    full_sample['Timestamp'] = full_sample.py_datetime.apply(lambda x: datetime.timestamp(x)).astype(int)
    full_sample.drop(['py_datetime'], axis = 1)
    return full_sample

