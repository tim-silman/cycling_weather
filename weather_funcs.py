import pandas as pd
import requests
from datetime import datetime
import numpy as np

def get_weather(lat, lon, timestamp):
    """Runs a single API call to extract weather information for a given location and time
        Parameters:
            lat:latitude of the location
            lon:longitude of the location
            timestamp:date and time for which weather is required in timestamp format
        Returns:
            Weather information from the API in JSON if available"""
    endpoint="https://api.openweathermap.org/data/3.0/onecall/timemachine"
    payload= {'lat': lat, 
          'lon':  lon,
          'dt': timestamp,
          'appid': '*******'}
    
    cols=['sunrise', 'sunset', 'temp', 'wind_speed', 'wind_deg', 'visibility', 'clouds','rain', 
          'weather']
    try:
        resp= requests.get(endpoint, params=payload)
        weath=resp.json()['data'][0]
        for col in cols:
            if col not in weath.keys():
                weath[col]=None
    except:
        weath={}
        for col in cols:
            weath[col]=None
    return [weath[col] for col in cols]

def weather_into_df(df):
    """Runs the get_weather function for an entire dataframe then adds it to the dataframe
    with each weather variable in its own column"""
    cols=['sunrise', 'sunset', 'temp', 'wind_speed', 
      'wind_deg', 'visibility', 'clouds','rain', 'weather']
    df=df.reset_index() 
    df['weath']=df.apply(
        lambda row: get_weather(row['Latitude'], row['Longitude'], row['Timestamp']), axis=1)
    weath_df=pd.DataFrame(df.weath.tolist(), columns = cols)
    try:
        weath_df['weather']=weath_df.weather.apply(lambda x: x[0])
        weath_df['main_descr'] = weath_df.weather.apply(lambda x: x['main'])
        weath_df['detail_descr'] = weath_df.weather.apply(lambda x: x['description'])
    except:
        weath_df['main_descr'] = None
        weath_df['detail_descr'] = None
    weath_df.drop(['weather'], axis = 1, inplace=True)
    df=pd.concat([df, weath_df], axis=1)
    df['site_time'] = df.UnqID + "_" + df.timestamp.astype(str)
    df=df.set_index('site_time')
    return df

def add_weather(thou, sample):
    """Adds weather information to a sub-sample of data and pickles it. This was done rather than pass the entire sample
     to the weather_into_df function given API call limits
            Parameters:
                thou: an integer representing the starting index to use as the sub-sample of 1000 e.g. '3' means rows 3000-3999 would be used
                sample: the sample file from which the sub-sample is taken 
            returns:
                No object returned, however the sample with weather is pickled so all can be concatenated once done"""
    low=int(thou * 1000)
    high = low + 1000
    df=sample.iloc[low:high]
    df = weather_into_df(df)
    path = f"samples/{str(low)}to{str(high-1)}.pkl"
    df.to_pickle(path)
    print(f"{str(low)} to {str(high-1)} DONE")

def get_darkness (row):
    """Estimates how dark it was when an observation was taken i.e. if between sunrise and sunset it's light, 
    half an hour before sunrise or after sunset is gloomy, between these is dark"""
    try:
        since_sunrise  = row.Timestamp - row.sunrise
        before_sunset = row.sunset - row.Timestamp
        if since_sunrise > 0 and before_sunset > 0:
            darkness = 'light'
        elif -1800 < since_sunrise <= 0 or -1800 < before_sunset <= 0:
            darkness = 'gloomy'
        elif since_sunrise < -1800 or before_sunset < -1800:
            darkness = 'dark'
        else:
            darkness = None
    except:
        darkness = None
    return darkness

def tidy_main(row):
    """helper function to resolve some missing data in the main description column where the API return is formatted incorrectly"""
    if pd.isna(row.main_descr): 
        try: 
            weather = row.weath[-1][0]
            row['main_descr'] = weather['main']
        except:
            row['main_descr'] = None
    else:
        row['main_descr'] = row.main_descr    
    return row['main_descr']

def tidy_detail(row):
    """As above but for the detailed descriptoin"""
    if pd.isna(row.detail_descr): 
        try: 
            weather = row.weath[-1][0]
            row['detail_descr'] = weather['description']
        except:
            row['detail_descr'] = None
    else:
        row['detail_descr'] = row.detail_descr    
    return row['detail_descr'] 

def add_combined(df):
    """To assist analysis, creates a new combined description variable i.e. it retains some granularity of the detail,
    but for some categories only the main description is used"""
    df['main_descr_adj'] = np.where(df.main_descr.isin(['Snow', 'Thunderstorm', 'Rain']), 'Rain/ storm/ snow', 
                                  (np.where(df.main_descr.isin(["Mist", "Fog"]), 'Mist or fog', df.main_descr)))
    df['combined_descr'] = np.where(df.main_descr_adj == "Rain/ storm/ snow", df.detail_descr, df.main_descr)
    df['combined_descr'] = np.where(df.combined_descr.isin(['light intensity shower rain', 'light rain', 'Drizzle', 'sleet']), 'light rain/ drizzle', 
                                  (np.where(df.combined_descr.isin(['moderate rain', 'shower rain' , 'heavy intensity rain', 'heavy intensity shower rain',
                                     'light snow', 'thunderstorm', 'thunderstorm with rain', 'light shower snow', 'thunderstorm with light rain']), 
                                            'mod+heavy rain/ snow/ storm', df.combined_descr)))
    return df


def get_daily_mean(df, row):
    """Calculate the daily mean rain"""
    day_df = df.loc[df.Datetime.dt.date == row.Datetime.date()]
    if len(day_df[day_df.rain>0]) > 0:
        mean_rain = day_df.rain.mean()
    else:
        month_df = df.loc[df.Datetime.dt.month == row.Datetime.month]
        mean_rain = month_df.rain.mean()
    return mean_rain

def prep_for_analysis(df):
    """Performs further processing on data frame once weather added ahead of analysis
    Adds darkness, drops irrelevant columns, tidies the description columns, adds combined_descr, calculate daily mean, add an hour column"""
    df['darkness'] = df.apply(lambda row: get_darkness(row), axis = 1)  
    df = df.drop(['index', 'timestamp', 'month_year', 'day_site', 'Latitude', 'Longitude', 
                 'py_datetime', 'Timestamp'], axis = 1)
    df['rain'] = df.rain.apply(lambda x: list(x.values())[0]if x else None)
    df['main_descr'] = df.apply(lambda row: tidy_main(row), axis = 1)
    df['detail_descr'] = df.apply(lambda row: tidy_detail(row), axis = 1)
    df = add_combined(df)
    df['visibility_tidied'] = df.visibility_tidied.apply(lambda x: round(x, -3))
    df['rain_adj'] = df.apply(lambda row: 0 if row.combined_descr not in ['light rain/ drizzle', 'mod+heavy rain/ snow/ storm'] else get_daily_mean(row), axis =1)
    df['hour'] = df.Datetime.dt.hour
    return df

