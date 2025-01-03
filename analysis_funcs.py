import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_validate, cross_val_score
import pandas as pd

def standardise(df, predictors):
    """Build a function that standardises predictors as recommended in literature (i.e. scales some, use cube root of rain, square root of windspeed)
        Parameters:
            df: the dataframe to standardise
            predictors: a list of variables within the dataframe to standardise. The function determines how to treat each predictor
        Returns:
            df: an updated df with standardised predictors"""
    if 'wind_speed' in predictors:
        df['wind_speeed'] = np.sqrt(df.wind_speed)
        wind_scale = StandardScaler()
        df['wind_speed'] = wind_scale.fit_transform(np.reshape(df.wind_speed, (-1,1)))
    if 'temp' in predictors:
        temp_scale = StandardScaler()
        df['temp'] = temp_scale.fit_transform(np.reshape(df.temp, (-1,1)))
    if 'rain_adj' in predictors:
        df['rain_adj'] = df.rain_adj ** (1/3)
        rain_scale = MinMaxScaler()
        df['rain_adj'] = rain_scale.fit_transform(np.reshape(df.rain_adj, (-1,1)))
    if 'tidied_clouds' in predictors:
        cloud_scale = MinMaxScaler()
        df['tidied_clouds'] = cloud_scale.fit_transform(np.reshape(df.tidied_clouds, (-1,1)))
    if 'visibility_tidied' in predictors:
        vis_scale = MinMaxScaler()
        df['visibility_tidied'] = vis_scale.fit_transform(np.reshape(df.visibility_tidied, (-1,1)))
    return df

def do_regression(df, predictors, model, cat_vars = None):
    """Allows lots of regression models to be run with a few lines of code
        Parameters:
            df: the input pandas dataframe
            predictors: the predictor variables of the regression model (the dependent variable is always the cycling count)
            model: the type of regression model (linear, Lasso, Ridge)
            cat_vars: categorical variables among the predictors which are converted to dummies"""
    df = standardise(df, predictors) 
    df = pd.get_dummies(df, columns = cat_vars, dtype = int)
    for var in cat_vars:
        cat_labels = [col for col in df.columns if var in col]
        predictors += predictors + cat_labels
    predictors = list(set(predictors).difference(set(cat_vars)))
    nulls = [] 
    for predictor in predictors:
        if df[predictor].isnull().sum() > 0.01 * len(df):
            nulls.append(predictor)
    if len(nulls) > 0:
        print(f"Too many null values in {nulls}, remove/ fill them or choose different predictors")
    else:
        clean = df[predictors + ['Count']].dropna()
        X = clean[predictors]
        y = clean.Count
        score = cross_val_score(model, X, y, scoring = 'r2')
        return score