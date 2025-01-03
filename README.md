# Weather conditions and cycling

## Background
This repo was created as part of my Masters coursework to investigate the relationship between weather conditions and cycling levels in London. I took cycling data from a series of cycle count files published by TfL. To this, I added weather data for the given time and location of a count using a REST API.   After extensive data processing,  feature engineering and EDA I modelled these relationships using linear regression and k-means clustering.


## File information

For my coursework assignment, I was required to submit all code in a single Jupyter notebook with extensive comments. This is **Complete_notebook.ipynb**. It contains step-by-step descriptions of the entire analytical process including data aquisition and linking, data transformation/ feature engineering, EDA including data visualisations, sampling and statistical analyses.  

This notebook is highly informative and detailed but also long. As a streamlined alternative, I have created **Analysis_notebook.ipynb** which starts with a clean sample file including weather data and then runs the predictive models.  In this notebook, functions are stripped out and imported from separate scripts: **processing.py** includes functions with explanations for all data processing, cleaning and sampling on the cycling data. **weather_funcs.py** covers the extraction of weather data from the API, linking and wrangling this with cycling data, and further feature engineering  of weather data for richer analysis. **analysis_funcs.py** includes functions for more efficient repeated calls for regression models. 
