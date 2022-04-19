import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model 

oecd_bil = pd.read_csv("oecd_bil_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv("gdp_per_captia.csv", thousands=',', delimiter='\t', encoding='latin1', na_values="n/a")


