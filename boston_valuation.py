from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

boston_dataset=load_boston()
data=pd.DataFrame(data=boston_dataset.data,columns=boston_dataset.feature_names)

features=data.drop(['INDUS','AGE'],axis=1)
log_prices=np.log(boston_dataset.target)
target=pd.DataFrame(log_prices,columns=['PRICE'])

CRIM_IDX=0
ZN_IDX=1
CHAS_IDX=2
RM_IDX=4
PT_RATIO=8


property_stats=np.ndarray(shape=[1,11])
property_stats=features.mean().values.reshape(1,11)

regr=LinearRegression().fit(features,target)
fitted_vals=regr.predict(features)

mse=mean_squared_error(target,fitted_vals)
rmse=np.sqrt(mse)
def get_log_estimate(nr_rooms,students_per_class,next_to_river=False,high_confidence=True):
    property_stats[0][RM_IDX]=nr_rooms
    property_stats[0][PT_RATIO]=students_per_class
    
    if next_to_river:
        property_stats[0][CHAS_IDX]=1
    else:
        property_stats[0][CHAS_IDX]=0
        
    
    log_estimate=regr.predict(property_stats)
    
    if high_confidence:
        upper_bound=log_estimate+2*rmse
        lower_bound=log_estimate-2*rmse
        interval=95
        
    else:
        upper_bound=log_estimate+rmse
        lower_bound=log_estimate-rmse
        interval=68
        
    return log_estimate,upper_bound,lower_bound,interval
ZILLOW_MEDIAN_PRICE=583.1
SCALE_FACTOR=ZILLOW_MEDIAN_PRICE/np.median(boston_dataset.target)

def get_dollar_estimate(rm,ptr,chas=False,large_range=True):
    """ Estimate the price of a property in boston.
    Keyword Arguments:
        rm-- No. of rooms
        ptr-- No. of students per teacher in a class 
        chas-- True if property is close to a river
        large_range--True for 95% confidence
    
    """
    
    if rm<1 or ptr<1:
        print('Invalid rooms')
        return
    
    log_est,upper,lower,conf=get_log_estimate(rm,ptr,chas,large_range)

    dollar_est=(np.e**log_est*1000*SCALE_FACTOR)[0][0]
    dollar_high=(np.e**upper*1000*SCALE_FACTOR)[0][0]
    dollar_low=(np.e**lower*1000*SCALE_FACTOR)[0][0]
    
    print('Dollar est: ',dollar_est)
    print('Upper bound: ',dollar_high)
    print('Lower bound:',dollar_low)
    print('Percentage confidence: ',conf)