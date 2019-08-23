

```python
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np
```

    C:\Users\Sushant Lenka\.conda\envs\machineLearning\lib\importlib\_bootstrap.py:219: RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 192 from C header, got 216 from PyObject
      return f(*args, **kwds)
    C:\Users\Sushant Lenka\.conda\envs\machineLearning\lib\importlib\_bootstrap.py:219: RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 192 from C header, got 216 from PyObject
      return f(*args, **kwds)
    


```python
boston_dataset=load_boston()
```


```python
data=pd.DataFrame(data=boston_dataset.data,columns=boston_dataset.feature_names)

features=data.drop(['INDUS','AGE'],axis=1)
```


```python
features.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
    </tr>
  </tbody>
</table>
</div>




```python
log_prices=np.log(boston_dataset.target)
target=pd.DataFrame(log_prices,columns=['PRICE'])
target.shape
```




    (506, 1)




```python
CRIM_IDX=0
ZN_IDX=1
CHAS_IDX=2
RM_IDX=4
PT_RATIO=8


property_stats=np.ndarray(shape=[1,11])
```


```python
property_stats[0][CRIM_IDX]=features['CRIM'].mean()

```


```python
property_stats=features.mean().values.reshape(1,11)
```


```python
regr=LinearRegression().fit(features,target)
fitted_vals=regr.predict(features)

mse=mean_squared_error(target,fitted_vals)
rmse=np.sqrt(mse)


```


```python
rmse

```




    0.18751213519713034




```python
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

```


```python
get_log_estimate(3,20)
```




    (array([[2.67160974]]), array([[3.04663401]]), array([[2.29658547]]), 95)




```python
rmse
```




    0.18751213519713034




```python
np.median(boston_dataset.target)
```




    21.2




```python
ZILLOW_MEDIAN_PRICE=583.1
SCALE_FACTOR=ZILLOW_MEDIAN_PRICE/np.median(boston_dataset.target)
```


```python
log_est,upper,lower,conf=get_log_estimate(9,15)

# Taking inflation

dollar_est=(np.e**log_est*1000*SCALE_FACTOR)[0][0]
dollar_high=(np.e**upper*1000*SCALE_FACTOR)[0][0]
dollar_low=(np.e**lower*1000*SCALE_FACTOR)[0][0]
```


```python
dollar_est
```




    826445.1232418079




```python
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
```


```python
get_dollar_estimate(1,20)
```

    Dollar est:  331829.63061220833
    Upper bound:  482820.9817700748
    Lower bound: 228057.4123944571
    Percentage confidence:  95
    


```python
import boston_valuation as val
```


```python
val.get_dollar_estimate(6,12,True)
```

    Dollar est:  782537.0719624956
    Upper bound:  1138612.3555613281
    Lower bound: 537816.2866445839
    Percentage confidence:  95
    


```python

```
