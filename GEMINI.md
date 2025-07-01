# stock machine learning portfolio pipeline
## data source 
- use DB to get stock and index data
- in database,  dates, symbol, name columns are common columns for all the tables and the rest features are different by each table
- sometimes it is necessary to pivot or unstack to use original database
- however index data is in timeseries data format, index is datetime and columns is the name of the indicies


## preprocessing
- bring raw data from database
- create raw factors for stocks
- initially there are 3 groups of factors and each group calculates factors slightly differently but need to be the same in the end
- preprocessing those factor data like normalizing, winsorization


## learning
- create 1month future returns of stocks to use it as target to train prediction models
- NO time series prediction. DO crosss sectional prediction which means input data should have multi index , datetime and symbol.
- do feature engineering to get better result
- WARNING split train and test set carefully to prevent Lookahead bias

## models
- use simple machine learning models such as regression models, lasso and ridge to deep learning models such as dnn and simple transformer model
- but deep learning models should be simple and not big because of limited computing resources


## portfolio
- construct portfolio based on the benchmark weights 
- it should consider 3 constraints, individual stock weight, sector weight and size(large cap stock, middle cap, and small cap)


## additional comments 
- there are already codes for most of the parts, preprocessing, learning and portfolio
- they are in archive folder as .ipynb 
- but they may not be efficient enough so you do not need to follow all the contents in those files
- i just want simple, efficient, direct way to implement all those processes
- never do random number sample

