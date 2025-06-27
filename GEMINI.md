# stock machine learning portfolio pipeline
## data source 
- use DB to get stock and index data
- mostly they are KOSPI stocks

## preprocessing
- bring raw data from database
- create raw factors for stocks
- preprocessing those factor data


## learning
- 1month future return of stocks
- NO time series prediction. DO crosss sectional prediction
- do feature engineering to get better result
- WARNING Lookahead bias

## models
- use simple machine learning models to deep learning models
- but deep learning models should be simple and not big because of computing resources


## portfolio
- construct portfolio based on the benchmark weights 
- it should consider 3 constraints, individual stock weight, sector weight and size(large cap stock, middle cap, and small cap)


## additional comments 
- there are already codes for most of the parts, preprocessing, learning and portfolio
- they are in archive folder as .ipynb 
- but they may not be efficient enough so you do not need to follow all the contents in those files
- i just want simple, efficient, direct way to implement all those processes

