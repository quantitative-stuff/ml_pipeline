# Python을 활용한 Dynamic Quant 모델링
#
# 기존 엑셀 파일 Process
# Factor 수익률 구하기
#     - 삼성전자 코스피 비중 구하기
#     - 일자와 consensus를 비롯한 관련 연도 및 월 입력
#     - VBA 실행
#         1. 일자 데이터 copy 해서 dataguide 셀에 paste
#         2. dataguide 업데이트
#         3. 종목의 factor score 구하기(함수로 연동)
#         4. factor별 상위 20개 종목의 평균 수익률 구하기
# 
# Python Process
# 팩터 수익률 구하기
# QuantDBconnect -> QuantDB 연결
# GetDataTable(KOSPI200종목, 일자) 
# -> 일자에 해당하는 코스피200 종목의 데이터 불러오기
# GetRawData(Table, 일자, KOSPI200종목, 다이나믹 포트 유무) 
# -> 일자에 해당하는 코스피200 종목의 데이터를 가공하여 Sales, Cap, Sector 등등의 정보 계산 및 조회
# GetFactor(Table, 일자, KOSPI200종목, 삼성전자 코스피 비중, 다이나믹 포트 유무) 
# ->  일자에 해당하는 코스피200 종목의 RawData와 관련 데이터를 활용하여 factor score 구하기
# 
# 다이나믹 포트 폴리오 구하기
# Regime -> Regime 파일 읽어서 빈 일자 사이의 값을 채우고 해당 일자의 국면 값 불러오기
# ESG -> ESG 파일 읽고 관련 데이터 불러오기
# FactorWeight(일자, 팩터수익률, 선택 팩터, KOSPI200종목) -> 선택된 factor들과 Reigme을 활용하여 팩터 가중치 구하기
# DynamicPortFolio(일자, 팩터수익률, 선택 팩터, KOSPI200종목) -> 다이나믹 포트폴리오 구하기

# GetSales 수정, GetNP 수정
# GetAllRatio 수정

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from tqdm import tqdm
import cx_Oracle
import sqlite3
from collections import defaultdict
from scipy.optimize import linprog


def CallEntireData(DIR):
    """
    This function loads and returns all the required data from the pickle files.

    Parameters:
    None

    Returns:
    dict: A dictionary containing the loaded data, where the keys are the names of the data files and the values are the corresponding DataFrames.
    """

    # Initialize an empty dictionary to store the DataFrames
    DataFrames = {}

    # Define a list of the names of the data files to be loaded
    lists = ['AdjPrc', 'AdjPrc_High_60D', 'MktCap', 'ListedShares', 'Sales_TQ_ic', 'Sales_LQ_ic', 'Sales_2LQ_ic', 'Sales_3LQ_ic', 'OI_TQ_ic', 'OI_LQ_ic', 'OI_2LQ_ic', 'OI_3LQ_ic', 'CIE_TQ_ic', 'CIE_LQ_ic', 'CIE_2LQ_ic', 'CIE_3LQ_ic', 'NICI_TQ_ic', 'NICI_LQ_ic', 'NICI_2LQ_ic', 'NICI_3LQ_ic', 'OI_E3_ic', 'OI_E3_NextYear_ic', 'OI_E3_2yr_ic', 'NIP_E3_ic', 'NIP_E3_NextYear_ic', 'KLCAIndustry', 'NetVol_Inst_20D', 'Dep_Amort_TQ_fs1', 'Dep_Amort_LQ_fs1', 'Dep_Amort_2LQ_fs1', 'Dep_Amort_3LQ_fs1', 'Int_Inc_TQ_fs1', 'Int_Inc_LQ_fs1', 'Int_Inc_2LQ_fs1', 'Int_Inc_3LQ_fs1', 'Int_Exp_TQ_fs1', 'Int_Exp_LQ_fs1', 'Int_Exp_2LQ_fs1', 'Int_Exp_3LQ_fs1', 'Div_fs1', 'Assets_TQ_fs1', 'Assets_LQ_fs1', 'Assets_2LQ_fs1', 'Assets_3LQ_fs1', 'TA_TQ_fs1', 'TA_LQ_fs1', 'TA_2LQ_fs1', 'TA_3LQ_fs1']

    # Iterate through the list of data file names
    for name in lists:
        # Load the data from the pickle file and store it in the DataFrames dictionary
        df_temp = pd.read_pickle(DIR+name+'.pkl')
        DataFrames[name] = df_temp

    # Return the dictionary containing the loaded data
    return DataFrames
    
def GetBusinessDayAfter(DatesDataFrame, Date):
    '''
    This function returns the nearest business day after the given date.

    Parameters:
    DatesDataFrame (pd.DataFrame): DataFrame containing the dates of interest.
    Date (numpy.datetime64): Date for which the nearest later business day is to be found.

    Returns:
    numpy.datetime64: The nearest later business day.

    Raises:
    ValueError: If the given date is not found in the DataFrame.
    '''
    Day = DatesDataFrame[(DatesDataFrame >= Date) & (DatesDataFrame <= Date + np.timedelta64(20, 'D'))]
    if len(Day) == 0:
        raise ValueError("The given date is not found in the DataFrame.")
    return Day[0]

def GetBusinessDayBefore(DatesDataFrame, Date):
    """
    This function returns the nearest business day before the given date.

    Parameters:
    DatesDataFrame (pd.DataFrame): DataFrame containing the dates of interest.
    Date (numpy.datetime64): The date for which the nearest business day before is to be found.

    Returns:
    numpy.datetime64: The nearest business day before the given date.

    Raises:
    ValueError: If the given date is not found in the DataFrame.
    """
    Day = DatesDataFrame[(DatesDataFrame <= Date) & (DatesDataFrame > Date - np.timedelta64(20, 'D'))]
    if len(Day) == 0:
        raise ValueError("The given date is not found in the DataFrame.")
    return Day[-1]

def GetBusinessDay(DatesDataFrame, Date, dd):
    """
    This function returns the nearest business day from the given date.

    Parameters:
    DatesDataFrame (pd.DataFrame): DataFrame containing the dates of interest.
    Date (numpy.datetime64): The date for which the nearest business day is to be found.
    dd (int): The number of business days to look ahead or behind.

    Returns:
    numpy.datetime64: The nearest business day to the given date.

    Raises:
    ValueError: If the given date is not found in the DataFrame.
    """
    Day = DatesDataFrame[np.where(DatesDataFrame==Date)[0]+dd][0]
    return Day

def GetEndOfMonth(DatesDF):
    '''
    This function takes a DataFrame containing dates and returns a list of dates that fall on the last day of a month.

    Parameters:
    DatesDF (pandas DataFrame): A DataFrame containing dates.

    Returns:
    list: A list of dates that fall on the last day of a month.
    '''
    months = []
    for val in DatesDF.values.flatten():
        val = pd.to_datetime(val)
        val = val.strftime('%Y-%m-%d')
        month = int(val[5:7]) % 13
        months.append(month)

    EndofMonth = DatesDF.copy(deep=True)
    EndofMonth['Month'] = months
    EndofMonth['Month'] = EndofMonth['Month'].diff(-1).fillna(-1)
    EndofMonth = EndofMonth[EndofMonth['Month'] != 0][['Dates']].values.flatten()
    return EndofMonth

def GetKOSPI200AtD(kospi200, date):
    """
    Get KOSPI200 data at a specific date.

    Parameters
    ----------
    kospi200 : pandas DataFrame
        DataFrame containing KOSPI200 data.
    date : YYYY-MM-DD
        The date for which the data is required.

    Returns
    -------
    kospi200atD : numpy ndarray
        A 2D numpy array containing the symbols and names of the companies in KOSPI200 at the specified date.

    Raises
    ------
    ValueError
        If the specified date does not exist in the data.
    """
    # kospi200atD = kospi200[kospi200['Dates'] == date][['Symbol','Name']].values
    kospi200atD = kospi200[kospi200['Dates'] == date]['Symbol'].values
    return kospi200atD

def GetKOSPI(Date, DB):
    '''
    This function retrieves the KOSPI data from the database.

    Parameters:
    Date (str): The date for which the KOSPI data is required.

    Returns:
    pd.DataFrame: A DataFrame containing the KOSPI data for the given date.

    Raises:
    Exception: If the date is not found in the database.
    '''
    Day = pd.to_datetime(Date)
    Day = Day.strftime('%Y-%m-%d')
    fields = "Symbol, Name, MktCap"
    table = "SSC_table_Bak"
    conditions = "Dates = '{date}'".format(date = Day)
    SQL = (f"SELECT {fields} "
        f"FROM {table} "
        f"WHERE {conditions};")
    KOSPI = DB.Query(SQL)
    return KOSPI


class QuantDBconnect:
    """
    QuantDBconnect class provides a connection to the QuantDB database.

    Attributes:
        host (str): The host address of the database.
        port (str): The port number of the database.
        username (str): The username for accessing the database.
        password (str): The password for accessing the database.
        db_name (str): The name of the database.

    Methods:
        connect(self):
            Connects to the QuantDB database using the provided credentials.

    Returns:
        None

    Raises:
        Exception: If there is an error in connecting to the database.
    """

    def __init__(self) -> None:
        
        self.host = '192.168.1.27'
        self.port = '3306'
        self.db_name = 'quantdb_maria'
        self.username = 'quantdb'
        self.password = 'QuantDb2023!'
        self.ssctable = 'fn_SSC'
        self.comtable = 'fn_COM'

    def connect(self):
        self.engine = create_engine("mysql+pymysql://" + self.username + ":" + self.password + "@" + self.host + ":" + self.port + "/" + self.db_name)
        self.conn = self.engine.connect()

    def Query(self, SQL):
        self.connect()
        temp_query = text(SQL)
        QueryDataFrame = pd.read_sql(temp_query, self.conn)
        return QueryDataFrame

    def GetTradingDate(self):
        self.connect()
        SQL = "SELECT UNIQUE(Dates) FROM " + self.ssctable
        temp_query = text(SQL)
        TradingDates = pd.read_sql(temp_query, self.conn)
        return TradingDates

    def GetAllKOSPI200(self):
        SQL = "SELECT Dates, Symbol, Name from fn_COM WHERE KOSPI200YN = 'Y'"
        self.kospi200 = self.Query(SQL)
        self.kospi200 = self.kospi200.drop_duplicates()
        return self.kospi200


def CallData2(Name, DataFrameDict, Date, kospi200atD):
    '''
    This function is used to call the data from the pickle file.

    Parameters:
    - name (str): The name of the data file to be called.
    - date (str, optional): The date for which the data is to be called. If not provided, the current date is used.

    Returns:
    - df_temp (pandas DataFrame): A DataFrame containing the data for the given name and date.

    Raises:
    - Exception: If the data file for the given name does not exist or if the date is not found in the data file.
    '''
    # kospi200atD = GetKOSPI200AtD(kospi200, date)
    dfTemp = DataFrameDict[Name].set_index('Dates').loc[Date,:].reset_index()
    dfTemp = dfTemp[(dfTemp['Symbol'].isin(kospi200atD))].drop_duplicates()

    if len(np.array(Date).flatten()) > 1:
        return dfTemp
    return dfTemp.set_index(['Symbol', 'Name']).drop('Dates', axis=1)

def MakeMultiindex(Accounting, DateName):
    '''
    This function creates a MultiIndex object from two input arrays.

    Parameters:
    - accounting (str): The accounting item name, such as "Sales" or "EPS".
    - date_name (str): The date name, such as "Current" or "NextYear".

    Returns:
    - index (pd.MultiIndex): A MultiIndex object with the provided accounting and date names.
    '''
    tuples = list(zip(*[[Accounting],[DateName]]))
    index = pd.MultiIndex.from_tuples(tuples, names=["name", "date"])
    return index
    
def QuaterNameconverter(KeyName):
    """
    This function takes a KeyName as input and returns the corresponding quarter name.

    Parameters:
    - KeyName (str): The input string containing the KeyName.

    Returns:
    - QuarterName (str): The corresponding quarter name based on the input KeyName.

    The function checks the input KeyName and returns the appropriate quarter name. If the input KeyName contains '_TQ', it returns '1QBefore'. If it contains '_LQ', it returns '2QBefore'. If it contains '_2LQ', it returns '3QBefore'. If it contains '_3LQ', it returns '4QBefore'. If none of these conditions are met, it returns 'Current'.
    """
    if '_TQ' in KeyName:
        QuarterName = '1QBefore'
    elif '_LQ' in KeyName:
        QuarterName = '2QBefore'
    elif '_2LQ' in KeyName:
        QuarterName = '3QBefore'
    elif '_3LQ' in KeyName:
        QuarterName = '4QBefore'
    else:
        QuarterName = 'Current'

    return QuarterName

def Get4Dates(DateDiff, DatesList, Date):
    """
    This function generates a list of dates based on the input date and the given date differences.

    Returns:
    list: A list of dates.
    """
    date_list = [Date]
    for dd in DateDiff:
        d_temp = Date + np.timedelta64(dd,'D')
        if dd > 0:
            d_temp = GetBusinessDayAfter(DatesList, d_temp)
        else:
            d_temp = GetBusinessDayBefore(DatesList, d_temp)
        date_list.append(d_temp)
    return date_list
    
def GetSSCTable(DataFrameDict, DatesList, Date, Dynamicportfolio, kospi200atD):
    """
    This function retrieves the SSCTable data from the QuantDB.

    Parameters:
    DynamicPortfolio (bool, optional): If True, it will consider the DynamicPortfolio data.

    Returns:
    pandas.DataFrame: A DataFrame containing the SSCTable data.

    The function retrieves the SSCTable data from the QuantDB database. It first calculates the dates for the next day, 60 days before the next year, and 180 days before the next year. It then retrieves the AdjPrc data for these dates and constructs a DataFrame with the retrieved data. The function then constructs a MultiIndex object from two input arrays and sets the DataFrame's columns to this MultiIndex object. Finally, the function returns the DataFrame containing the SSCTable data.
    """
    Datediff = [+1, -60, -180]

    if Dynamicportfolio:
        Datediff = [0, -60, -180]     

    DateList = Get4Dates(Datediff, DatesList, Date)
    dfAdjPrc = CallData2('AdjPrc', DataFrameDict, DateList, kospi200atD).pivot(index=['Symbol', 'Name'], columns='Dates', values='AdjPrc')

    DateName = ['180DayBefore', '60DayBefore', 'Current', '1DayAfter']
    indexAdjPrc = pd.MultiIndex.from_tuples([('수정주가(원)', name) for name in DateName], names=["name", "date"])
    dfAdjPrc.columns = indexAdjPrc

    metrics = {
        'AdjPrc_High_60D': '수정주가 (60일 최고)(원)',
        'MktCap': '시가총액 (티커-상장예정주식수 포함)(백만원)',
        'ListedShares': '상장주식수(주)'
    }

    data_frames = {'AdjPrc': dfAdjPrc}

    # Vectorized fetching and column setting
    for key, column_name in metrics.items():
        df = CallData2(key, DataFrameDict, Date, kospi200atD)
        index = MakeMultiindex(column_name, 'Current')
        df.columns = index
        data_frames[key] = df

    # Combine all data frames into a single data frame
    SSCTable = pd.concat(data_frames.values(), axis=1)

    return SSCTable
        
def GetIFRSTable(DataFrameDict, Date, kospi200atD):
    """
    Retrieves the IFRS data from the QuantDB database.

    Parameters:
    None

    Returns:
    pandas.DataFrame: A DataFrame containing the IFRS data.

    Raises:
    Exception: If the data retrieval fails.

    This function retrieves the IFRS data from the QuantDB database. It first checks if the data for the given IFRS item already exists in the cache. If not, it retrieves the data from the database and stores it in the cache. The function then constructs a DataFrame with the retrieved data and returns it.
    """
    metrics = {
        'Sales': ['Sales_TQ_ic', 'Sales_LQ_ic', 'Sales_2LQ_ic', 'Sales_3LQ_ic', '매출액(천원)_ic'],
        'OI': ['OI_TQ_ic', 'OI_LQ_ic', 'OI_2LQ_ic', 'OI_3LQ_ic', '영업이익(천원)_ic'],
        'CIE': ['CIE_TQ_ic', 'CIE_LQ_ic', 'CIE_2LQ_ic', 'CIE_3LQ_ic', '지배주주지분(천원)_ic'],
        'NICI': ['NICI_TQ_ic', 'NICI_LQ_ic', 'NICI_2LQ_ic', 'NICI_3LQ_ic', '지배주주순이익(천원)_ic']
    }

    data_frames = {}

    for metric, details in metrics.items():
        data_types = details[:-1]
        column_name = details[-1]

        for data_type in data_types:
            df = CallData2(data_type, DataFrameDict, Date, kospi200atD)
            quarter_name = QuaterNameconverter(data_type)
            index = MakeMultiindex(column_name, quarter_name)
            df.columns = index
            data_frames[data_type] = df

    # Combine all data frames into a single data frame
    IFRSTable = pd.concat(data_frames.values(), axis=1)

    return IFRSTable
    
def GetCONTable(DataFrameDict, DatesList, Date, kospi200atD):
    """
    This function retrieves the CON table data from the QuantDB database.

    Parameters
    ----------
    None

    Returns
    -------
    pandas.DataFrame: A DataFrame containing the CON table data.

    Raises
    ------
    Exception: If the data cannot be retrieved from the database.

    Notes
    -----
    The function first retrieves the CON table data for the current date. 
    It then retrieves the data for the next year and 60 days before the next year.
    The data is then concatenated and returned as a DataFrame.
    """
    metrics = {
        'CIE': [
            ('OI_E3_ic', '영업이익 (E3)(억원)', 'Current'),
            ('OI_E3_NextYear_ic', '영업이익 (E3)(억원)', 'NextYear'),
            ('OI_E3_2yr_ic', '영업이익 (E3)(억원)', 'TwoYear'),
            ('OI_E3_ic', '영업이익 (E3)(억원)', '60DayBefore'),
            ('OI_E3_NextYear_ic', '영업이익 (E3)(억원)', '60DayBefore NextYear'),
            ('OI_E3_2yr_ic', '영업이익 (E3)(억원)', '60DayBefore TwoYear'),
        ],
        'NICI': [
            ('NIP_E3_ic', '지배주주귀속순이익 (E3)(억원)', 'Current'),
            ('NIP_E3_NextYear_ic', '지배주주귀속순이익 (E3)(억원)', 'NextYear'),
        ]
    }

    data_frames = {}

    for metric, details in metrics.items():
        for data_type, column_name, period in details:
            if '60DayBefore' in period:
                dd = Date + np.timedelta64(-60, 'D')
                dd = GetBusinessDayBefore(DatesList, dd)
                df = CallData2(data_type, DataFrameDict, dd, kospi200atD)
            else:
                df = CallData2(data_type, DataFrameDict, Date, kospi200atD)
            index = MakeMultiindex(column_name, period)
            df.columns = index
            data_frames[(data_type, period)] = df/10**5

    # If needed, combine all data frames into a single data frame
    CONTable = pd.concat(data_frames.values(), axis=1)

    return CONTable


def GetCOMTable(DataFrameDict, Date, kospi200atD):
    """
    This function retrieves the COM table data from the QuantDB.

    Parameters:
    None

    Returns:
    pandas.DataFrame: A DataFrame containing the COM table data.

    Raises:
    Exception: If the data for the specified key is not found in the QuantDB.

    This function first identifies the COMToFind in the QuantDB. 
    Then, it calls the CallData function to retrieve the data for each COMToFind.
    Finally, it creates a MultiIndex for the DataFrame columns and returns the DataFrame.
    """
    # COMToFind = ['업종구분 (KLCA)']
    metrics = {
        'KLCAIndustry': '업종구분 (KLCA)'
    }

    data_frames = { }

    for key, column_name in metrics.items():
        df = CallData2(key, DataFrameDict, Date, kospi200atD)
        index = MakeMultiindex(column_name, 'Current')
        df.columns = index
        data_frames[key] = df

    COMTable = pd.concat(data_frames.values(), axis=1)

    return COMTable

def GetCIATable(DataFrameDict, Date, kospi200atD):
    """
    This function retrieves the CIATable from the QuantDB.

    Parameters
    ----------
    None

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the CIATable data.

    Raises
    ------
    None

    This function first identifies the CIAToFind in the QuantDB. 
    Then, it calls the CallData function to retrieve the data for each CIAToFind. 
    Finally, it creates a MultiIndex for the DataFrame columns and returns the DataFrame.
    """
    metrics = {
        'NetVol_Inst_20D': '순매수수량(기관계)(20일합산)(주)'
    }

    data_frames = {}

    for key, column_name in metrics.items():
        df = CallData2(key, DataFrameDict, Date, kospi200atD)
        index = MakeMultiindex(column_name, 'Current')
        df.columns = index
        data_frames[key] = df
                
    CIATable = pd.concat(data_frames.values(), axis=1)

    return CIATable
        
def GetFS1Table(DataFrameDict, Date, kospi200atD):
    """
    This function retrieves the FS1 data from the Quantopian database.

    Parameters
    ----------
    None

    Returns
    -------
    DataFrame: A DataFrame containing the FS1 data.

    Raises
    ------
    None

    This function first defines a list of keys to search for in the Quantopian database. It then iterates through this list, calling the `CallData` function for each key. If the key matches the first key in the list, it creates a new DataFrame and sets its MultiIndex columns based on the key and the corresponding quarter name. If the key does not match the first key, it appends the returned DataFrame to the existing DataFrame.

    The function returns the concatenated DataFrame containing all the FS1 data.
    """
    
    metrics = {
        'Dep_Amort': ['Dep_Amort_TQ_fs1', 'Dep_Amort_LQ_fs1', 'Dep_Amort_2LQ_fs1', 'Dep_Amort_3LQ_fs1', '감가상각비(천원)_fs1'],
        'Int_Inc': ['Int_Inc_TQ_fs1', 'Int_Inc_LQ_fs1', 'Int_Inc_2LQ_fs1', 'Int_Inc_3LQ_fs1', '이자수익(천원)_fs1'],
        'Int_Exp': ['Int_Exp_TQ_fs1', 'Int_Exp_LQ_fs1', 'Int_Exp_2LQ_fs1', 'Int_Exp_3LQ_fs1', '이자비용(천원)_fs1'],
        'Div': ['Div_fs1', '배당금(천원)_fs1'],
        'Assets': ['Assets_TQ_fs1', 'Assets_LQ_fs1', 'Assets_2LQ_fs1', 'Assets_3LQ_fs1', '자산총계(천원)_fs1'],
        'TA': ['TA_TQ_fs1', 'TA_LQ_fs1', 'TA_2LQ_fs1', 'TA_3LQ_fs1', '자본총계(천원)_fs1']
    }
    
    data_frames = {}

    for metric, details in metrics.items():
        data_types = details[:-1]
        column_name = details[-1]
        
        for data_type in data_types:
            df = CallData2(data_type, DataFrameDict, Date, kospi200atD)
            quarter_name = QuaterNameconverter(data_type)
            index = MakeMultiindex(column_name, quarter_name)
            df.columns = index
            data_frames[data_type] = df

    # Combine all data frames into a single data frame
    FS1Table = pd.concat(data_frames.values(), axis=1)
    
    return FS1Table

    
def GetDataTable(DataFrameDict, DatesList, Date, Dynamicportfolio, kospi200atD):
    """
    This function retrieves the DataTable from the database.

    Parameters:
    Dynamicportfolio (bool): A boolean value indicating whether the data is for a dynamic portfolio.

    Returns:
    pandas.DataFrame: A DataFrame containing the retrieved data.

    Raises:
    Exception: If there is an error in retrieving the data.

    This function first defines a list of keys to search for in the Quantopian database. It then iterates through this list, calling the `CallData` function for each key. If the key matches the first key in the list, it creates a new DataFrame and sets its MultiIndex columns based on the key and the corresponding quarter name. If the key does not match the first key, it appends the returned DataFrame to the existing DataFrame. Finally, the function returns the concatenated DataFrame containing all the FS1 data.
    """
    tables = [
        GetSSCTable(DataFrameDict, DatesList, Date, Dynamicportfolio, kospi200atD).reset_index().set_index(['Symbol','Name']),
        GetIFRSTable(DataFrameDict, Date, kospi200atD).reset_index().set_index(['Symbol','Name']),
        GetCOMTable(DataFrameDict,Date, kospi200atD).reset_index().set_index(['Symbol','Name']),
        GetCIATable(DataFrameDict,Date, kospi200atD).reset_index().set_index(['Symbol','Name']),
        GetCONTable(DataFrameDict, DatesList, Date, kospi200atD).reset_index().set_index(['Symbol','Name']),
        GetFS1Table(DataFrameDict,Date, kospi200atD).reset_index().set_index(['Symbol','Name'])
    ]

    # Concatenate all tables into a single DataFrame
    Table = pd.concat(tables, axis=1)

    return Table

def Getratio(Date, DateList):
    """
    This function calculates the ratio of the current month to the next 12 months.

    Parameters:
    Date (str): The date for which the ratio is to be calculated.
    DateList (list of str): A list of dates for which the ratio has already been calculated.

    Returns:
    float: The ratio of the current month to the next 12 months.

    Notes:
    The function first retrieves the current month from the input date. If the date is in the DateList, it calculates the ratio by adding 1 to the current month and dividing by 12. Otherwise, it calculates the ratio by taking the current month modulo 12 and dividing by 12.
    """
    val = pd.to_datetime(Date)       
    val = val.strftime('%Y-%m-%d')
    if Date in DateList:
        month = (int(val[5:7])+1) % 13
    else:
        month = int(val[5:7]) % 13
    ratio = month / 12
    return ratio

def GetAllratio(tradingDates, EoM):
    """
    This function calculates the ratio of the current month to the next 12 months.

    Parameters:
    - tradingDates (list of str): A list of dates for which the ratio has already been calculated.
    - EoM (list of str): A list of end-of-month dates.

    Returns:
    - DataFrame: A DataFrame containing the calculated ratios for each date in the input list.

    Notes:
    The function first creates a DataFrame with the input dates and their corresponding months. It then adjusts the months to account for the fact that the ratio is calculated with respect to the next 12 months. The adjusted months are calculated by adding 1 to the current month and dividing by 12. If the date is in the EoM list, the month is adjusted by subtracting 1. Finally, the function calculates the ratio by dividing the adjusted month by 12 and returns the resulting DataFrame.
    """
    df = pd.DataFrame(tradingDates, columns=['Date'])
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Adjusted_Month'] = df.apply(
        lambda row: ((row['Month'] + 1) % 13) + ((row['Month'] + 1) / 13) if row['Date'] in EoM else row['Month'] % 13,
        axis=1
    )
    df['Ratio'] = df['Adjusted_Month'] / 12
    df.drop(columns=['Month', 'Adjusted_Month'], inplace=True)
    df.set_index('Date', inplace=True)
    return df
    

def GetSales(Table):
    """
    This function calculates the sales of each stock in the dataset.

    Parameters
    ----------
    self : object
        An instance of the GetRawData class.

    Returns
    -------
    SalesTable : pandas.DataFrame
        A DataFrame containing the sales of each stock in the dataset.

    Notes
    ----
    The function first creates an empty DataFrame with the same index as the input Table.
    It then calculates the mean sales for each stock, considering only the stocks with more than 2 non-zero sales values.
    For these stocks, the mean sales value is multiplied by the difference between 4 and the count of zero sales values.
    The resulting value is then stored in the 'Sales' column of the DataFrame.
    For stocks with less than or equal to 2 non-zero sales values, the 'Sales' column is set to zero.

    The function finally returns the resulting DataFrame.
    """
    SalesTable = pd.DataFrame(index = Table.index)
    Sales = Table.loc[:, ('매출액(천원)_ic', slice(None))].fillna(0)
    
    positive_count = (Sales > 0).sum(axis=1)
    zero_count = (Sales == 0).sum(axis=1)
    
    mask = positive_count > 2
    
    #SalesTable['Sales'] = Sales.mean(axis=1) * (4 - zero_count)
    #SalesTable['Sales'] = np.where(mask, SalesTable['Sales'], 0)

    SalesTable['Sales'] = np.where(mask, Sales.sum(axis=1) / (4 - zero_count) * 4, 0)
    
    
    return SalesTable[['Sales']]

def GetNP(Table):
    """
    Calculates the Net Income Per Share (NP) for each stock in the dataset.

    Parameters
    ----------
    self : GetRawData instance
        An instance of the GetRawData class, containing the necessary data and attributes.

    Returns
    -------
    NPTable : pandas DataFrame
        A DataFrame containing the calculated NP values for each stock in the dataset.

    Notes
    ----
    The function first counts the number of non-zero values in the 'ShareHolderNetIncome' column of the input dataset. 
    It then creates a boolean mask based on whether the 'ShareHolderNetIncome' column contains any non-zero values.

    Next, the function calculates the total number of positive and negative values in the 'ShareHolderNetIncome' column. 
    It then creates another boolean mask based on whether the total number of positive and negative values is greater than 1.

    The function then calculates the mean value of the 'ShareHolderNetIncome' column for each stock in the dataset. 
    It then multiplies this mean value by the total number of positive and negative values, if the total number of positive and negative values is greater than 1.
    """
    NPTable = pd.DataFrame(index = Table.index)
    ShareHolderNetIncome = Table.loc[:, ('지배주주순이익(천원)_ic', slice(None))].fillna(0)

    NPTable['countNonzero'] = ShareHolderNetIncome[ShareHolderNetIncome != 0].count(axis = 1)

    positive_count = (ShareHolderNetIncome > 0).sum(axis=1)
    negative_count = (ShareHolderNetIncome < 0).sum(axis=1)

    mask = positive_count + negative_count > 1

    #NPTable['NP'] = ShareHolderNetIncome.mean(axis = 1)
    #NPTable['NP'] = np.where(mask, NPTable['NP'] * (positive_count + negative_count), 0)
    NPTable['NP'] = ShareHolderNetIncome.sum(axis = 1)
    NPTable['NP'] = np.where(mask, NPTable['NP'] , 0)
    del mask

    return NPTable[['NP']]

def GetEQT(Table):
    """
    This function calculates the Equity Ratio (EQT) for each stock in the dataset.

    Parameters
    ----------
    self : object
        An instance of the GetRawData class.

    Returns
    -------
    EQTTable : pandas.DataFrame
        A DataFrame containing the EQT values for each stock in the dataset.

    Notes
    -----
    The function first creates an empty DataFrame with the same index as the input Table. 
    It then calculates the mean Book value for each stock, considering only the stocks with more than 2 non-zero Book values. 
    For these stocks, the mean Book value is multiplied by the difference between 4 and the count of zero Book values. 
    The resulting value is then stored in the 'EQT' column of the DataFrame. 
    For stocks with less than or equal to 2 non-zero Book values, the 'EQT' column is set to zero.

    The function finally returns the resulting DataFrame.
    """
    EQTTable = pd.DataFrame(index = Table.index)
    Book = Table.loc[:, ('지배주주지분(천원)_ic', slice(None))]
    positive_count = (Book > 0).sum(axis=1)
    zero_count = (Book == 0).sum(axis=1)
    
    mask = positive_count > 2
    
    EQTTable['EQT'] = Book.mean(axis=1) * (4 - zero_count)
    EQTTable['EQT'] = np.where(mask, EQTTable['EQT'], 0)
    
    del mask
    del positive_count
    del zero_count

    return EQTTable[['EQT']]

def GetNP_12M(Table, ratio):
    """
    This function calculates the Net Income Per Share (NP) for the next 12 months (NP_12M)
    using the provided Net Income Attributes and the ratio of the current month to the next 12 months.

    Parameters
    ----------
    None

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the NP_12M values for each stock in the dataset.

    Raises
    ------
    Exception
        If the provided Net Income Attributes are not valid or if the ratio is not within the valid range.

    Notes
    ----
    The function first creates a DataFrame with the Net Income Attributes and the Net Income Attribute 12 months forward.
    It then calculates the NP_12M value for each stock by multiplying the current Net Income Attribute by (1 - ratio) and adding the Net Income Attribute 12 months forward multiplied by the ratio.
    The function then applies a mask to the DataFrame to filter out any stocks with invalid or missing values.
    Finally, the function returns the DataFrame containing the NP_12M values.
    """

    NP_12MTable = pd.DataFrame(index = Table.index)
    NetIncomeAttribute      = Table.loc[:, ('지배주주귀속순이익 (E3)(억원)', slice(None))].fillna(0)[[('지배주주귀속순이익 (E3)(억원)', 'Current')]]
    NetIncomeAttribute12Fwd = Table.loc[:, ('지배주주귀속순이익 (E3)(억원)', slice(None))].fillna(0)[[('지배주주귀속순이익 (E3)(억원)', 'NextYear')]]
    
    NP_12MTable['NetIncomeAttribute'] = NetIncomeAttribute
    NP_12MTable['NetIncomeAttribute12Fwd'] = NetIncomeAttribute12Fwd
    NP_12MTable['NP_12M'] = (NP_12MTable['NetIncomeAttribute'] * (1 - ratio) + NP_12MTable['NetIncomeAttribute12Fwd'] * (ratio)) * 100000
    mask = (NP_12MTable['NetIncomeAttribute'] != 0) & (NP_12MTable['NetIncomeAttribute12Fwd'] != 0)

    NP_12MTable['NP_12M'] = np.where(mask, NP_12MTable['NP_12M'], 0)
    del mask

    return NP_12MTable[['NP_12M']]

def GetDebt(Table):

    """
    This function calculates the debt-to-total-assets ratio for each stock in the dataset.

    Parameters
    ----------
    self : object
        An instance of the class containing the necessary data and methods.

    Returns
    -------
    DebtTable : pandas.DataFrame
        A DataFrame containing the debt-to-total-assets ratio for each stock in the dataset.

    Raises
    ------
    Exception
        If there is an error in retrieving the data or in the calculation process.

    Notes
    ----
    The function first extracts the 'Book' and 'Asset' values from the dataset for each stock. 
    
    It then calculates the 'DebtTotal' by subtracting the 'Book' value from the 'Asset' value. 
    
    Finally, it checks if the 'Book' and 'Asset' values are not zero, and if so, it assigns the 'DebtTotal' value to the 'DebtTotal' column of the DataFrame. 
    
    If either the 'Book' or 'Asset' value is zero, the 'DebtTotal' value is set to NaN.
    """

    DebtTable = pd.DataFrame(index = Table.index)
    Book = Table.loc[:, ('지배주주지분(천원)_ic', slice(None))]
    Asset = Table.loc[:, ('자산총계(천원)_fs1', slice(None))].fillna(0)
    
    DebtTable['Book'] = Book[[('지배주주지분(천원)_ic', '1QBefore')]]
    DebtTable['Asset'] = Asset[[('자산총계(천원)_fs1', '1QBefore')]]
    DebtTable['DebtTotal'] = DebtTable['Asset'] - DebtTable['Book']

    mask = (DebtTable['Book'] != 0) & (DebtTable['Asset'] != 0)
    DebtTable['DebtTotal'] = np.where(mask, DebtTable['DebtTotal'], np.nan)
    del mask

    return DebtTable[['DebtTotal']]

def GetOP(Table):
    """
    This function calculates the Operating Profit (OP) for each stock in the dataset.

    Parameters
    ----------
    self : object
        An instance of the class containing the dataset and other necessary attributes.

    Returns
    -------
    OPTable : pandas.DataFrame
        A DataFrame containing the calculated Operating Profit (OP) for each stock in the dataset.

    Notes
    ----
    The function first creates an empty DataFrame called OPTable with the same index as the input dataset. 
    
    It then calculates the sum of the values in the 'OP' column of the input dataset for each stock, and assigns this value to the 'OP' column of the OPTable.

    The function then calculates the number of positive and negative values in the 'OP' column of the input dataset for each stock, 
    
    and assigns these values to the 'count_positive' and 'count_negative' columns of the OPTable, respectively.

    The function then calculates the total number of positive and negative values in the 'OP' column of the input dataset for each stock, 
    
    and assigns this value to the 'total' column of the OPTable.

    Finally, the function applies a mask to the 'OP' column of the OPTable, setting its values to NaN 
    
    if the total number of positive and negative values in the 'OP' column is not equal to 4 for each stock.

    The function returns the OPTable containing the calculated Operating Profit (OP) for each stock in the dataset, with the appropriate NaN values where necessary.
    """
    OPTable = pd.DataFrame(index = Table.index)
    OP = Table.loc[:, ('영업이익(천원)_ic', slice(None))].fillna(0)
    cols = OP.columns
    OPTable['OP'] = OP.sum(axis = 1)
    
    OPTable['count_positive'] = OP[OP[cols] > 0].count(axis=1)
    OPTable['count_negative'] = OP[OP[cols] < 0].count(axis=1)

    OPTable['total'] = OPTable['count_positive'] + OPTable['count_negative']

    mask = OPTable['total'] == 4
    OPTable['OP'] = np.where(mask, OPTable['OP'], np.nan)
    del mask

    return OPTable[['OP']]

def GetOP_12M(Table, ratio):
    """
    This function calculates the Operating Profit for the next 12 months.

    Parameters
    ----------
    self : object
        An instance of the class containing the necessary attributes.

    Returns
    -------
    OP_12MTable : pandas.DataFrame
        A DataFrame containing the calculated Operating Profit for the next 12 months.

    Notes
    ----
    The function first retrieves the Operating Profit for the current year and the Operating Profit forecast for the next year.
    It then calculates the Operating Profit for the next 12 months using a weighted average of the current year's Operating Profit and the forecasted Operating Profit for the next year.
    The weights used are based on the ratio of the remaining months in the current year to the total number of months in the next 12 months.
    If either the current year's Operating Profit or the forecasted Operating Profit for the next year is zero, the function sets the calculated Operating Profit for the next 12 months to zero.
    """

    OP_12MTable = pd.DataFrame(index = Table.index)
    OperatingProfit         = Table.loc[:, ('영업이익 (E3)(억원)', slice(None))].fillna(0)[[('영업이익 (E3)(억원)', 'Current'),('영업이익 (E3)(억원)', '60DayBefore')]]
    OperatingProfit12Fwd    = Table.loc[:, ('영업이익 (E3)(억원)', slice(None))].fillna(0)[[('영업이익 (E3)(억원)', 'NextYear'),('영업이익 (E3)(억원)', '60DayBefore NextYear')]]
    
    OP_12MTable['OperatingProfit'] = OperatingProfit.loc[:, (slice(None), 'Current')]
    OP_12MTable['OperatingProfit12Fwd'] = OperatingProfit12Fwd.loc[:, (slice(None), 'NextYear')]
    OP_12MTable['OP_12M'] = (OP_12MTable['OperatingProfit'] * (1 - ratio) + OP_12MTable['OperatingProfit12Fwd'] * (ratio)) * 100000

    mask = (OP_12MTable['OperatingProfit'] != 0) & (OP_12MTable['OperatingProfit12Fwd'] != 0)
    OP_12MTable['OP_12M'] = np.where(mask, OP_12MTable['OP_12M'], 0)
    del mask

    return OP_12MTable[['OP_12M']]

def GetDep(Table):
    """
    Calculates the Depreciation for each stock in the dataset.

    Parameters
    ----------
    self : GetRawData
        An instance of GetRawData class.

    Returns
    -------
    DepTable : pandas.DataFrame
        A DataFrame containing the Depreciation values for each stock in the dataset.

    Notes
    ----
    The Depreciation is calculated by summing the 'Dep' column for each stock in the dataset.
    A boolean mask is applied to filter out the stocks with less than 4 non-zero 'Dep' values.
    If the mask is True, the 'Dep' values are retained; otherwise, they are replaced with NaN values.
    """
    DepTable = pd.DataFrame(index = Table.index)
    Dep = Table.loc[:, ('감가상각비(천원)_fs1', slice(None))].fillna(0)
    Sector = Table.loc[:, ('업종구분 (KLCA)', slice(None))]
    DepTable['Dep'] = Dep.sum(axis = 1)
    DepTable['Sector'] = Sector
    
    mask = DepTable['Sector'] == '제조업'
    mask &= (Dep.astype(bool).sum(axis=1) == 4)
    DepTable['Dep'] = np.where(mask, DepTable['Dep'], np.nan)
    del mask

    return DepTable[['Dep']]

def GetNetInterest(Table):        
    """
    This function calculates the net interest for each stock in the dataset.

    Parameters
    ----------
    self : GetRawData instance
        An instance of the GetRawData class, which contains the necessary data and attributes.

    Returns
    -------
    NetInterestTable : pandas DataFrame
        A pandas DataFrame containing the net interest for each stock in the dataset. The DataFrame has the same index as the input Table.

    Notes
    ----
    The function first creates a DataFrame called NetInterestTable with the same index as the input Table. It then calculates the net interest for each stock by subtracting the total interest expense from the total interest income. The result is stored in the 'NetInterest' column of the DataFrame.

    The function applies a boolean mask to the DataFrame to filter out stocks that do not belong to the '제조업' sector. If a stock does not belong to this sector, its 'NetInterest' value is set to NaN.

    Finally, the function returns the NetInterestTable DataFrame containing the net interest values for the stocks in the '제조업' sector.
    """
    NetInterestTable = pd.DataFrame(index = Table.index)
    Sector = Table.loc[:, ('업종구분 (KLCA)', slice(None))]
    InterestIncomeTable     = Table.loc[:, ('이자수익(천원)_fs1', slice(None))].fillna(0)
    InterestExpenseTable    = Table.loc[:, ('이자비용(천원)_fs1', slice(None))].fillna(0)  
    
    NetInterestTable['Sector'] = Sector
    NetInterestTable['NetInterest'] = InterestExpenseTable.sum(axis = 1) - InterestIncomeTable.sum(axis = 1)

    mask = NetInterestTable['Sector'] == '제조업'
    NetInterestTable['NetInterest'] = np.where(mask, NetInterestTable['NetInterest'], np.nan)
    del mask

    return NetInterestTable[['NetInterest']]

def GetEBITDA(Table):
    """
    Calculates the Earnings Before Interest, Taxes, Depreciation, and Amortization (EBITDA) for each stock in the dataset.

    Parameters
    ----------
    self : GetRawData instance
        Instance of GetRawData class containing the necessary data and attributes.

    Returns
    -------
    EBITDATable : pandas DataFrame
        DataFrame containing the calculated EBITDA values for each stock in the dataset.

    Notes
    ----
    The EBITDA is calculated using the following formula:

    EBITDA = Operating Profit + Depreciation - Net Interest

    The Operating Profit, Depreciation, and Net Interest values are obtained from the respective data tables in the dataset.

    The function first checks if the Operating Profit, Depreciation, and Net Interest values are not zero and not NaN. If these conditions are met, the EBITDA value is calculated and stored in the EBITDATable DataFrame.

    The resulting EBITDATable DataFrame contains the calculated EBITDA values for each stock in the dataset, with the index matching the original stock index in the dataset.
    """
    EBITDATable = pd.DataFrame(index = Table.index)
    EBITDATable['OP'] = GetOP(Table)
    EBITDATable['Dep'] = GetDep(Table)
    EBITDATable['NetInterest'] = GetNetInterest(Table)
    EBITDATable['EBITDA'] = EBITDATable['OP'] + EBITDATable['Dep'] - EBITDATable['NetInterest']

    mask = (EBITDATable['OP'] != 0) & (EBITDATable['Dep'] != np.nan) & (EBITDATable['NetInterest'] != np.nan)

    EBITDATable['EBITDA'] = np.where(mask, EBITDATable['EBITDA'], np.nan)
    del mask

    return EBITDATable[['EBITDA']]

def GetRawDataTable(Table, ratio):

    RawTable = {
          'Sales': GetSales(Table),
          'NP': GetNP(Table),
          'EQT': GetEQT(Table),
          'NP_12M': GetNP_12M(Table, ratio),
          'Debt': GetDebt(Table),
          'OP': GetOP(Table),
          'OP_12M': GetOP_12M(Table, ratio),
          'Dep': GetDep(Table),
          'NetInterest': GetNetInterest(Table),
          'EBITDA': GetEBITDA(Table),
        
          'DY': Table.loc[:, ('배당금(천원)_fs1', slice(None))].fillna(0),
          'Asset' : Table.loc[:, ('자산총계(천원)_fs1', slice(None))].fillna(0),
          'Book' : Table.loc[:, ('지배주주지분(천원)_ic', slice(None))],
          'InstituitionalBuy': Table.loc[:, ('순매수수량(기관계)(20일합산)(주)', slice(None))].fillna(0),
          'StockNum': Table.loc[:, ('상장주식수(주)', slice(None))].fillna(0),
          'Price': Table.loc[:, ('수정주가(원)', slice(None))].fillna(0),
          'HighestPrice': Table.loc[:, ('수정주가 (60일 최고)(원)', slice(None))].fillna(0),
          'Cap': Table.loc[:, ('시가총액 (티커-상장예정주식수 포함)(백만원)', slice(None))],
          'Sector': Table.loc[:, ('업종구분 (KLCA)', slice(None))],
          'ShareHolderNetIncome' : Table.loc[:, ('지배주주순이익(천원)_ic', slice(None))].fillna(0),
          'OperatingProfit' : Table.loc[:, ('영업이익 (E3)(억원)', slice(None))].fillna(0)[[('영업이익 (E3)(억원)', 'Current'),('영업이익 (E3)(억원)', '60DayBefore')]],
          'OperatingProfit12Fwd' : Table.loc[:, ('영업이익 (E3)(억원)', slice(None))].fillna(0)[[('영업이익 (E3)(억원)', 'NextYear'),('영업이익 (E3)(억원)', '60DayBefore NextYear')]],
          'OperatingProfit2yr' : Table.loc[:, ('영업이익 (E3)(억원)', slice(None))].fillna(0)[[('영업이익 (E3)(억원)', 'TwoYear'),('영업이익 (E3)(억원)', '60DayBefore TwoYear')]],
          'index': Table.index
          }
    return RawTable


def GetSalesP(RawTable):
    """
    This function calculates the Sales-to-Capitalization Ratio (Sales P/B) for each stock in the dataset.

    Parameters
    ----------
    None

    Returns
    -------
    DataFrame: A DataFrame containing the Sales-to-Capitalization Ratio (Sales P/B) for each stock in the dataset.

    Raises
    ------
    None
    """

    SalesPTable = pd.DataFrame(index = RawTable['index'])

    SalesPTable['Sales'] = RawTable['Sales']
    SalesPTable['Cap'] = RawTable['Cap']
    SalesPTable['Sector'] = RawTable['Sector']    

    mask = (SalesPTable['Sector'] == '제조업') & ~SalesPTable['Sales'].isna()

    SalesPTable['SalesP'] = SalesPTable['Sales'] / SalesPTable['Cap'] / 1000
    mask &= (SalesPTable['SalesP'] >= -10) & (SalesPTable['SalesP'] <= 10)
    
    SalesPTable['SalesP'] = np.where(mask, SalesPTable['SalesP'], np.nan)
    
    mask = (SalesPTable['Sales'] != 0.0)
    SalesPTable['SalesP'] = np.where(mask, SalesPTable['SalesP'], np.nan)
    del mask

    return SalesPTable[['SalesP']]

def GetBP(RawTable):
    """
    This function calculates the Book-to-Price (BP) ratio for each stock in the dataset.

    Parameters
    ----------
    self : object
        An instance of the `GetFactorScore` class.

    Returns
    -------
    BPTable : pandas.DataFrame
        A DataFrame containing the calculated BP ratio for each stock in the dataset.

    Notes
    ----
    The BP ratio is calculated by dividing the book value (equity) by the market value (capitalization). The result is then divided by 1000 to obtain a percentage value. The function applies a mask to filter out the stocks with less than 4 non-zero 'BP' values in the 'Sales', 'OP', 'Debt', and 'Dep' columns.
    """

    BPTable = pd.DataFrame(index = RawTable['index'])
    BPTable['Cap'] = RawTable['Cap']
    BPTable['Book'] = RawTable['Book'][[('지배주주지분(천원)_ic', '1QBefore')]]
    BPTable['BP'] = BPTable['Book'] / BPTable['Cap'] / 1000

    mask = BPTable['BP'] != 0
    mask &= (BPTable['BP'] >= -10) & (BPTable['BP'] <= 10)   

    BPTable['BP'] = np.where(mask, BPTable['BP'], np.nan)
    del mask

    return BPTable[['BP']]

def GetAssetP(RawTable):
    """
    This function calculates the Asset-to-Capital (AssetP) ratio for each stock in the dataset.

    Parameters
    ----------
    None

    Returns
    -------
    DataFrame
        A DataFrame containing the Asset-to-Capital (AssetP) ratio for each stock in the dataset.

    Raises
    ------
    None

    Notes
    ----
    The AssetP ratio is calculated by dividing the total assets of a company by its total capital (i.e., the sum of its equity and long-term debt). This ratio is used to assess a company's financial leverage and its ability to meet its long-term obligations.

    The function first retrieves the total assets and total capital from the dataset. It then calculates the AssetP ratio by dividing the total assets by the total capital and multiplying the result by 1000 to convert it to a percentage.

    The function applies a boolean mask to filter out stocks that are not in the '제조업' sector or have an AssetP ratio of 0. It then applies another boolean mask to filter out NaN values in the AssetP ratio.

    Finally, the function returns a DataFrame containing the AssetP ratio for each stock in the dataset that meets the specified criteria.
    """

    AssetPTable = pd.DataFrame(index = RawTable['index'])

    AssetPTable['Cap'] = RawTable['Cap']
    AssetPTable['Asset'] = RawTable['Asset'].fillna(0)[[('자산총계(천원)_fs1', '1QBefore')]]
    AssetPTable['Sector'] = RawTable['Sector']
    AssetPTable['AssetP'] = AssetPTable['Asset'] / AssetPTable['Cap'] /1000

    mask = (AssetPTable['Sector'] != '제조업') | (AssetPTable['AssetP'] == 0)
    AssetPTable['AssetP'] = np.where(mask, np.nan, AssetPTable['AssetP'])
    
    mask = (AssetPTable['AssetP'] >= -10) & (AssetPTable['AssetP'] <= 10)
    AssetPTable['AssetP'] = np.where(mask, AssetPTable['AssetP'], np.nan)
    del mask

    return AssetPTable[['AssetP']]

def GetROE(RawTable):
    """
    This function calculates the Return on Equity (ROE) for each stock in the dataset.

    Parameters
    ----------
    self : object
        An instance of the `GetRawData` class.

    Returns
    -------
    DataFrame
        A DataFrame containing the ROE values for each stock in the dataset.

    Notes
    ----
    The function first creates a DataFrame called `ROETable` and populates it with the necessary data from the `RawTable`. 
    
    It then applies a boolean mask to filter out stocks that are not in the "제조업" sector and have non-zero NP and non-zero EQT values. 
    
    The ROE is then calculated as NP divided by EQT.

    The function also applies another boolean mask to filter out ROE values that are not within the range of -10 to 10. 
    
    The function finally returns the `ROETable` DataFrame containing the ROE values for each stock in the dataset.
    """
    ROETable = pd.DataFrame(index = RawTable['index'])
    ROETable['NP'] = RawTable['NP']
    ROETable['EQT'] = RawTable['Book'][[('지배주주지분(천원)_ic', '1QBefore')]]
    ROETable['Sector'] = RawTable['Sector']
    ROETable['ROE'] = ROETable['NP'] / ROETable['EQT']

    mask = (ROETable['Sector'] == "제조업")
    mask &= (ROETable['NP'] != 0) & (ROETable['EQT'] > 0) & (ROETable['EQT'] != np.nan)
    mask &= (ROETable['ROE'] >= -10) & (ROETable['ROE'] <= 10)

    ROETable['ROE'] = np.where(mask, ROETable['ROE'], np.nan)
    del mask

    return ROETable[['ROE']]

def GetEBITDAY(RawTable):
    """
    Calculates the Earnings Before Interest, Taxes, Depreciation, and Amortization (EBITDA)
    Day (EBITDAY) for each stock in the dataset.

    Parameters
    ----------
    None

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the EBITDAY values for each stock in the dataset.

    Raises
    ------
    None

    Notes
    ----
    The EBITDAY is calculated by dividing the EBITDA by the capitalization of the company.
    The mask is applied to filter out the stocks that belong to the "제조업" sector and have non-nan EBITDAY values within the range of -10 to 10.
    """
    EBITDAYTable = pd.DataFrame(index = RawTable['index'])
    EBITDAYTable['Cap'] = RawTable['Cap']
    EBITDAYTable['Sector'] = RawTable['Sector']
    EBITDAYTable['EBITDAY'] = RawTable['EBITDA'] / 1000
    EBITDAYTable['EBITDAY'] /= EBITDAYTable['Cap']

    mask = EBITDAYTable['Sector'] == "제조업"
    mask &= EBITDAYTable['EBITDAY'] != np.nan
    mask &= (EBITDAYTable['EBITDAY'] >= -10) & (EBITDAYTable['EBITDAY'] <= 10)

    EBITDAYTable['EBITDAY'] = np.where(mask, EBITDAYTable['EBITDAY'], np.nan)
    del mask

    return EBITDAYTable[['EBITDAY']]

def GetEY_12M(RawTable):
    """
    This function calculates the Earnings Yield (EY) for each stock in the dataset.

    Parameters
    ----------
    self : object
        An instance of the `GetFactorScore` class.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the EY values for each stock in the dataset.

    Notes
    -----
    The EY is calculated as net income (profit) divided by the market value of the company (capitalization).
    The function applies a boolean mask to filter out stocks that have a negative EY or a positive EY greater than 10.
    """

    EY_12MTable = pd.DataFrame(index = RawTable['index'])
    EY_12MTable['NP_12M'] = RawTable['NP_12M']
    EY_12MTable['Cap'] = RawTable['Cap']
    EY_12MTable['EY_12M'] = EY_12MTable['NP_12M']/EY_12MTable['Cap'] * 100

    mask = (EY_12MTable['NP_12M'] != 0)
    mask &= (EY_12MTable['EY_12M'] <= 10) & (EY_12MTable['EY_12M'] >= -10)

    EY_12MTable['EY_12M'] = np.where(mask, EY_12MTable['EY_12M'], np.nan)
    del mask

    return EY_12MTable[['EY_12M']]

def GetEPSG(RawTable):
    """
    This function calculates the Earnings Per Share Growth (EPSG) for each stock in the dataset.

    Parameters
    ----------
    self : object
        An instance of the class containing the dataset and other necessary variables.

    Returns
    -------
    DataFrame
        A DataFrame containing the calculated EPSG values for each stock in the dataset.

    Notes
    ----
    The function first initializes an empty DataFrame called EPSGTable. 
    It then sets the 'Sector' column of the DataFrame to the value of the 'Sector' attribute of the self object. 
    The 'NP' and 'NP_12M' columns of the DataFrame are set to the corresponding columns of the NPTable and NP_12MTable DataFrames, respectively.

    The function then calculates the EPSG values for each stock in the dataset based on the following conditions:

    - If the stock belongs to the '제조업' sector and has non-zero 'NP' and 'NP_12M' values, 
        the EPSG value is calculated as the difference between the 'NP_12M' and 'NP' values divided by the absolute value of the 'NP' value.

    - If the stock belongs to the '제조업' sector and has non-zero 'NP_12M' and zero 'NP' values, the EPSG value is set to np.nan.

    - If the stock belongs to the '제조업' sector and has non-zero 'NP_12M' and non-zero 'NP' values 
        but the difference between the 'NP_12M' and 'NP' values divided by the absolute value of the 'NP' value is outside the range of -10 to 10, 
        the EPSG value is set to np.nan.

    - If the stock belongs to a sector other than '제조업', the EPSG value is set to np.nan.

    Finally, the function applies a mask to the 'EPSG' column of the DataFrame to filter out any rows with non-finite values or values outside the range of -10 to 10. 
        The function then returns the filtered DataFrame containing the calculated EPSG values for each stock in the dataset.
    """
    EPSGTable = pd.DataFrame(index =  RawTable['index'])
    EPSGTable['Sector'] = RawTable['Sector']
    EPSGTable['NP'] = RawTable['NP']
    EPSGTable['NP_12M'] = RawTable['NP_12M']  * 10**5

    mask1 = (EPSGTable['NP_12M'] > 0) & (EPSGTable['NP'] > 0)
    mask2 = (EPSGTable['NP_12M'] > 0) & (EPSGTable['NP'] < 0)
    mask3 = (EPSGTable['NP_12M'] < 0) & (EPSGTable['NP'] < 0)

    EPSGTable.loc[mask1, 'EPSG'] = (EPSGTable['NP_12M'] - EPSGTable['NP']) / abs(EPSGTable['NP'])
    EPSGTable.loc[mask2, 'EPSG'] = (EPSGTable['NP_12M'] - EPSGTable['NP']) / abs(EPSGTable['NP'])
    EPSGTable.loc[mask3, 'EPSG'] = (EPSGTable['NP_12M'] - EPSGTable['NP']) / abs(EPSGTable['NP'])

    EPSGTable.loc[abs(EPSGTable['NP']) == 0, 'EPSG'] = np.nan

    mask = (EPSGTable['Sector'] == "제조업")
    mask &= (EPSGTable['EPSG'] >= -10) & (EPSGTable['EPSG'] <= 10)

    EPSGTable['EPSG'] = np.where(mask, EPSGTable['EPSG'], np.nan)
    del mask
    
    return EPSGTable[['EPSG']]

def GetOPG(RawTable):
    """
    This function calculates the Operating Profit Growth (OPG) for the given stocks.

    Parameters
    ----------
    self : object
        An instance of the `FactorScore` class.

    Returns
    -------
    OPGTable : pandas.DataFrame
        A DataFrame containing the calculated OPG values for each stock in the dataset.

    Notes
    ----
    The function first creates a DataFrame (OPGTable) with the necessary columns. 
        It then applies a series of boolean masks to filter the data based on certain conditions. 
        The first mask (mask1) checks if the OP_12M and OP values are both positive. The second mask (mask2) checks 
        if the OP_12M value is positive while the OP value is negative. The third mask (mask3) checks if both the OP_12M and OP values are negative.

    Based on these masks, the function calculates the OPG values for each stock. 
        If the stock belongs to the "제조업" sector and the OPG value is within the range of -10 to 10, it is retained. 
        Otherwise, it is replaced with NaN.

    Finally, the function returns the OPGTable DataFrame containing the calculated OPG values.
    """
    OPGTable = pd.DataFrame(index = RawTable['index'])
    OPGTable['Sector'] = RawTable['Sector']
    OPGTable['OP'] = RawTable['OP'] / 10**5
    OPGTable['OP_12M'] = RawTable['OP_12M']

    mask1 = (OPGTable['OP_12M'] > 0) & (OPGTable['OP'] > 0)
    mask2 = (OPGTable['OP_12M'] > 0) & (OPGTable['OP'] < 0)
    mask3 = (OPGTable['OP_12M'] < 0) & (OPGTable['OP'] < 0)

    OPGTable.loc[mask1, 'OPG'] = (OPGTable['OP_12M'] - OPGTable['OP']) / OPGTable['OP']
    OPGTable.loc[mask2, 'OPG'] = 0.5
    OPGTable.loc[mask3, 'OPG'] = -0.5

    mask = (OPGTable['Sector'] == "제조업")
    mask &= (OPGTable['OPG'] >= -10) & (OPGTable['OPG'] <= 10)
    OPGTable['OPG'] = np.where(mask, OPGTable['OPG'], np.nan)
    del mask

    return OPGTable[['OPG']]

def GetDY(RawTable):
    """
    This function calculates the Dividend Yield (DY) for the given stocks.

    Parameters
    ----------
    self : object
        An instance of the `FactorScore` class.

    Returns
    -------
    DYTable : pandas.DataFrame
        A DataFrame containing the calculated DY values for each stock in the dataset.

    Notes
    ----
    The function first creates a DataFrame (DYTable) with the necessary columns. It then applies a boolean mask to filter the data based on certain conditions. The first mask (mask1) checks if the SalesP value is not equal to zero. The second mask (mask2) checks if the stock belongs to the "제조업" sector.

    Based on these masks, the function calculates the DY values for each stock. If the stock belongs to the "제조업" sector and the SalesP value is not equal to zero, the DY value is calculated as the Diviend value divided by the Capitalization value (Cap) and then divided by 1000 to get the percentage value. The resulting DY values are then stored in the DYTable DataFrame.

    Finally, the function returns the DYTable DataFrame containing the calculated DY values.
    """
    DYTable = pd.DataFrame(index = RawTable['index'])
    DYTable['Sector'] = RawTable['Sector']
    DYTable['Cap'] = RawTable['Cap']
    DYTable['DY'] = RawTable['DY']
    DYTable['DY'] = DYTable['DY'] / DYTable['Cap'] / 1000
    DYTable['SalesP'] = GetSalesP(RawTable)

    mask1 = (DYTable['SalesP'] != 0)
    mask2 = (DYTable['Sector'] == '제조업')
    DYTable['DY'] = np.where(mask1 & mask2, DYTable['DY'], np.nan)
    del mask1
    del mask2

    return DYTable[['DY']]

def GetInstituitionalBuy(RawTable):
    """
    This function calculates the Institutional Buy value for each stock in the dataset.

    Parameters
    ----------
    self : object
        An instance of the `FactorScore` class.

    Returns
    -------
    DataFrame
        A DataFrame containing the calculated Institutional Buy values for each stock in the dataset.

    Notes
    ----
    The function first creates a DataFrame (InstituitionalBuyTable) with the necessary columns. It then populates the DataFrame with the Institutional Buy values from the 'InstitutionalBuy' attribute of the 'FactorScore' class. The Institutional Buy values are then divided by the 'StockNum' attribute to obtain the normalized Institutional Buy values.

    The function returns the DataFrame containing the calculated Institutional Buy values.
    """
    InstituitionalBuyTable = pd.DataFrame(index = RawTable['index'])
    InstituitionalBuyTable['InstituitionalBuy'] = RawTable['InstituitionalBuy']
    InstituitionalBuyTable['StockNum'] = RawTable['StockNum']
    InstituitionalBuyTable['InstituitionalBuy'] = InstituitionalBuyTable['InstituitionalBuy'] / InstituitionalBuyTable['StockNum']

    return InstituitionalBuyTable[['InstituitionalBuy']]

def GetLeverage(RawTable):
    """
    This function calculates the Leverage value for each stock in the dataset.

    Parameters
    ----------
    self : object
        An instance of the `FactorScore` class.

    Returns
    -------
    DataFrame
        A DataFrame containing the calculated Leverage values for each stock in the dataset.

    Notes
    ----
    The function first creates a DataFrame (LeverageTable) with the necessary columns. It then populates the DataFrame with the Debt and EQT values from the 'DebtTable' and 'Book' attributes of the 'FactorScore' class, respectively. The EQT value is filtered to include only the '1QBefore' column. The Leverage value is then calculated as the ratio of EQT to Debt.

    A boolean mask is applied to the DataFrame to filter out stocks that do not belong to the '제조업' sector, have non-zero EQT and Debt values, and have non-nan Leverage values within the range of -20 to 20. The Leverage values are then replaced with NaN values for stocks that do not meet these criteria.

    Finally, the function returns the DataFrame containing the calculated Leverage values.
    """
    LeverageTable = pd.DataFrame(index = RawTable['index'])
    LeverageTable['Debt'] = RawTable['Debt']
    LeverageTable['Sector'] = RawTable['Sector']
    LeverageTable['EQT'] = RawTable['Book'][[('지배주주지분(천원)_ic', '1QBefore')]]
    LeverageTable['Leverage'] = LeverageTable['EQT'] / LeverageTable['Debt']

    mask = (LeverageTable['Sector'] == '제조업') & (LeverageTable['EQT'] > 0) & (LeverageTable['EQT'] != np.nan)
    mask &= (LeverageTable['Leverage'] != np.nan) & (LeverageTable['Leverage'] <= 20) & (LeverageTable['Leverage'] >= -20)

    LeverageTable['Leverage'] = np.where(mask, LeverageTable['Leverage'], np.nan)
    del mask

    return LeverageTable[['Leverage']]

def GetOPM(RawTable):
    """
    This function calculates the Operating Profit Margin (OPM) for each stock in the dataset.

    Parameters
    ----------
    self : object
        An instance of the `FactorScore` class.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the calculated Operating Profit Margin (OPM) values for each stock in the dataset.

    Notes
    ----
    The function first creates a DataFrame (OPMTable) with the necessary columns. It then populates the DataFrame with the Operating Profit (OP) and Sales values from the 'OPTable' and 'SalesTable' attributes of the 'FactorScore' class, respectively. The Sales value is filtered to include only the '1QBefore' column. The Operating Profit Margin (OPM) value is then calculated as the ratio of Operating Profit (OP) to Sales.

    A boolean mask is applied to the DataFrame to filter out stocks that do not belong to the '제조업' sector, have non-nan Operating Profit Margin (OPM) values, and have non-inf Operating Profit Margin (OPM) values. The Operating Profit Margin (OPM) values are then replaced with NaN values for stocks that do not meet these criteria.

    Finally, the function returns the DataFrame containing the calculated Operating Profit Margin (OPM) values.
    """
    OPMTable = pd.DataFrame(index = RawTable['index'])
    OPMTable['OP'] = RawTable['OP']
    OPMTable['Sector'] = RawTable['Sector']
    OPMTable['Sales'] = RawTable['Sales']
    OPMTable['OPM'] = OPMTable['OP'] / OPMTable['Sales']

    mask = (OPMTable['Sector'] == '제조업') & (OPMTable['OPM'] != np.nan)
    mask &= (OPMTable['OPM'] != np.inf)

    OPMTable['OPM'] = np.where(mask, OPMTable['OPM'], np.nan)
    del mask

    return OPMTable[['OPM']]

def GetOP_M1M(RawTable, ratio, flag2yr):
    """
    This function calculates the Operating Profit Margin (OP_M1M) for each stock in the dataset.

    Parameters
    ----------
    self : object
        An instance of the `FactorScore` class.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the calculated Operating Profit Margin (OP_M1M) values for each stock in the dataset.

    Notes
    -----
    The function first creates a DataFrame (OP_M1MTable) with the necessary columns. It then populates the DataFrame with the Operating Profit (OP) and Operating Profit 12 Months Forward (OP12Fwd) values from the 'OperatingProfit' and 'OperatingProfit12Fwd' attributes of the 'FactorScore' class, respectively. The Sales value is filtered to include only the '1QBefore' column. The Operating Profit Margin (OP_M1M) value is then calculated as the ratio of Operating Profit (OP) to Sales.

    A boolean mask is applied to the DataFrame to filter out stocks that do not belong to the '제조업' sector, have non-nan Operating Profit Margin (OP_M1M) values, and have non-inf Operating Profit Margin (OP_M1M) values. The Operating Profit Margin (OP_M1M) values are then replaced with NaN values for stocks that do not meet these criteria.

    Finally, the function returns the DataFrame containing the calculated Operating Profit Margin (OP_M1M) values.
    """
    OP_M1MTable = pd.DataFrame(index = RawTable['index'])
    OP_M1MTable['OPCurrent'] = RawTable['OperatingProfit'].loc[:, (slice(None), 'Current')]
    OP_M1MTable['OP12FwdCurrent'] = RawTable['OperatingProfit12Fwd'].loc[:, (slice(None), 'NextYear')]
    OP_M1MTable['OP2yrCurrent'] = RawTable['OperatingProfit2yr'].loc[:, (slice(None), 'TwoYear')]
    OP_M1MTable['OP60DaysBefore'] = RawTable['OperatingProfit'].loc[:, (slice(None), '60DayBefore')]
    OP_M1MTable['OP12Fwd60DaysBefore'] = RawTable['OperatingProfit12Fwd'].loc[:, (slice(None), '60DayBefore NextYear')]
    OP_M1MTable['OP2yr60DaysBefore'] = RawTable['OperatingProfit2yr'].loc[:, (slice(None), '60DayBefore TwoYear')]

    if flag2yr == 0 :
        OP_M1MTable['OP_M1M'] = (OP_M1MTable['OPCurrent'] * (1 - ratio) + OP_M1MTable['OP12FwdCurrent'] * (ratio)) \
                                / (OP_M1MTable['OP60DaysBefore'] * (1 - ratio) + OP_M1MTable['OP12Fwd60DaysBefore'] * (ratio)) - 1
    else :
        OP_M1MTable['OP_M1M'] = (OP_M1MTable['OPCurrent'] * (1 - ratio) + OP_M1MTable['OP12FwdCurrent'] * (ratio)) \
                                / (OP_M1MTable['OP12Fwd60DaysBefore'] * (1 - ratio) + OP_M1MTable['OP2yr60DaysBefore'] * (ratio)) - 1

    mask = (OP_M1MTable['OP_M1M'] <= 1) & (OP_M1MTable['OP_M1M'] >= -1)
    mask &= (OP_M1MTable['OP_M1M'] != np.nan)
    mask &= (OP_M1MTable['OP_M1M'] != np.inf)

    OP_M1MTable['OP_M1M'] = np.where(mask, OP_M1MTable['OP_M1M'], np.nan)
    del mask

    return OP_M1MTable[['OP_M1M']]

def GetPriceMomentum(RawTable):
    """
    This function calculates the Price Momentum for each stock in the dataset.

    Parameters
    ----------
    self : object
        An instance of the `FactorScore` class.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the calculated Price Momentum values for each stock in the dataset.

    Notes
    ----
    The function first creates a DataFrame (PriceMomentumTable) with the necessary columns. It then populates the DataFrame with the Price values from the 'Price' attribute of the 'FactorScore' class, specifically the 'Current' and '180DayBefore' columns. The Price Momentum value is then calculated as the percentage change of the Price value from the '180DayBefore' column to the 'Current' column.

    A boolean mask is applied to the DataFrame to filter out stocks that do not belong to the '제조업' sector, have non-nan Price Momentum values, and have non-inf Price Momentum values. The Price Momentum values are then replaced with NaN values for stocks that do not meet these criteria.

    Finally, the function returns the DataFrame containing the calculated Price Momentum values.
    """
    PriceMomentum = pd.DataFrame(index = RawTable['index'])
    PriceMomentum['PriceCurrent'] = RawTable['Price'].loc[:, (slice(None), 'Current')]
    PriceMomentum['Price180DayBefore'] = RawTable['Price'].loc[:, (slice(None), '180DayBefore')]
    PriceMomentum['PriceMomentum'] = RawTable['Price'].loc[:, (slice(None), ['180DayBefore', 'Current'])].pct_change(axis=1).iloc[:,1]

    mask = (PriceMomentum['PriceMomentum'] != np.nan)
    mask &= np.where(np.isfinite(PriceMomentum['PriceMomentum']),PriceMomentum['PriceMomentum'],np.nan)

    PriceMomentum['PriceMomentum'] = np.where(mask, PriceMomentum['PriceMomentum'], np.nan)
    PriceMomentum['PriceMomentum'] = PriceMomentum['PriceMomentum'].fillna(0)
    del mask

    return PriceMomentum[['PriceMomentum']]

def GetReverseMomentum(RawTable):
    """
    This function calculates the Reverse Momentum for each stock in the dataset.

    Parameters
    ----------
    self : object
        An instance of the `FactorScore` class.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the calculated Reverse Momentum values for each stock in the dataset.

    Notes
    ----
    The function first creates a DataFrame (ReverseMomentumTable) with the necessary columns. It then populates the DataFrame with the Price values from the 'Price' attribute of the 'FactorScore' class, specifically the 'Current' and 'HighestPrice' columns. The Reverse Momentum value is then calculated as the ratio of the 'HighestPrice' to the 'PriceCurrent' - 1.

    A boolean mask is applied to the DataFrame to filter out stocks that do not belong to the '제조업' sector. The Reverse Momentum values are then replaced with NaN values for stocks that do not belong to the '제조업' sector.

    Finally, the function returns the DataFrame containing the calculated Reverse Momentum values.
    """
    ReverseMomentum = pd.DataFrame(index = RawTable['index'])
    ReverseMomentum['PriceCurrent'] = RawTable['Price'].loc[:, (slice(None), 'Current')]
    ReverseMomentum['HighestPrice'] = RawTable['HighestPrice']
    ReverseMomentum['Sector'] = RawTable['Sector']
    ReverseMomentum['ReverseMomentum'] = ReverseMomentum['HighestPrice'] / ReverseMomentum['PriceCurrent'] - 1

    mask = (ReverseMomentum['ReverseMomentum'] != np.nan)
    mask &= (ReverseMomentum['Sector'] == '제조업')

    ReverseMomentum['ReverseMomentum'] = np.where(mask, ReverseMomentum['ReverseMomentum'], np.nan)
    del mask

    return ReverseMomentum[['ReverseMomentum']]

###########################################수정###########################################
def GetFactorInitialTable(RawTable, ratio, flag2yr):
    """
    This function creates a DataFrame with the necessary columns to calculate factor scores for each stock in the dataset.
    It then applies a boolean mask to filter the data based on certain conditions. The first mask (mask1) checks if the SalesP value is not equal to zero.
    The second mask (mask2) checks if the stock belongs to the "제조업" sector.
    The function then calculates the factor scores for each stock in the dataset using the boolean masks and the RawTable DataFrame.
    If the SalesP value is not equal to zero and the stock belongs to the "제조업" sector, the factor scores are calculated as the Diviend value divided by the Capitalization value (Cap) and then divided by 1000 to get the percentage value.
    The resulting FactorTable DataFrame containing the calculated factor scores for each stock in the dataset is then returned.
    Parameters
    ----------
    RawTable : pandas.DataFrame
        A DataFrame containing the necessary columns for calculating factor scores.
    ratio : float
        A ratio used in the calculation of the OP_M1M factor score.
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the calculated factor scores for each stock in the dataset.
    Notes
    ----
    The function first creates a DataFrame with the necessary columns. It then applies a boolean mask to filter the data based on certain conditions. The first mask checks if the SalesP value is not equal to zero. The second mask checks if the stock belongs to the "제조업" sector. The factor scores are then calculated for each stock in the dataset using the boolean masks and the RawTable DataFrame. If the SalesP value is not equal to zero and the stock belongs to the "제조업" sector, the factor scores are calculated as the Diviend value divided by the Capitalization value (Cap) and then divided by 1000 to get the percentage value. A DataFrame containing the calculated factor scores for each stock in the dataset is then returned.
    """
    factors = {
        'BP': GetBP(RawTable),
        'AssetP': GetAssetP(RawTable),
        'SalesP': GetSalesP(RawTable),
        'ROE': GetROE(RawTable),
        'EBITDAY': GetEBITDAY(RawTable),
        'EY_12M': GetEY_12M(RawTable),
        'EPSG': GetEPSG(RawTable),
        'OPG': GetOPG(RawTable),
        'DY': GetDY(RawTable),
        'InstituitionalBuy': GetInstituitionalBuy(RawTable),
        'Leverage': GetLeverage(RawTable),
        'OPM': GetOPM(RawTable),
        'OP_M1M': GetOP_M1M(RawTable, ratio, flag2yr),
        'ReverseMomentum': GetReverseMomentum(RawTable),
        'PriceMomentum': GetPriceMomentum(RawTable)
    }

    # Construct the DataFrame in one go
    FactorTable = pd.concat(factors.values(), axis=1)

    # Replace infinities with NaNs
    # FactorTable = FactorTable.applymap(lambda x: np.nan if not np.isfinite(x) else x)
    FactorTable = FactorTable.map(lambda x: np.nan if not np.isfinite(x) else x)

    return FactorTable

def AdjustLeverage(FactorTable):
    """
    This function adjusts the Leverage values in the FactorTable DataFrame.

    Parameters
    ----------
    FactorTable : pandas.DataFrame
        A DataFrame containing the necessary columns for calculating factor scores.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the adjusted Leverage values for each stock in the dataset.

    Notes
    ----
    The function first checks if the 'Leverage' column is present in the FactorTable DataFrame. If it is, the function calculates the mean and standard deviation of the Leverage values. It then creates masks to filter out stocks with non-finite Leverage values and Leverage values less than zero. The function then adjusts the Leverage values using a clipping function to ensure they fall within the range of -3 to 3. The adjusted Leverage values are then replaced in the FactorTable DataFrame.
<|im_start|>
    """
    if 'Leverage' in FactorTable.columns:
        leverage = FactorTable['Leverage']
        m, s = leverage.mean(), leverage.std()
        mask1, mask2 = leverage < 0, np.isfinite(leverage)
        FactorTable['Leverage'] = np.where(
            mask1,
            3,
            np.where(mask2, np.clip((leverage - m) / s, -3, 3), np.nan)
        )
    return FactorTable

def StandardizeColumns(FactorTable):
    """
    This function standardizes the columns of the FactorTable DataFrame by subtracting the mean and dividing by the standard deviation.
    It then clips the values to fall within the range of -3 to 3.

    Parameters
    ----------
    FactorTable : pandas.DataFrame
        A DataFrame containing the necessary columns for calculating factor scores.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the standardized columns of the input FactorTable DataFrame.

    Notes
    -----
    The function first calculates the mean and standard deviation of each column in the FactorTable DataFrame, excluding the 'Leverage' column.
    It then iterates through each column in the DataFrame, subtracting the mean and dividing by the standard deviation to standardize the values.
    Finally, the function clips the standardized values to fall within the range of -3 to 3 using the clip function from the pandas library.
    """
    FactorMean = FactorTable.mean()
    FactorStd = FactorTable.std()
    for col in FactorTable.columns:
        if col != 'Leverage':
            FactorTable[col] = (FactorTable[col] - FactorMean[col]) / FactorStd[col]
            FactorTable[col] = FactorTable[col].clip(-3, 3)
    return FactorTable

def RankFactors(FactorTable):
    """
    This function ranks the columns of the FactorTable DataFrame based on their values.

    Parameters
    ----------
    FactorTable : pandas.DataFrame
        A DataFrame containing the necessary columns for calculating factor scores.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the ranked columns of the input FactorTable DataFrame.

    Notes
    -----
    The function first calculates the mean and standard deviation of each column in the FactorTable DataFrame, excluding the 'Leverage' column.
    It then iterates through each column in the DataFrame, ranking the values in descending order using the 'rank' function from pandas, with the 'method' parameter set to 'min'.
    Finally, the function returns the DataFrame containing the ranked columns.
    """
    FactorRankTable_Full = FactorTable.rank(ascending=False, method='min')
    FactorRankTable_Full['count'] = FactorRankTable_Full.count(axis=1)
    FactorRankTable_Full.loc['A005930', 'count'] = 15
    return FactorRankTable_Full

def SelectRankFactors(FactorRankTable_Full):
    """
    This function selects the top factors based on their ranks.

    Parameters
    ----------
    FactorRankTable_Full : pandas.DataFrame
        A DataFrame containing the factors with their ranks.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the selected factors with their ranks.

    Notes
    -----
    The function first filters the factor rank table to include only the factors with a count greater than 14. This is done by creating a boolean mask that checks if the 'count' column is greater than 14. The function then selects the rows in the factor rank table that satisfy this condition.

    The function returns a DataFrame containing the selected factors with their ranks.
    """
    mask = FactorRankTable_Full['count'] > 14
    FactorRankTable_Selected = FactorRankTable_Full.loc[mask]
    del mask  # Free memory
    return FactorRankTable_Selected

def RankSelectedFactors(FactorRankTable_Selected):
    """
    This function ranks the selected factors based on their values.

    Parameters
    ----------
    FactorRankTable_Selected : pandas.DataFrame
        A DataFrame containing the selected factors with their ranks.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the ranked selected factors.

    Notes
    -----
    The function first filters the factor rank table to include only the selected factors. Then, it ranks the selected factors based on their values using the 'rank' function from pandas, with the 'method' parameter set to 'min'. Finally, the function returns the DataFrame containing the ranked selected factors.
    """
    FactorRankTable_Selected = FactorRankTable_Selected.rank(method='min')
    return FactorRankTable_Selected

def GetWeights(FactorRankTable_Selected, A005930Weight):
    """
    This function calculates the factor weights based on the factor ranks and the given weight for factor A005930.

    Parameters
    ----------
    FactorRankTable_Selected : pandas.DataFrame
        A DataFrame containing the selected factors with their ranks.
    A005930Weight : float
        The given weight for factor A005930.
    ratio : float
        A ratio used in the calculation of the OP_M1M factor score.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the factor weights for each factor.

    Notes
    ----
    The function first calculates the factor weights based on the factor ranks and the given weight for factor A005930. The factor weights are calculated using a linear combination of the factor ranks and the given weight for factor A005930. The factor weights are then stored in a DataFrame with the same index as the input DataFrame.
    """
    FactorScoreTable = pd.DataFrame(index=FactorRankTable_Selected.index)
    AdditionalWeight = 0.07
    A005930Weight1 = (A005930Weight + AdditionalWeight) * 20
    AdjustedCoeff = (20 - A005930Weight1) / 20

    for col in FactorRankTable_Selected.columns:
        mask = FactorRankTable_Selected[col] < 21
        mask = np.where(mask, AdjustedCoeff, 0)
        FactorScoreTable[col] = mask

    FactorScoreTable.loc['A005930', :] = A005930Weight1
    FactorScoreTable.drop(columns=['count'], inplace=True)

    return FactorScoreTable

def GetFactorTable(RawTable, A005930Weight, ratio):
    FactorTable = GetFactorInitialTable(RawTable, ratio, 0)
    FactorTable = AdjustLeverage(FactorTable)
    FactorTable = StandardizeColumns(FactorTable)
    FactorRankTable_Full = RankFactors(FactorTable)
    FactorRankTable_Selected = SelectRankFactors(FactorRankTable_Full)
    FactorRankTable_Selected = RankSelectedFactors(FactorRankTable_Selected)
    FactorScoreTable = GetWeights(FactorRankTable_Selected, A005930Weight)

    Tables = {
        'FactorTable': FactorTable,
        'FactorRankTable_Full': FactorRankTable_Full,
        'FactorRankTable_Selected': FactorRankTable_Selected,
        'FactorScoreTable': FactorScoreTable
    }

    return Tables

def GetFactorTop(Tables):
    """
    This function returns a dictionary of the top stocks based on their factor scores.

    Parameters
    ----------
    self : object
        An instance of the `FactorScore` class.

    Returns
    -------
    dict
        A dictionary containing the top stocks based on their factor scores. 
        The keys of the dictionary are the factor names, and the values are tuples containing the stock symbols of the top stocks for that factor.

    Notes
    ----
    The function first filters the factor score table to include only the factor scores that are greater than zero. 
    It then iterates through each column of the filtered table, extracting the indices of the top stocks for that factor. T
    he top stocks are defined as those with the highest factor scores. 
    The function then constructs a dictionary, with the factor names as the keys and the tuples of top stock symbols as the values. 
    Finally, the function returns the dictionary of top stocks based on their factor scores.
    """
    FactorScoreTable = Tables['FactorScoreTable']
    ports = {}
    for col in FactorScoreTable.columns:
        mask = FactorScoreTable[col] > 0
        port = tuple(FactorScoreTable.loc[mask, col].index)
        ports[col] = port

    return ports

def Get005930Ratio(MktCap):
    results = []

    # Group the DataFrame by 'Dates'
    grouped = MktCap.groupby('Dates')

    for date, group in grouped:
        # Drop 'Dates' column, set multi-index, and drop rows with NaN in 'MktCap'
        MktCaptemp = group.drop(columns='Dates').set_index(['Symbol', 'Name'])
        MktCaptemp = MktCaptemp[MktCaptemp['MktCap'].notna()]

        # Calculate the total market cap and the ratio
        total_mktcap = MktCaptemp['MktCap'].sum()
        MktCaptemp['ratio'] = MktCaptemp['MktCap'] / total_mktcap

        # Add the 'Dates' column back for the final results
        MktCaptemp['Dates'] = date

        # Append the temporary DataFrame to the results list
        results.append(MktCaptemp.reset_index())

    # Concatenate all results into a single DataFrame
    final_results = pd.concat(results, ignore_index=True)

    # Filter the results for the specific symbol and drop unnecessary columns
    filtered_results = final_results[final_results['Symbol'] == 'A005930'].drop(['Symbol', 'Name', 'MktCap'], axis=1)
    
    # Set 'Dates' as the index
    filtered_results = filtered_results.set_index('Dates')

    return filtered_results
##################################################################다이나믹 퀀트 포트폴리오 코드#######################################################################################

def GetCumulativeReturn(ReturnDF):
    return (ReturnDF + 1).cumprod()

def GetFactorOfReturnInterval(Date, FactorRetCum, SelectedFactors, tradingDates):

    EndDate = pd.to_datetime(Date)
    StartDate = tradingDates[tradingDates <= Date][-251]

    FactorOfReturnIntervalDF = pd.DataFrame(FactorRetCum.loc[EndDate, SelectedFactors] / FactorRetCum.loc[StartDate, SelectedFactors] - 1)
    FactorOfReturnIntervalDF.columns = ['return']
    FactorOfReturnIntervalDF['rank'] = FactorOfReturnIntervalDF['return'].rank(ascending=False)
    
    return FactorOfReturnIntervalDF
    
def GetIndexAtDate(indexCap, Date):
    tmpIndexCap = indexCap.loc[:Date]
    nRow = tmpIndexCap.shape[0] - 1
    date2 = tmpIndexCap.index[nRow][0]
    return indexCap.loc[date2].sort_values('idx_weight', ascending=False)

def LoadRegime(REGIMEDIR):
    df = pd.read_excel(REGIMEDIR)
    df.columns = ['Date', 'Regime']
    df = df.set_index('Date')
    return df


# def GetRegime(Date, DatesFrame, RegimeFrame):
#     """
#     This method is used to get the regime factor return at a given date.
#     The regime factor return is calculated based on the regime values obtained from the RegimeTemp data source.

#     Parameters:
#     Date (str): The date for which the regime factor return is to be obtained.
#     DatesFrame (pd.DataFrame): A DataFrame containing the dates for which factor returns are available.
#     RegimeFrame (pd.DataFrame): A DataFrame containing the regime factor values.

#     Returns:
#     pd.DataFrame: A DataFrame containing the regime factor return at the given date.

#     Raises:
#     ValueError: If the regime factor return for the given date is not available.
#     """
#     Date = pd.to_datetime(Date)

#     # Align RegimeFrame to DatesFrame and forward fill
#     regime = RegimeFrame.reindex(DatesFrame).ffill()
        
#     regime = regime.ffill()
#     regime = regime.bfill()
#     regime = regime.loc[pd.Timestamp('2012-01-02'):,:]
#     regime.columns = ['Regime']

#     return regime

def GetRegime(tradingDates, REGIMEDIR):
    regime_temp = pd.read_excel(REGIMEDIR)

    regime_temp.columns = ['date', 'regime_temp']
    regime_temp = regime_temp.set_index('date')

    regime = pd.DataFrame(index = tradingDates)
    regime['Regime'] = np.nan

    regime_temp_df = pd.concat([regime, regime_temp], axis=1)
    regime_temp_df['Regime'] = regime_temp_df['regime_temp']

    regime_temp_df = regime_temp_df.ffill()
    regime_temp_df = regime_temp_df.loc[pd.to_datetime('2012-01-02'):,:]
    
    return regime_temp_df[['Regime']]

# def GetRegimeFactorReturn(FactorReturnDF, regime):
#     """
#     This function calculates the regime factor return for the given FactorReturnDF.

#     Parameters:
#     FactorReturnDF (pd.DataFrame): A DataFrame containing the factor returns for the selected factors.
#     regime (pd.DataFrame): A DataFrame containing the regime values with the same index as FactorReturnDF.

#     Returns:
#     FactorRetDict (dict): A dictionary containing DataFrames for each regime.
#     """
#     # Ensure the regime DataFrame has the same index as FactorReturnDF
#     regime = regime.reindex(FactorReturnDF.index, method='ffill')

#     # Create boolean masks based on regime values
#     regime_masks = {
#         1: (regime['Regime'] == 1),
#         2: (regime['Regime'] == 2),
#         3: (regime['Regime'] == 3)
#     }

#     # Apply the masks to filter FactorReturnDF
#     FactorRetDict = {
#         regime_num: FactorReturnDF[mask].reindex(FactorReturnDF.index).fillna(0)
#         for regime_num, mask in regime_masks.items()
#     }

#     return FactorRetDict

def GetRegimeFactorReturn(factor_ret_final,regime):
    
    dates = factor_ret_final.index
    
    factor_ret_final_1 = pd.DataFrame(index=dates, columns=factor_ret_final.columns, data=0, dtype=float)
    factor_ret_final_2 = pd.DataFrame(index=dates, columns=factor_ret_final.columns, data=0, dtype=float)
    factor_ret_final_3 = pd.DataFrame(index=dates, columns=factor_ret_final.columns, data=0, dtype=float)
    
    factor_ret_final_dict = {
        1: factor_ret_final_1,
        2: factor_ret_final_2,
        3: factor_ret_final_3
    }
    
    for idx, date in enumerate(dates):
        regime_val = regime.loc[date, 'Regime']
        
        selected_df = factor_ret_final_dict.get(regime_val, None)
    
        if selected_df is not None:
            for (col, val) in factor_ret_final.loc[date, :].to_dict().items():
                selected_df.loc[date, :][col] = val
    
    return factor_ret_final_1, factor_ret_final_2, factor_ret_final_3

# def GetRegimeFactorReturnAtDate(Date, Regime, FactorRetDict):
#     """
#     Get the regime factor return at a given date.

#     Parameters
#     ----------
#     Date : str
#         The date for which the regime factor return is to be obtained.
#     Regime : pd.DataFrame
#         A DataFrame containing the regime values.
#     FactorRetDict : dict
#         A dictionary containing DataFrames for each regime.

#     Returns
#     -------
#     pd.DataFrame
#         A DataFrame containing the regime factor return at the given date.

#     Raises
#     ------
#     ValueError
#         If the regime factor return for the given date is not available.
#     """
#     Date = pd.to_datetime(Date)
#     regimeVal = Regime.loc[Date, 'Regime']
    
#     if regimeVal in FactorRetDict:
#         FactorRetRegimeAtDate = FactorRetDict[regimeVal].loc[:Date]
#     else:
#         FactorRetRegimeAtDate = pd.DataFrame()
    
#     return FactorRetRegimeAtDate

def GetRegimeFactorReturnAtDate(date, indexCapTotal, regime, factor_ret_full, factor_ret_final_1, factor_ret_final_2, factor_ret_final_3):
    
    #indexCapatDate = GetIndexAtDate(indexCapTotal, date)
    regime_val = regime.loc[date,'Regime']
    
    if regime_val == 1:
        factor_ret_regime_val = factor_ret_final_1.loc[:date, :]
    elif regime_val == 2:
        factor_ret_regime_val = factor_ret_final_2.loc[:date, :]
    else:
        factor_ret_regime_val = factor_ret_final_3.loc[:date, :]
    
    return factor_ret_regime_val

EsgScoreSustin = {'AA':50,'A':40,'BB':30,'B':20,'C':10,'D':0,'E':0,np.nan:0,0:0,'-':0}
EsgScoreDB = {'S':'AA','AA':'AA','BB':'BB','A':'A','B':'BB','C':'B','D':'C','E':np.nan,'-':np.nan, np.nan: np.nan, 0:0}

def LoadESG(date, esg_data_dir):
    """
    Load ESG data from the specified directory.

    Parameters
    ----------
    date : str or datetime
        The date to determine the ESG data file to load.
    esg_data_dir : str
        The directory path to the ESG data file.

    Returns
    -------
    esg : pandas.DataFrame
        A DataFrame containing the ESG data.

    Raises
    ------
    ValueError
        If the ESG data file is not found or if the DataFrame columns are not as expected.
    """
    year = pd.to_datetime(date).year
    month = pd.to_datetime(date).month

    if month < 7:
        month = '2'
        year -= 1
        year = str(year)[-2:]
    else:
        month = '1'
        year = str(year)[-2:]

    sheet_name = month + 'H' + year

    try:
        esg = pd.read_excel(esg_data_dir, sheet_name=sheet_name)
    except FileNotFoundError:
        raise ValueError(f"ESG data file not found in directory: {esg_data_dir}")
    except ValueError:
        raise ValueError(f"Sheet {sheet_name} not found in the ESG data file.")

    if len(esg.columns) > 3:
        esg.columns = ['Symbol', 'Name', 'DB', 'Sustin']
    else:
        esg.columns = ['Symbol', 'Name', 'Sustin']
        esg['DB'] = 0

    esg = esg.set_index('Symbol')

    return esg


def GetESGScore(Date, indexCap, PortfolioDF, esg_score_db, esg_score_sustin):
    """
    Calculate the ESG scores for a given portfolio.

    Parameters
    ----------
    date : str or datetime
        The date to determine the ESG data file to load.
    portfolio_df : pandas.DataFrame
        A DataFrame containing the portfolio data.
    esg_score_db : dict
        A dictionary mapping ESG DB scores.
    esg_score_sustin : dict
        A dictionary mapping ESG Sustin scores.

    Returns
    -------
    PortfolioDF : pandas.DataFrame
        The portfolio DataFrame with added ESG scores and ranks.
    """
    index_at_date = GetIndexAtDate(indexCap,Date)
    
    esg_score_sum = []
    
    for idx in PortfolioDF.index:
        score1 = PortfolioDF.loc[idx, 'esgDB']
        if pd.isna(score1):
            score1 = np.nan
        score1 = EsgScoreDB[score1]
        score1 = EsgScoreSustin[score1]
    
        score2 = PortfolioDF.loc[idx, 'esgSustin']
        score2 = EsgScoreSustin[score2]
    
        esg_score_sum.append(score1 + score2)
        
    PortfolioDF = PortfolioDF.reset_index().set_index('Symbol')
    PortfolioDF['idx_weight'] = index_at_date['idx_weight']
    PortfolioDF = PortfolioDF.reset_index().set_index(['Symbol','Name'])
    
    EsgScoreSum = np.asarray(esg_score_sum)
    
    PortfolioDF['EsgScoreSum'] = EsgScoreSum + PortfolioDF['idx_weight'] *0.01
    PortfolioDF['EsgScoreSum_rank'] = PortfolioDF['EsgScoreSum'].rank(ascending=False, method = 'min')
    
    PortfolioDF = PortfolioDF.sort_values('idx_weight', ascending=False)
    
    return PortfolioDF

# def GetFactorWeight(date, FactorRetFull, selectedFactors, RegimeDF, RegimeFactorReturnDict, tradingDates):
#     """
#     This function calculates the factor weights for the selected factors.

#     Parameters:
#     self (object): An instance of the `FactorWeight` class.
#     FactorReturnDF (pd.DataFrame): A DataFrame containing the factor returns for the selected factors.
#     SelectedFactors (list): A list of the selected factor names.
#     KOSPI200 (pd.DataFrame): A DataFrame containing the KOSPI 200 stock data.

#     Returns:
#     None

#     This function is used to calculate the factor weights for the selected factors based on the factor returns and other factors. 
#     The factor weights are calculated using a combination of the factor returns and a regime factor return. 
#     The regime factor return is calculated by applying the factor returns for the selected factors to the regime value. 
#     The regime factor return is then used to adjust the factor weights. 
#     The adjusted factor weights are then stored in the `FactorWeight` class instance.
#     """

#     # Get the regime factor returns and calculate cumulative returns
#     factorRetRegime = GetRegimeFactorReturnAtDate(date, RegimeDF, RegimeFactorReturnDict)
#     factorRetRegimeCum = GetCumulativeReturn(factorRetRegime)
    
#     # Calculate cumulative returns for the full factor return DataFrame
#     factorRetFullCum = GetCumulativeReturn(FactorRetFull)

#     startDate = tradingDates[tradingDates <= date][-251]

#     # Rank the factors based on their cumulative returns at the given date
#     factorRetRegimeRank = GetFactorOfReturnInterval(date, factorRetRegimeCum, selectedFactors, tradingDates)
#     factorRetFullRank = GetFactorOfReturnInterval(date, factorRetFullCum, selectedFactors, tradingDates)
    
#     # Define constants for weight calculation
#     regimeConstant, cumConstant = 0.1, 0.1
#     weightRegime, weightCum = 1, 2
    
#     # Calculate weights based on ranks
#     factorRetRegimeRank['weight'] = np.maximum(weightRegime - factorRetRegimeRank['rank'] * regimeConstant, 0)
#     factorRetFullRank['weight'] = np.maximum(weightCum - factorRetFullRank['rank'] * cumConstant, 0)
    
#     # Sum the weights from both rankings
#     factorWeight = factorRetFullRank['weight'] + factorRetRegimeRank['weight']
    
#     return factorWeight

def GetFactorWeight(factorRetRegimeRank, factorRetFullRank):
    
    # Define constants for weight calculation
    regimeConstant, cumConstant = 0.1, 0.1
    weightRegime, weightCum = 1, 2
    
    # Calculate weights based on ranks
    factorRetRegimeRank['weight'] = np.maximum(weightRegime - factorRetRegimeRank['rank'] * regimeConstant, 0)
    factorRetFullRank['weight'] = np.maximum(weightCum - factorRetFullRank['rank'] * cumConstant, 0)
    
    # Sum the weights from both rankings
    factorWeight = factorRetFullRank['weight'] + factorRetRegimeRank['weight']
    
    return factorWeight


def GetStockFactorScoreTable(factorTable, selectedFactors, factorWeight):
    """
    Calculate the stock factor score table.

    Parameters:
    factorTable (pd.DataFrame): DataFrame containing the factor data.
    selectedFactors (list): List of selected factors.
    factorWeight (pd.Series): Series containing the factor weights.

    Returns:
    pd.DataFrame: A DataFrame containing the stock factor scores.
    """
    # Select relevant factors
    stockFactorScoreTable = factorTable.loc[:, selectedFactors].copy()
    stockFactorScoreTable['weight division'] = 0.0

    # Sum of the factor weights
    factorWeightSum = factorWeight.sum()

    # Calculate weight division
    for idx in stockFactorScoreTable.index:
        stockFactorScoreTableTemp = stockFactorScoreTable.loc[idx, :]
        sumWeightWONA = sum(weight if isna else 0 for isna, weight in zip(stockFactorScoreTableTemp.isna(), factorWeight))
        weightTemp = factorWeightSum - sumWeightWONA
        stockFactorScoreTable.loc[idx, 'weight division'] = weightTemp

    # Calculate ASCORE and SCORE
    stockFactorScoreTable['ASCORE'] = stockFactorScoreTable[selectedFactors].fillna(0) @ factorWeight
    stockFactorScoreTable['SCORE'] = stockFactorScoreTable['ASCORE'] / stockFactorScoreTable['weight division']

    return stockFactorScoreTable

def GetLowerBound(portfolioDF):
    """
    This function calculates the lower bound of the portfolio weights.

    Parameters
    ----------
    portfolioDF : pd.DataFrame
        A DataFrame containing the portfolio data.

    Returns
    -------
    pd.Series
        A pandas Series object containing the lower bound of the portfolio weights.

    Notes
    -----
    The lower bound is calculated based on the following conditions:
    1. If the portfolio weight is greater than 1.1, subtract 1.1 from it.
    2. If the portfolio weight belongs to the '은행', '보험', or '증권' sectors, subtract 0.001 from it.
    3. If the portfolio weight's Amt_Avg_20D is 0, set the lower bound to 0.
    4. If the portfolio weight's SCORE is NaN, set the lower bound to 0.
    5. If the portfolio weight's SCORE is 0, set the lower bound to 0.
    """
    temp_df = portfolioDF.copy(deep=True)

    mask = (temp_df['idx_weight'] - 1.1 > 0)
    temp_df['lowerbound'] = np.where(mask, temp_df['idx_weight'] - 1.1, 0)

    mask = (temp_df['FGSector'] == '은행')
    mask |= (temp_df['FGSector'] == '보험')
    mask |= (temp_df['FGSector'] == '증권')
    temp_df['lowerbound'] = np.where(mask, temp_df['idx_weight']-0.001, temp_df['lowerbound'])

    mask = (temp_df['Amt_Avg_20D'] == 0)
    temp_df['lowerbound'] = np.where(mask, 0, temp_df['lowerbound'])

    mask = (temp_df['score'].isna())
    temp_df['lowerbound'] = np.where(mask, 0, temp_df['lowerbound'])

    mask = (temp_df['score'] == 0)
    temp_df['lowerbound'] = np.where(mask, 0, temp_df['lowerbound'])
    
    return temp_df['lowerbound']

def GetUpperBound(portfolioDF):
    """
    This function calculates the upper bound of the portfolio weights.

    Parameters
    ----------
    portfolioDF : pd.DataFrame
        A DataFrame containing the portfolio data.

    Returns
    -------
    pd.Series
        A pandas Series object containing the upper bound of the portfolio weights.

    Notes
    -----
    The upper bound is calculated based on the following conditions:
    1. If the portfolio weight belongs to the '은행', '보험', or '증권' sectors, subtract 0.001 from it.
    2. If the portfolio weight's Amt_Avg_20D is 0, set the upper bound to 0.
    3. If the portfolio weight's EsgScoreSum_rank is greater than 150, set the upper bound to 0.
    """
    
    temp_df = portfolioDF.copy(deep=True)
    temp_df['upperbound1'] = temp_df['idx_weight'] * 9
    temp_df['upperbound2'] = temp_df['idx_weight'] + 1
    temp_df['upperbound'] = temp_df[['upperbound1','upperbound2']].min(axis=1)

    mask = (temp_df['FGSector'] == '은행')
    mask |= (temp_df['FGSector'] == '보험')
    mask |= (temp_df['FGSector'] == '증권')
    temp_df['upperbound'] = np.where(mask, temp_df['idx_weight']-0.001, temp_df['upperbound'])

    mask = (temp_df['Amt_Avg_20D'] == 0)
    temp_df['upperbound'] = np.where(mask, 0, temp_df['upperbound'])
    
    mask = (temp_df['esg_score_sum_rank'] > 150)
    temp_df['upperbound'] = np.where(mask, 0, temp_df['upperbound'])

    return temp_df['upperbound']
    
def ConditioningScore(portfolioDF):
    """
    This function conditions the portfolio scores based on certain conditions.

    Parameters
    ----------
    portfolioDF : pd.DataFrame
        A DataFrame containing the portfolio data.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the conditioned portfolio scores.

    Notes
    -----
    The function iterates through the portfolio DataFrame and updates the 'SCORE' column based on the following conditions:
    1. If the portfolio weight belongs to the '은행', '보험', or '증권' sectors, it subtracts 0.001 from it.
    2. If the portfolio weight's Amt_Avg_20D is 0, it sets the score to 0.
    3. If the portfolio weight's consensus is less than 0 or its Amt_Avg_20D is less than or equal to 10, it sets the score to 0.
    Otherwise, it keeps the original score.
    """
    for idx in portfolioDF.index:
        ascore = portfolioDF.loc[idx, 'SCORE']
        
        if portfolioDF.loc[idx, 'FGSector'] == '은행' or portfolioDF.loc[idx, 'FGSector'] == '보험' or portfolioDF.loc[idx, 'FGSector'] == '증권' :
            portfolioDF.loc[idx, 'SCORE'] = ascore
        elif portfolioDF.loc[idx, 'EsgScoreSum_rank'] >= 151:
            portfolioDF.loc[idx, 'SCORE'] = 0
        elif pd.isna(portfolioDF.loc[idx, 'EY_12M']):
            portfolioDF.loc[idx, 'SCORE'] = -1000
        elif portfolioDF.loc[idx, 'consensus'] < 0 or (portfolioDF.loc[idx, 'Amt_Avg_20D'] / 10**8) <= 10:
            portfolioDF.loc[idx, 'SCORE'] = 0
        else:
            portfolioDF.loc[idx, 'SCORE'] = ascore

    return portfolioDF
    

def GatherInfo(date, indexCap, kospi200atD, consensus, transaction, IFRSsector, size, StockFactorScoreTable, esgData):
    """
    Gathers necessary information for portfolio optimization.

    Parameters
    ----------
    date : str
        The date for which the information is to be gathered.
    kospi200 : pd.DataFrame
        DataFrame containing KOSPI200 data.
    consensus : pd.DataFrame
        DataFrame containing consensus data.
    transaction : pd.DataFrame
        DataFrame containing transaction volume data.
    ifrsSector : pd.DataFrame
        DataFrame containing IFRS sector data.
    size : pd.DataFrame
        DataFrame containing size information.
    stockFactorScoreTable : pd.DataFrame
        DataFrame containing stock factor scores.
    factorTable : pd.DataFrame
        DataFrame containing factor data including 'OPG'.
    esgData : pd.DataFrame
        DataFrame containing ESG data.
    esgClass : object
        An instance of the ESG class with the method GetESGScore.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the gathered and conditioned portfolio data.
    """

    filteredData = {}

    # Filter and process dataframes for the given date and align with KOSPI200 symbols
    dataFrames = {
        'consensus': consensus,
        'Amt_Avg_20D': transaction,
        'IFRSsector': IFRSsector,
        'size': size
        }

    for colName, df in dataFrames.items():
        tempDf = df[df['Dates'] == date].drop(columns=['Dates']).set_index(['Symbol', 'Name'])
        tempDf.columns = [colName]
        filteredData[colName] = tempDf.loc[kospi200atD]
        
    # Create portfolio dataframe
    portfolioDf = pd.DataFrame(index=tempDf.loc[kospi200atD].index)
    portfolioDf['ASCORE'] = StockFactorScoreTable['ASCORE']
    portfolioDf['SCORE'] = StockFactorScoreTable['SCORE']
    portfolioDf['OPG'] = StockFactorScoreTable['OPG']
    portfolioDf['EY_12M'] = StockFactorScoreTable['EY_12M']
    
    portfolioDf['consensus'] = filteredData['consensus']['consensus']
    portfolioDf['Amt_Avg_20D'] = filteredData['Amt_Avg_20D']['Amt_Avg_20D']
    portfolioDf['FGSector'] = filteredData['IFRSsector']['IFRSsector']
    portfolioDf['size'] = filteredData['size']['size']

    # ifrsSectorList = [sector for sector in set(portfolioDf['FGSector'].values.flatten())]
    
    portfolioDf = portfolioDf.reset_index().set_index('Symbol')
    
    portfolioDf['esgDB'] = esgData['DB']
    portfolioDf['esgSustin'] = esgData['Sustin']
    
    #portfolioDf = portfolioDf.reset_index().set_index(['Symbol', 'Name'])
    
    portfolioDf = GetESGScore(date, indexCap, portfolioDf, esgData['DB'], esgData['Sustin'])

    # sector = factorTable['Sector']
    # sector.columns = ['Sector']
    # portfolioDf['Sector'] = sector
    
    portfolioDf = ConditioningScore(portfolioDf)    

    return portfolioDf
    
def GetPortfolioLowerBound(portfolioDF, bound_ratio):
    
    temp_df = portfolioDF.copy(deep=True)
    
    mask = (temp_df['idx_weight'] - bound_ratio > 0)
    temp_df['lowerbound'] = np.where(mask, temp_df['idx_weight'] - bound_ratio, 0)

    mask = (temp_df['FGSector'] == '은행')
    mask |= (temp_df['FGSector'] == '보험')
    mask |= (temp_df['FGSector'] == '증권')
    temp_df['lowerbound'] = np.where(mask, temp_df['idx_weight']-0.001, temp_df['lowerbound'])

    mask = (temp_df['Amt_Avg_20D'] == 0)
    temp_df['lowerbound'] = np.where(mask, 0, temp_df['lowerbound'])

    mask = (temp_df['SCORE'].isna())
    temp_df['lowerbound'] = np.where(mask, 0, temp_df['lowerbound'])

    mask = (temp_df['SCORE'] == 0)
    temp_df['lowerbound'] = np.where(mask, 0, temp_df['lowerbound'])
    
    return temp_df['lowerbound']
    
def GetPortfolioUpperBound(portfolioDF, bound_ratio):
    
    temp_df = portfolioDF.copy(deep=True)
    temp_df['upperbound1'] = temp_df['idx_weight'] * 9 # 최대 9배 못넘어가게
    temp_df['upperbound2'] = temp_df['idx_weight'] + bound_ratio # 1~5 까지 조정
    temp_df['upperbound'] = temp_df[['upperbound1','upperbound2']].min(axis=1)

    mask = (temp_df['FGSector'] == '은행')
    mask |= (temp_df['FGSector'] == '보험')
    mask |= (temp_df['FGSector'] == '증권')
    temp_df['upperbound'] = np.where(mask, temp_df['idx_weight']-0.001, temp_df['upperbound'])

    mask = (temp_df['Amt_Avg_20D'] == 0)
    temp_df['upperbound'] = np.where(mask, 0, temp_df['upperbound'])
    
    mask = (temp_df['EsgScoreSum_rank'] > 150)
    temp_df['upperbound'] = np.where(mask, temp_df['lowerbound'], temp_df['upperbound'])
    
    return temp_df['upperbound']

def GetPortfolioBounds(portfolioDF, bound_ratio):
    """
    This function sets the lower and upper bounds for the portfolio weights.

    Parameters
    ----------
    portfolioDF : pd.DataFrame
        A DataFrame containing the portfolio data.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with updated lower and upper bounds for the portfolio weights.

    This function sets the lower and upper bounds for the portfolio weights. It first defines the lower and upper bounds for the '삼성전자' and 'SK하이닉스' stocks. Then, it updates the lower and upper bounds for the entire portfolio. Finally, it fills the missing 'ASCORE' values with -1000.

    Note:
    - The lower and upper bounds are defined based on the current portfolio weights.
    - The function sorts the portfolio weights in descending order before setting the bounds.
    """

    upper005930, lower005930 = -3, -3
    upper000660, lower000660 = 2, 1

    portfolioDF['lowerbound'] = GetPortfolioLowerBound(portfolioDF, bound_ratio)
    portfolioDF['upperbound'] = GetPortfolioUpperBound(portfolioDF, bound_ratio)

    portfolioDF.loc[('A005930', '삼성전자'), 'lowerbound'] = portfolioDF.loc[('A005930', '삼성전자'), 'idx_weight'] + lower005930
    portfolioDF.loc[('A005930', '삼성전자'), 'upperbound'] = portfolioDF.loc[('A005930', '삼성전자'), 'idx_weight'] + upper005930

    portfolioDF.loc[('A000660', 'SK하이닉스'), 'lowerbound'] = portfolioDF.loc[('A000660', 'SK하이닉스'), 'idx_weight'] + lower000660
    portfolioDF.loc[('A000660', 'SK하이닉스'), 'upperbound'] = portfolioDF.loc[('A000660', 'SK하이닉스'), 'idx_weight'] + upper000660

    portfolioDF['ASCORE'] = portfolioDF['ASCORE'].fillna(-1000)

    portfolioDF = portfolioDF.sort_values('idx_weight', ascending=False)

    return portfolioDF


def GetSectorConstraints(portfolioDF, ifrsSectorList, sector_upper_const, sector_lower_const):
    """
    This function constructs the sector constraint conditions for the portfolio optimization.

    Parameters:
    portfolioDF : pd.DataFrame
        A DataFrame containing the portfolio data.
    ifrsSectorList : list
        A list of sectors in the portfolio.
    sector_upper_const (float): The upper constraint for the sector weights.
    sector_lower_const (float): The lower constraint for the sector weights.

    Returns:
    tuple
        A tuple containing:
        - A_SectorBounds (list): List of arrays representing the sector upper bound constraints.
        - b_SectorBounds (list): List of values representing the sector upper bound constraints.
    """
    A_SectorBounds = []
    b_SectorBounds = []

    for col in ifrsSectorList:
        mask = portfolioDF['FGSector'] == col
        benchmarkSectorWeight = round(portfolioDF.loc[mask, 'idx_weight'].sum(), 2)

        boundTemp = np.where(mask, 1, 0)
        
        A_SectorBounds.append(boundTemp)
        b_SectorBounds.append(round(benchmarkSectorWeight + sector_upper_const, 2))
        A_SectorBounds.append(-boundTemp)

        if col == '반도체' :
            sector_lower_const_semi = -2.0
            sector_lower_const_semi = min(sector_lower_const_semi, sector_lower_const)
            lowerTemp = benchmarkSectorWeight + sector_lower_const_semi

        else :
            lowerTemp = benchmarkSectorWeight + sector_lower_const
        
        if lowerTemp < 0:
            lowerTemp = 0
        lowerTemp = -lowerTemp
        b_SectorBounds.append(round(lowerTemp, 2))

    return A_SectorBounds, b_SectorBounds

def GetSizeConstraints(portfolioDF, size_upper_const, size_lower_const):
    """
    This function constructs the size constraint conditions for the portfolio optimization.

    Parameters:
    portfolioDF : pd.DataFrame
        A DataFrame containing the portfolio data.
    size_upper_const (float): The upper constraint for the size weights.
    size_lower_const (float): The lower constraint for the size weights.

    Returns:
    tuple
        A tuple containing:
        - A_SizeBounds (list): List of arrays representing the size upper bound constraints.
        - b_SizeBounds (list): List of values representing the size upper bound constraints.
    """
    A_SizeBounds = []
    b_SizeBounds = []

    for s in ['코스피 대형주', '코스피 중형주', '코스피 소형주']:
        mask = portfolioDF['size'] == s
        benchmarkSizeWeight = round(portfolioDF.loc[mask, 'idx_weight'].sum(), 2)
        
        boundTemp = np.where(mask, 1, 0)
        
        A_SizeBounds.append(boundTemp)
        b_SizeBounds.append(round(benchmarkSizeWeight + size_upper_const, 2))
        
        A_SizeBounds.append(-boundTemp)
        lowerTemp = benchmarkSizeWeight + size_lower_const
        if lowerTemp < 0:
            lowerTemp = 0
        lowerTemp = -lowerTemp
        b_SizeBounds.append(round(lowerTemp, 2))

    return A_SizeBounds, b_SizeBounds

def GetConstraint(portfolioDF, sector_upper_const, sector_lower_const, size_upper_const, size_lower_const):
    """
    This function constructs the constraint conditions for the portfolio optimization by combining sector and size constraints.

    Parameters:
    portfolioDF : pd.DataFrame
        A DataFrame containing the portfolio data.
    ifrsSectorList : list
        A list of sectors in the portfolio.
    sector_upper_const (float): The upper constraint for the sector weights.
    sector_lower_const (float): The lower constraint for the sector weights.
    size_upper_const (float): The upper constraint for the size weights.
    size_lower_const (float): The lower constraint for the size weights.

    Returns:
    tuple
        A tuple containing:
        - A_UpperBound (list): List of arrays representing the upper bound constraints.
        - b_UpperBound (list): List of values representing the upper bound constraints.
    """
    ifrsSectorList = list(set(portfolioDF['FGSector'].values))
    A_SectorBounds, b_SectorBounds = GetSectorConstraints(portfolioDF, ifrsSectorList, sector_upper_const, sector_lower_const)
    A_SizeBounds, b_SizeBounds = GetSizeConstraints(portfolioDF, size_upper_const, size_lower_const)

    A_UpperBound = A_SectorBounds + A_SizeBounds
    b_UpperBound = b_SectorBounds + b_SizeBounds

    return A_UpperBound, b_UpperBound
    
    
def optimizePortfolio(portfolioDF, A_UpperBound, b_UpperBound):
    """
    This function optimizes the portfolio weights based on the given constraints.

    Parameters:
    portfolioDF : pd.DataFrame
        A DataFrame containing the portfolio data.
    A_UpperBound : list
        List of arrays representing the upper bound constraints.
    b_UpperBound : list
        List of values representing the upper bound constraints.

    Returns:
    pandas.DataFrame
        A DataFrame containing the optimized portfolio weights.

    This function optimizes the portfolio weights based on the given constraints.
    It first constructs the constraint conditions for the portfolio optimization.
    Then, it uses the `linprog` function from the `scipy.optimize` module to solve the linear programming problem.
    Finally, it updates the portfolio DataFrame with the optimized weights.

    Note:
    - The upper and lower bounds are defined based on the current portfolio weights.
    - The function sorts the portfolio weights in descending order before setting the bounds.
    """
    c = portfolioDF['SCORE'].values

    lowerbounds = portfolioDF['lowerbound'].values
    upperbounds = portfolioDF['upperbound'].values
    
    bounds = [(l, u) for l, u in zip(lowerbounds, upperbounds)]
    
    A_EQ = np.ones(portfolioDF.shape[0]).reshape(1, -1)
    b_EQ = [portfolioDF['idx_weight'].sum()]
    
    A_UpperBound = np.asarray(A_UpperBound)
    b_UpperBound = np.asarray(b_UpperBound)

    # options = {'tol': 0.01}
    
    res = linprog(-c,
                  A_ub=A_UpperBound,
                  b_ub=b_UpperBound,
                  A_eq=A_EQ,
                  b_eq=b_EQ,
                  bounds=bounds,
                  method='highs')
    if not res.success:
        print("Optimization failed:", res.message) 
    
    portfolioDF['opt'] = res.x
    
    return portfolioDF