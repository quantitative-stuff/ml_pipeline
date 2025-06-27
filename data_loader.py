import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import date, timedelta
import os

def get_data(period1, period2):
    """
    Get data from the database for a given period.
    """
    host = '192.168.1.27'
    port = '3306'
    db_name = 'quantdb_maria'
    username = 'quantdb'
    password = 'QuantDb2023!'

    engine = create_engine(f"mysql+pymysql://{username}:{password}@{host}:{port}/{db_name}")
    
    with engine.connect() as conn:
        query_com = f"select * from fn_COM where (Dates>='{period1}' and Dates<='{period2}')"
        data_com = pd.read_sql(query_com, conn)

        query_cia = f"select * from fn_CIA where Dates>='{period1}' and Dates<='{period2}'"
        data_cia = pd.read_sql(query_cia, conn)

        query_ssc = f"select * from fn_SSC where Dates>='{period1}' and Dates<='{period2}'"
        data_ssc = pd.read_sql(query_ssc, conn)

        query_nfr_ifrs = f"select * from fn_NFR_IFRS where Dates>='{period1}' and Dates<='{period2}'"
        data_nfr_ifrs = pd.read_sql(query_nfr_ifrs, conn)

        query_nfs_ifrs = f"select * from fn_NFS_IFRS where Dates>='{period1}' and Dates<='{period2}'"
        data_nfs_ifrs = pd.read_sql(query_nfs_ifrs, conn)

        query_con = f"select * from fn_CON where Dates>='{period1}' and Dates<='{period2}'"
        data_con = pd.read_sql(query_con, conn)

    return data_com, data_cia, data_ssc, data_nfr_ifrs, data_nfs_ifrs, data_con

if __name__ == '__main__':
    end_date = date(2025, 5, 30)
    start_date = end_date - timedelta(days=30)
    
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    data_com, data_cia, data_ssc, data_nfr_ifrs, data_nfs_ifrs, data_con = get_data(start_date_str, end_date_str)

    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)

    # You can now use these dataframes for your factor pipeline
    # For example, you can save them to pickle files
    data_com.to_pickle(os.path.join(output_dir, 'data_com.pkl'))
    data_cia.to_pickle(os.path.join(output_dir, 'data_cia.pkl'))
    data_ssc.to_pickle(os.path.join(output_dir, 'data_ssc.pkl'))
    data_nfr_ifrs.to_pickle(os.path.join(output_dir, 'data_nfr_ifrs.pkl'))
    data_nfs_ifrs.to_pickle(os.path.join(output_dir, 'data_nfs_ifrs.pkl'))
    data_con.to_pickle(os.path.join(output_dir, 'data_con.pkl'))

    print(f"Data from {start_date_str} to {end_date_str} has been downloaded and saved to the '{output_dir}' directory.")