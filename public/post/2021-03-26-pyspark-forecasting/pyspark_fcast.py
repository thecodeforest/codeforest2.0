import json
import pandas as pd
import numpy as np
import re
import os
from typing import List
from pathlib import Path
import logging
from datetime import datetime
import argparse

from fbprophet import Prophet

from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, PandasUDFType, lit
from pyspark.sql.types import (
    FloatType,
    StructType,
    StructField,
    IntegerType,
    DateType,
    StringType,
)
from pyspark_ts_fcast.fcast_data_frame import FcastDataFrame

logging.basicConfig(format='%(levelname)s - %(asctime)s - %(filename)s - %(message)s',
                    level=logging.INFO,
                    filename = 'run_{start_time}.log'.format(start_time = datetime.now().strftime('%Y-%m-%d %H-%M-%S'))                    
                    )
                  
# python3 pyspark_fcast.py --forecast-config-file 'config/conf.json'

def read_args() -> argparse.Namespace:
    """[summary]

    Returns:
        argparse.Namespace: [description]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast-config-file", type=str)
    return parser.parse_args()


def log_input_params(config: dict) -> None:
    """[summary]

    Args:
        config (dict): [description]
    """
    params = pd.json_normalize(config).transpose()
    [logging.info('input params:' + x[0] + '-' +str(x[1])) 
     for x in zip(params.index, params.iloc[:,0])]


def clean_names(df: pd.DataFrame) -> pd.DataFrame:
    """[summary]

    Args:
        df (pd.DataFrame): [description]

    Returns:
        pd.DataFrame: [description]
    """
    cols = df.columns
    cols = [re.sub(r'(?<!^)(?=[A-Z])', '_', x).lower() for x in cols] # removes camel case 
    cols = [re.sub(r'_{2,}', '_', x) for x in cols] # replaces any double underscores with single underscore
    cols = [re.sub(r'\s', '', x) for x in cols] # removes spaces in middle 
    cols = [re.sub(r'\.', '_', x) for x in cols] # replaces periods with an underscore
    df.columns = cols
    return df 

def prep_for_prophet(df: pd.DataFrame, yvar_field: str, date_field: str, group_fields: List[str]) -> pd.DataFrame:
    """[summary]

    Args:
        df (pd.DataFrame): [description]
        yvar_field (str): [description]
        date_field (str): [description]
        group_fields (List[str]): [description]

    Returns:
        pd.DataFrame: [description]
    """
    fields = df.columns.tolist()
    cap_value_index = [index for index, value in enumerate(['cap_value' in x for x in fields]) if value]
    floor_value_index = [index for index, value in enumerate(['floor_value' in x for x in fields]) if value]
    if cap_value_index and floor_value_index:
        df = df.rename(columns={fields[cap_value_index[0]]: 'cap', 
                                fields[floor_value_index[0]]: 'floor'})
        group_fields = group_fields + ['cap', 'floor']                                
    df = df[group_fields + [date_field] + [yvar_field]]
    df = df.rename(columns={date_field: 'ds', yvar_field: 'y'})
    df['ds'] = pd.to_datetime(df['ds'])
    return df    


def run_forecast(keys, df):
    """[summary]

    Args:
        keys ([type]): [description]
        df ([type]): [description]

    Returns:
        [type]: [description]
    """
    fields = [
        "ds",
        "store",
        "dept",
        "yhat_lower",
        "yhat_upper",
        "yhat",
    ]
    store, dept = keys
    cap = df['cap'][0]
    floor = df['floor'][0]
    model = Prophet(
            interval_width=0.95,
            growth="logistic",
            yearly_seasonality=True,
            seasonality_mode="additive",
        )
    model.add_country_holidays(country_name='US')
    model.fit(df)
    future_df = model.make_future_dataframe(
            periods=13, freq="W-FRI", include_history=False
        )
    future_df['cap'] = cap
    future_df['floor'] = floor
    results_df = model.predict(future_df)
    results_df["store"] = store
    results_df["dept"] = dept
    results_df = results_df[fields]
    return results_df


def bind_actuals_and_forecast(actuals_df: pd.DataFrame, forecast_df: pd.DataFrame, yvar_field: str, group_fields: List[str], date_field: str, exp_yvar_field: bool = True) -> pd.DataFrame:
    """[summary]

    Args:
        actuals_df (pd.DataFrame): [description]
        forecast_df (pd.DataFrame): [description]
        yvar_field (str): [description]
        group_fields (List[str]): [description]
        date_field (str): [description]
        exp_yvar_field (bool, optional): [description]. Defaults to True.

    Returns:
        pd.DataFrame: [description]
    """
    # prep actuals
    actuals_df['part'] = 'actuals'
    actuals_df = actuals_df.rename(columns={'y': yvar_field})
    if exp_yvar_field:
        actuals_df[yvar_field] = actuals_df[yvar_field].apply(lambda x: np.expm1(x))
    if any(['cap' in x for x in actuals_df.columns]):
        del actuals_df['cap']
        del actuals_df['floor']
    # prep forecast    
    forecast_df = forecast_df.rename(columns={'yhat': yvar_field, 
                                             'yhat_lower': f'{yvar_field}_lb', 
                                             'yhat_upper': f'{yvar_field}_ub'}
                                            )
    bound_df = pd.concat([actuals_df, forecast_df]) 
    bound_df = (bound_df
                    .rename(columns={'ds': date_field})
                    .sort_values(group_fields + [date_field])
                    )
    return bound_df    
    
def main():
    args = read_args()
    with open(args.forecast_config_file) as f:
        config = json.load(f)

    log_input_params(config=config)

    input_data_path = config['input_data_path']   
    output_data_path =  config["output_data_dir"]
    fcast_params = config['fcast_parameters']
    group_fields = fcast_params['group_fields']
    date_field = fcast_params['date_field']
    yvar_field = fcast_params['yvar_field']
    ts_frequency = fcast_params['ts_frequency']
    fcast_floor = fcast_params['forecast_floor']
    fcast_cap = fcast_params['forecast_cap']

    sales_df = pd.read_csv(Path.cwd() / input_data_path / 'sales_data_raw.csv')
    sales_df = clean_names(sales_df)
    df_small = sales_df[sales_df['store'].isin([1, 2])]

    fcast_df = FcastDataFrame(df=df_small, group_fields=group_fields,date_field=date_field,
                          yvar_field=yvar_field,ts_frequency=ts_frequency)
    fcast_df.replace_negative_value_with_zero()
    fcast_df.pad_missing_values()
    fcast_df.fill_missing_values(method='interpolate')
    fcast_df.add_forecast_bounds(floor_multiplier=fcast_floor, cap_multiplier=fcast_cap)
    fcast_df.log_transform_values('weekly_sales_prep', 'floor_value', 'cap_value')
    fcast_df_trans = fcast_df.return_transformed_df()     

    fcast_df_prophet_input = prep_for_prophet(df=fcast_df_trans,
                                    yvar_field='weekly_sales_prep_log1p',
                                    date_field=date_field,
                                    group_fields=group_fields)

    # Prep for Spark
    INPUT_SCHEMA = StructType(
        [
            StructField("store", IntegerType(), True),
            StructField("dept", IntegerType(), True),
            StructField("cap", FloatType(), True),
            StructField("floor", FloatType(), True),
            StructField("ds", DateType(), True),
            StructField("y", FloatType(), True),
        ]
    )
    OUTPUT_SCHEMA = StructType(
        [
            StructField("ds", DateType(), True),
            StructField("store", IntegerType(), True),
            StructField("dept", IntegerType(), True),
            StructField("yhat_lower", FloatType(), True),
            StructField("yhat_upper", FloatType(), True),
            StructField("yhat", FloatType(), True),
        ]
    )       
    os.environ["JAVA_HOME"] = config['java_home']
    SPARK = (
        SparkSession.builder.master("local[*]")
        .appName(config['app_name'])
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )                             
    # 
    fcast_spark_prophet_input = SPARK.createDataFrame(fcast_df_prophet_input, schema=INPUT_SCHEMA)
    fcast_df_prophet_output = (fcast_spark_prophet_input
                .groupBy(group_fields)
                .applyInPandas(func=run_forecast, schema=OUTPUT_SCHEMA)
                .withColumn("part", lit("forecast"))
                .withColumn("fcast_date", lit(datetime.now().strftime("%Y-%m-%d")))
                .toPandas()
                )
    fcast_df_prophet_output = fcast_df_prophet_output.apply(lambda x: round(np.expm1(x)) if 'yhat' in x.name else x)         

    df = bind_actuals_and_forecast(actuals_df=fcast_df_prophet_input, 
                                    forecast_df=fcast_df_prophet_output,
                                    yvar_field=yvar_field,
                                    group_fields=group_fields,
                                    date_field=date_field) 
    df.to_csv(Path.cwd() / output_data_path / 'sales_data_raw.csv', index=False)    

if __name__ == "__main__":
    main()