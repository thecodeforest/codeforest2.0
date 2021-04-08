import argparse
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from fbprophet import Prophet
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql.types import DateType, FloatType, IntegerType, StructField, StructType
from fcast_helpers.fcast_data_frame import FcastDataFrame

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(filename)s - %(message)s",
    level=logging.INFO,
    filename="run_{start_time}.log".format(
        start_time=datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    ),
)


def read_args() -> argparse.Namespace:
    """Read Forecasting arguments

    Returns:
        argparse.Namespace: argparse Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast-config-file", type=str)
    return parser.parse_args()


def log_input_params(config: dict) -> None:
    """Logs all parameters in configuration file

    Args:
        config (dict): Configuration parameters for forecast and data
    """
    params = pd.json_normalize(config).transpose()
    [
        logging.info("input params:" + x[0] + "-" + str(x[1]))
        for x in zip(params.index, params.iloc[:, 0])
    ]
    return None


def clean_names(df: pd.DataFrame) -> pd.DataFrame:
    """Applies the following transformations to column names:
        - Removes camel case
        - Replaces any double underscore with single underscore
        - Removes spaces in the middle of a string name
        - Replaces periods with underscores

    Args:
        df (pd.DataFrame): Dataframe with untransformed column names

    Returns:
        pd.DataFrame: Dataframe with transformed column names
    """
    cols = df.columns
    cols = [re.sub(r"(?<!^)(?=[A-Z])", "_", x).lower() for x in cols]
    cols = [re.sub(r"_{2,}", "_", x) for x in cols]
    cols = [re.sub(r"\s", "", x) for x in cols]
    cols = [re.sub(r"\.", "_", x) for x in cols]
    df.columns = cols
    return df


def prep_for_prophet(
    df: pd.DataFrame, yvar_field: str, date_field: str, group_fields: List[str]
) -> pd.DataFrame:
    """Renames key field names to be compatible with Prophet Forecasting API

    Args:
        df (pd.DataFrame): Contains data that will be used to generate forecasting
        yvar_field (str): outcome ("y") field name
        date_field (str): date field name
        group_fields (List[str]): grouping fields. These are often
                represented by attributes of each unit
                (e.g., store id, product id, etc.).

    Returns:
        pd.DataFrame: Data with compatible field names
    """
    fields = df.columns.tolist()
    cap_value_index = [
        index for index, value in enumerate(["cap_value" in x for x in fields]) if value
    ]
    floor_value_index = [
        index
        for index, value in enumerate(["floor_value" in x for x in fields])
        if value
    ]
    if cap_value_index and floor_value_index:
        df = df.rename(
            columns={
                fields[cap_value_index[0]]: "cap",
                fields[floor_value_index[0]]: "floor",
            }
        )
        group_fields = group_fields + ["cap", "floor"]
    df = df[group_fields + [date_field] + [yvar_field]]
    df = df.rename(columns={date_field: "ds", yvar_field: "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def run_forecast(keys, df):
    """Generate time-series forecast

    Args:
        keys: Grouping keys
        df: Spark Dataframe
    """
    fields = ["ds", "store", "dept", "yhat_lower", "yhat_upper", "yhat"]
    store, dept = keys
    cap = df["cap"][0]
    floor = df["floor"][0]
    model = Prophet(
        interval_width=0.95,
        growth="logistic",
        yearly_seasonality=True,
        seasonality_mode="additive",
    )
    model.add_country_holidays(country_name="US")
    model.fit(df)
    future_df = model.make_future_dataframe(
        periods=13, freq="W-FRI", include_history=False
    )
    future_df["cap"] = cap
    future_df["floor"] = floor
    results_df = model.predict(future_df)
    results_df["store"] = store
    results_df["dept"] = dept
    results_df = results_df[fields]
    return results_df


def bind_actuals_and_forecast(
    actuals_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    yvar_field: str,
    group_fields: List[str],
    date_field: str,
    exp_yvar_field: bool = True,
) -> pd.DataFrame:
    """Binds historical and Prophet forecasting data together
       into a single DataFrame.

    Args:
        actuals_df (pd.DataFrame): Historical data
        forecast_df (pd.DataFrame): Prophet Forecast data
        yvar_field (str): outcome variable
        group_fields (List[str]): grouping fields. These are often
                represented by attributes of each unit
                (e.g., store id, product id, etc.)
        date_field (str): date field
        exp_yvar_field (bool, optional): Flags if the outcome variable
        was log transformed. If set to True, the transformation is inverted
        back to the original scale. Defaults to True.

    Returns:
        pd.DataFrame: Complete dataset with original field names and scales
    """
    # prep actuals
    actuals_df["part"] = "actuals"
    actuals_df = actuals_df.rename(columns={"y": yvar_field})
    if exp_yvar_field:
        actuals_df[yvar_field] = actuals_df[yvar_field].apply(lambda x: np.expm1(x))
    if any(["cap" in x for x in actuals_df.columns]):
        del actuals_df["cap"]
        del actuals_df["floor"]
    # prep forecast
    forecast_df = forecast_df.rename(
        columns={
            "yhat": yvar_field,
            "yhat_lower": f"{yvar_field}_lb",
            "yhat_upper": f"{yvar_field}_ub",
        }
    )
    bound_df = pd.concat([actuals_df, forecast_df])
    bound_df = bound_df.rename(columns={"ds": date_field}).sort_values(
        group_fields + [date_field]
    )
    return bound_df


def main():
    args = read_args()
    with open(args.forecast_config_file) as f:
        config = json.load(f)

    log_input_params(config=config)
    # fcast parameters
    input_data_path = config["input_data_path"]
    fcast_params = config["fcast_parameters"]
    group_fields = fcast_params["group_fields"]
    date_field = fcast_params["date_field"]
    yvar_field = fcast_params["yvar_field"]
    ts_frequency = fcast_params["ts_frequency"]
    fcast_floor = fcast_params["forecast_floor"]
    fcast_cap = fcast_params["forecast_cap"]
    min_obs_threshold = fcast_params["min_obs_count"]
    # spark parameters
    spark_n_threads = str(config["spark_n_threads"])
    java_home = config["java_home"]

    sales_df = pd.read_csv(input_data_path)
    sales_df = clean_names(sales_df)

    fcast_df = FcastDataFrame(
        df=sales_df,
        group_fields=group_fields,
        date_field=date_field,
        yvar_field=yvar_field,
        ts_frequency=ts_frequency,
    )
    fcast_df.filter_groups_min_obs(min_obs_threshold=min_obs_threshold)
    fcast_df.replace_negative_value_with_zero()
    fcast_df.pad_missing_values()
    fcast_df.filter_groups_max_missing_streak(max_streak=4)
    fcast_df.fill_missing_values()
    fcast_df.add_forecast_bounds(floor_multiplier=fcast_floor, cap_multiplier=fcast_cap)
    fcast_df.log_transform_values(yvar_field, "floor_value", "cap_value")
    fcast_df_trans = fcast_df.return_transformed_df()

    fcast_df_prophet_input = prep_for_prophet(
        df=fcast_df_trans,
        yvar_field="weekly_sales_log1p",
        date_field=date_field,
        group_fields=group_fields,
    )

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
    os.environ["JAVA_HOME"] = config["java_home"]
    SPARK = (
        SparkSession.builder.master("local[*]")
        .appName(config["app_name"])
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    #
    fcast_spark_prophet_input = SPARK.createDataFrame(
        fcast_df_prophet_input, schema=INPUT_SCHEMA
    )
    fcast_df_prophet_output = (
        fcast_spark_prophet_input.groupBy(group_fields)
        .applyInPandas(func=run_forecast, schema=OUTPUT_SCHEMA)
        .withColumn("part", lit("forecast"))
        .withColumn("fcast_date", lit(datetime.now().strftime("%Y-%m-%d")))
        .toPandas()
    )
    fcast_df_prophet_output = fcast_df_prophet_output.apply(
        lambda x: round(np.expm1(x)) if "yhat" in x.name else x
    )

    df = bind_actuals_and_forecast(
        actuals_df=fcast_df_prophet_input,
        forecast_df=fcast_df_prophet_output,
        yvar_field=yvar_field,
        group_fields=group_fields,
        date_field=date_field,
    )
    df.to_csv(Path.cwd() / "sales_data_forecast.csv", index=False)


if __name__ == "__main__":
    main()
