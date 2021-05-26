import os
import sys
from typing import List
import json
from datetime import datetime
import logging

import pytz
import requests
import boto3

S3_BUCKET = "running-weather-data"
WEATHER_API_KEY = "6e2aa8894df5d3ca386734b2b640472b"
LOCATION_LATITUDE = "45.493748"
LOCATION_LONGITUDE = "-122.803087"

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def retrieve_weather_data(units_of_measure: str) -> dict:
    api_key = os.environ["WEATHER_API_KEY"]
    lat = os.environ["LOCATION_LATITUDE"]
    lon = os.environ["LOCATION_LONGITUDE"]
    base_url = "https://api.openweathermap.org/data/2.5/onecall?"
    url = f"{base_url}lat={lat}&lon={lon}&appid={api_key}&units={units_of_measure}"
    response = requests.get(url)
    weather_data = json.loads(response.text)
    return weather_data


def parse_weather_data(weather_hour: dict) -> dict:
    hour = datetime.fromtimestamp(
        weather_hour["dt"], pytz.timezone("America/Los_Angeles")
    ).hour
    temp = weather_hour["temp"]
    wind_speed = weather_hour["wind_speed"]
    weather_status = weather_hour["weather"][0]
    status = weather_status["main"]
    return {"hour": hour, "temp": temp, "wind_speed": wind_speed, "status": status}


def is_today_weather(weather_hour: dict, timezone: str = "America/Los_Angeles") -> bool:
    weather_fmt = "%Y-%m-%d"
    today_dt = datetime.now().strftime(weather_fmt)
    weather_dt = datetime.fromtimestamp(weather_hour["dt"], pytz.timezone(timezone))
    if weather_dt.strftime(weather_fmt) == today_dt:
        return True
    else:
        return False


def _generate_s3_path() -> str:
    year, month, day = datetime.now().strftime("%Y-%m-%d").split("-")
    s3_path = f"data/{year}-{month}-{day}-running-times.json"
    return s3_path


def save_json_to_s3(json_data: dict, s3_bucket: str) -> None:
    s3 = boto3.resource("s3")
    response = s3.Object(s3_bucket, _generate_s3_path()).put(
        Body=(bytes(json.dumps(json_data).encode("UTF-8")))
    )
    if response.get("HTTPStatusCode") == 200:
        print(f"Data successfully landed")


def lambda_handler(event, context):
    try:
        # retrieve weather forecast from OpenWeatherAPI
        weather_data = retrieve_weather_data(units_of_measure="imperial")
        # extract hourly forecast
        hourly_data = weather_data["hourly"]
        # filter to only today's forecast
        today_weather_bool = [is_today_weather(x) for x in hourly_data if x]
        # extract fields relevant to deciding if run outside
        hourly_data = [parse_weather_data(x) for x in hourly_data]
        # filter to today's hourly data
        today_hourly_data = [
            today_weather
            for (today_weather, is_today) in zip(hourly_data, today_weather_bool)
            if is_today
        ]
        # convert all data to dictionary
        hourly_data_dict = {"weather_data": today_hourly_data}
        # save hourly weather data to S3 Bucket as .json
        save_json_to_s3(json_data=json.dumps(hourly_data_dict), s3_bucket=S3_BUCKET)
        return {"statusCode": 200, "body": json.dumps(hourly_data_dict)}
    except Exception as exp:
        exception_type, exception_value, exception_traceback = sys.exc_info()
        err_msg = json.dumps(
            {"errorType": exception_type.__name__, "errorMessage": str(exception_value)}
        )
        logger.error(err_msg)
