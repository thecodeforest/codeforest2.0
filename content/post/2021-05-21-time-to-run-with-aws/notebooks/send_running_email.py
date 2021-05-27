import sys
import boto3
import json
import logging
from datetime import datetime
from typing import List

S3_BUCKET = "running-weather-data"
SENDER = "dpuresearch1@gmail.com"
RECIPIENT = "markleboeuf10@gmail.com"
AWS_REGION = "us-west-2"
SUBJECT = "Best Times to Run Today"
CHARSET = "UTF-8"
RUNNING_CONDS = {
    "hour": {"min_hour": 13, "max_hour": 19},
    "status": ["Rain", "Snow", "Smoke"],
    "wind_speed": {"min_speed": 0, "max_speed": 30},
    "temp": {"min_temp": 30, "max_temp": 90},
}

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def find_most_recent_data_path(s3_bucket: str) -> str:
    today_dt = datetime.now().strftime("%Y-%m-%d")
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(s3_bucket)
    existing_data = [
        x.key
        for x in bucket.objects.all()
        if str(x.key).startswith("data") and str(x.key).endswith("-running-times.json")
    ]
    most_recent_dt = max(
        [x.split("/")[-1].replace("-running-times.json", "") for x in existing_data]
    )
    assert most_recent_dt == today_dt, "No Data Found for Today's Date"
    s3_key = [x for x in existing_data if most_recent_dt in x][0]
    return s3_key


def read_json_from_s3(s3_bucket: str, s3_key: str) -> str:
    s3 = boto3.resource("s3")
    obj = s3.Object(s3_bucket, s3_key)
    file_content = obj.get()["Body"].read().decode("utf-8")
    json_content = json.loads(file_content)
    return json_content


def _convert_to_12hr_format(hr: int) -> str:
    return datetime.strptime(str(hr), "%H").strftime("%I:%M %p").strip("0")


def format_run_times(run_times: List[dict]) -> str:
    if run_times:
        hour_fmt = [
            f"<b>{_convert_to_12hr_format(x.get('hour'))}:</b>" for x in run_times
        ]
        temp_fmt = [f"{round(x.get('temp'))}F with" for x in run_times]
        wind_speed_fmt = [
            f"wind at {round(x.get('wind_speed'))} mph" for x in run_times
        ]
        status_fmt = [f"and {x.get('status').lower()}" for x in run_times]
        fmt_msg = zip(hour_fmt, temp_fmt, wind_speed_fmt, status_fmt)
        fmt_msg_list = [" ".join(x) for x in fmt_msg]
        return fmt_msg_list
    else:
        return ["No Times to Run Today!"]


def is_time_for_run(weather_hour: dict) -> bool:
    is_time = (
        RUNNING_CONDS["hour"]["min_hour"]
        <= weather_hour["hour"]
        <= RUNNING_CONDS["hour"]["max_hour"]
    )
    is_temp = (
        RUNNING_CONDS["temp"]["min_temp"]
        <= weather_hour["temp"]
        <= RUNNING_CONDS["temp"]["max_temp"]
    )
    is_wind = (
        RUNNING_CONDS["wind_speed"]["min_speed"]
        <= weather_hour["wind_speed"]
        <= RUNNING_CONDS["wind_speed"]["max_speed"]
    )
    is_status = weather_hour["status"] not in RUNNING_CONDS["status"]
    if all([is_time, is_temp, is_wind, is_status]):
        return True
    else:
        return False


def lambda_handler(event, context):
    # generate s3 key for most recent weather data
    running_data_path = find_most_recent_data_path(s3_bucket=S3_BUCKET)
    # read as str
    hourly_weather_data = read_json_from_s3(
        s3_bucket=S3_BUCKET, s3_key=running_data_path
    )
    # convert to dict and extract weather data
    hourly_data_dict = eval(hourly_weather_data)["weather_data"]

    # True or False filter based on hour, temperature, windspeed criteria
    run_time_bool = [is_time_for_run(x) for x in hourly_data_dict]

    # applies weather criteria and filters only to hours where critera are met
    run_times = [
        time
        for (time, time_to_run) in zip(hourly_data_dict, run_time_bool)
        if time_to_run
    ]
    running_msg_lst = format_run_times(run_times)
    running_msg_str = "<p>" + "<br/>".join(running_msg_lst) + "</p>"
    running_msg = f"""<html>
                        <head></head>
                        <body>
                        <h1>Best Times to Run</h1>
                        {running_msg_str}
                        </body>
                        </html>
                        """
    try:
        client = boto3.client("ses", region_name=AWS_REGION)
        response = client.send_email(
            Destination={
                "ToAddresses": [
                    RECIPIENT,
                ]
            },
            Message={
                "Body": {
                    "Html": {
                        "Charset": CHARSET,
                        "Data": running_msg,
                    },
                },
                "Subject": {
                    "Charset": CHARSET,
                    "Data": SUBJECT,
                },
            },
            Source=SENDER,
        )
    except Exception as exp:
        exception_type, exception_value, exception_traceback = sys.exc_info()
        err_msg = json.dumps(
            {"errorType": exception_type.__name__, "errorMessage": str(exception_value)}
        )
        logger.error(err_msg)
