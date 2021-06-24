import urllib
from bs4 import BeautifulSoup
import re
import pandas as pd
from typing import List
import time
from tqdm import tqdm

N_PAGES = 20
FIELD_NAMES = ["name", "rank", "type", "pct_stem", "early_pay", "mid_pay"]


def scrape_pay_data(page_number: int) -> List:
    base_url = f"https://www.payscale.com/college-salary-report/bachelors/page/{str(page_number)}"
    page = urllib.request.urlopen(base_url).read()
    soup = BeautifulSoup(page)
    college_data = soup.findAll("tbody")[0].findAll("tr")
    college_data = [str(x) for x in college_data]
    return college_data


def format_pay_data(college_data: str) -> pd.DataFrame:
    regex_name = re.compile('<a href="/research/US/School=(.+?)/Salary')
    regex_rank = re.compile(
        '">Rank<!-- -->:</span><span class="data-table__value">(.+?)</span>'
    )
    regex_type = re.compile(
        '>School Type<!-- -->:</span><span class="data-table__value">(.+?)</span>'
    )
    regex_early_pay = re.compile(
        '>Early Career Pay<!-- -->:</span><span class="data-table__value">(.+?)</span>'
    )
    regex_mid_pay = re.compile(
        '>Mid-Career Pay<!-- -->:</span><span class="data-table__value">(.+?)</span>'
    )
    regex_pct_stem = re.compile(
        '>% STEM Degrees<!-- -->:</span><span class="data-table__value">(.+?)</span>'
    )
    all_college_data = list()
    # TO DO - ADD LOGGING
    for college in college_data:
        try:
            name = re.search(regex_name, college).group(1)
            rank = re.search(regex_rank, college).group(1)
            type_ = re.search(regex_type, college).group(1)
            early_pay = re.search(regex_early_pay, college).group(1)
            mid_pay = re.search(regex_mid_pay, college).group(1)
            pct_stem = re.search(regex_pct_stem, college).group(1)
            row = [name, rank, type_, pct_stem, early_pay, mid_pay]
            all_college_data.append(row)
        except Exception as e:
            print(e)
    college_df = pd.DataFrame(all_college_data, columns=FIELD_NAMES)
    return college_df


def collect_payscale_college_salary_data() -> pd.DataFrame:
    all_colleges_df = pd.DataFrame(columns=FIELD_NAMES)
    for page_number in tqdm(range(1, (N_PAGES + 1))):
        college_data = scrape_pay_data(page_number=page_number)
        college_data_df = format_pay_data(college_data=college_data)
        all_colleges_df = all_colleges_df.append(college_data_df)
        time.sleep(2)
    return all_colleges_df
