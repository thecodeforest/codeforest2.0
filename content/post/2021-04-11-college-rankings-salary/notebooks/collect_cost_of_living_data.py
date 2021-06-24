# college_cost_of_living_data.py
import urllib
from bs4 import BeautifulSoup
import re
import pandas as pd
from datetime import datetime
from typing import List
import logging

start_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
logging.basicConfig(
    filename=f"college-rankings-pay-{start_time}.log",
    format="%(levelname)s - %(asctime)s - %(filename)s - %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)


def scrape_col_data() -> List[str]:
    base_url = "https://meric.mo.gov/data/cost-living-data-series"
    page = urllib.request.urlopen(base_url).read()
    soup = BeautifulSoup(page)
    col_data = soup.findAll("tbody")[0].findAll("tr")
    col_data_lst = str(col_data).split("</tr>")
    return col_data_lst


def format_col_data(col_data_lst: List[str]) -> pd.DataFrame:
    field_names = ["state", "cost_of_living"]
    regex_state = re.compile(">[0-9]{1,2}</td>\n<td>(.+?)\xa0<")
    regex_col_index = re.compile("\xa0</td>\n<td>([0-9]{2,3}.\d)</td>")
    all_states_col = list()
    for state in col_data_lst:
        try:
            state_name = re.search(regex_state, state).group(1)
            state_col = re.search(regex_col_index, state).group(1)
            row = [state_name, state_col]
            all_states_col.append(row)
        except Exception as e:
            logger.error(
                f"Problem extract table from wikipedia for {state}", exc_info=True
            )
    all_states_df = pd.DataFrame(all_states_col, columns=field_names)
    return all_states_df


def collect_col_data() -> pd.DataFrame():
    col_data_lst = scrape_col_data()
    col_df = format_col_data(col_data_lst=col_data_lst)
    return col_df
