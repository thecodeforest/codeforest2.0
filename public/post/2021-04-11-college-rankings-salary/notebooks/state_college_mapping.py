from datetime import datetime
import pandas as pd
import logging
import time
from typing import List
from tqdm import tqdm


start_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
logging.basicConfig(
    filename=f"state-college-mapping-{start_time}.log",
    format="%(levelname)s - %(asctime)s - %(filename)s - %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)


def format_state_college(
    state_colleges: List[pd.DataFrame], state_name: str
) -> pd.DataFrame:
    state_df = pd.DataFrame(columns=["state", "college_name"])
    for table in state_colleges:
        if isinstance(table, pd.core.frame.DataFrame):
            if "School" in table.columns:
                colleges = table[["School"]].rename(columns={"School": "college_name"})
                colleges["state"] = state_name
                state_df = state_df.append(colleges)
    return state_df


def collect_college_state_mappings():
    state_names = [
        "Alaska",
        "Alabama",
        "Arkansas",
        "American Samoa",
        "Arizona",
        "California",
        "Colorado",
        "Connecticut",
        "Delaware",
        "Florida",
        "Georgia",
        "Guam",
        "Hawaii",
        "Iowa",
        "Idaho",
        "Illinois",
        "Indiana",
        "Kansas",
        "Kentucky",
        "Louisiana",
        "Massachusetts",
        "Maryland",
        "Maine",
        "Michigan",
        "Minnesota",
        "Missouri",
        "Mississippi",
        "Montana",
        "North Carolina",
        "North Dakota",
        "Nebraska",
        "New Hampshire",
        "New Jersey",
        "New Mexico",
        "Nevada",
        "New York",
        "Ohio",
        "Oklahoma",
        "Oregon",
        "Pennsylvania",
        "Puerto Rico",
        "Rhode Island",
        "South Carolina",
        "South Dakota",
        "Tennessee",
        "Texas",
        "Utah",
        "Virginia",
        "Virgin Islands",
        "Vermont",
        "Washington",
        "Wisconsin",
        "West Virginia",
        "Wyoming",
    ]
    wiki_url = "https://en.wikipedia.org/wiki/List_of_colleges_and_universities_in_"
    state_names_fmt = [x.replace(" ", "_") for x in state_names]
    all_state_colleges_df = pd.DataFrame()
    for index, state in tqdm(enumerate(state_names_fmt)):
        if state == "Georgia":
            state += "_(U.S._state)"
        if state == "New_York":
            state += "_(state)"
        state_url = wiki_url + state
        try:
            state_colleges = pd.read_html(state_url, header=0)
            state_colleges_df = format_state_college(
                state_colleges=state_colleges, state_name=state_names[index]
            )
            all_state_colleges_df = all_state_colleges_df.append(state_colleges_df)
            time.sleep(1)
        except Exception:
            logger.error(
                f"Problem extract table from wikipedia for {state}", exc_info=True
            )
    return all_state_colleges_df
