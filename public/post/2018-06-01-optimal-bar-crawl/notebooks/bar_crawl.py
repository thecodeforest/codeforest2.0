import urllib
from bs4 import BeautifulSoup
import re
from typing import List
import googlemaps
from tqdm import tqdm
import pandas as pd

API_KEY = "<GOOGLE-MAPS-API-KEY"


def find_best_bars() -> str:
    base_url = "http://www.oregonlive.com/dining/index.ssf/2014/10/portlands_100_best_bars_bar_ta.html"
    page = urllib.request.urlopen(base_url)
    soup = BeautifulSoup(page, "html.parser")
    bar_descriptors = soup.find_all("div", class_="entry-content")
    bar_descriptors = str(bar_descriptors).split("<p>")[0]
    best_bars_raw_lst = re.findall(r"\<strong>(.*?)</strong>", bar_descriptors)
    return best_bars_raw_lst


def clean_bar_names(raw_bar_lst: str) -> List[str]:
    # exclude emphasis tags
    best_bars = [re.sub(r"<em> (.*?)</em>", "", x) for x in raw_bar_lst]
    # exclude number included in bar name
    best_bars = [re.sub(r"No. \d+ --", "", x).strip() for x in best_bars]
    # exclude headers in all caps
    best_bars = [x for x in best_bars if not x.isupper()]
    # exclude all lower case tags
    best_bars = [x for x in best_bars if not x.islower()]
    # exclude bold tags in html
    best_bars = [x.replace("&amp;", "&") for x in best_bars]
    # exclude other emphasis tags
    best_bars = [re.sub(r": <em>(.*?)</em>", "", x) for x in best_bars]
    # strip colons
    best_bars = [x.replace(":", "") for x in best_bars]
    # exclude blanks
    best_bars = [x for x in best_bars if x]
    return best_bars


def geocode_best_portland_bars(bar_names: List[str]) -> pd.DataFrame:
    best_bars_lst = find_best_bars()
    bar_names_clean = clean_bar_names(raw_bar_lst=best_bars_lst)
    bar_names = bar_names_clean
    bar_names = [f"{x}, Portland, OR" for x in bar_names]
    gmaps = googlemaps.Client(key=API_KEY)
    geocoded_bars_lst = []
    for name in tqdm(bar_names):
        geocode_result = gmaps.geocode(name)
        lat_lng = geocode_result[0].get("geometry").get("location")
        lat, lng = lat_lng.get("lat"), lat_lng.get("lng")
        geocoded_bars_lst.append([name, lat, lng])
    geocoded_bars_df = pd.DataFrame(geocoded_bars_lst)
    geocoded_bars_df.columns = ["name", "lat", "lng"]
    return geocoded_bars_df
