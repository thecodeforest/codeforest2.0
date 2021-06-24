import os
import pandas as pd
import polyline
import googlemaps
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('GOOGLE_API_KEY')

def extract_polyline(coords: dict) -> pd.DataFrame:
    gmaps_polyline = coords["overview_polyline"]["points"]
    polyline_df = pd.DataFrame(polyline.decode(gmaps_polyline))
    polyline_df.columns = ["lat", "lng"]
    polyline_df["path_order"] = range(1, polyline_df.shape[0] + 1)
    return polyline_df


def create_travel_path(
    route_df: pd.DataFrame, travel_mode: str = "walking"
) -> pd.DataFrame:
    gmaps = googlemaps.Client(key=API_KEY)
    out_route_df = pd.DataFrame()
    for row in route_df.itertuples():
        coords = gmaps.directions(
            origin=[row.start_lat, row.start_lng],
            destination=[row.end_lat, row.end_lng],
            mode=travel_mode,
        )
        coords_df = extract_polyline(coords=coords[0])
        coords_df["location_index"] = row.location_index
        coords_df["travel_time"] = coords[0]["legs"][0]["duration"]["value"]
        coords_df["miles"] = coords[0]["legs"][0]["distance"]["text"]
        coords_df["route_order"] = row.route_order
        out_route_df = out_route_df.append(coords_df)
    out_route_df = out_route_df.reset_index(drop=True)
    return out_route_df
