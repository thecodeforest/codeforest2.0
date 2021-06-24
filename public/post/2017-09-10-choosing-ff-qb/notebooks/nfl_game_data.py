import nflgame
import pandas as pd
from itertools import product
from tqdm import tqdm
from typing import Tuple

GAME_YEARS = range(2013, 2016)
GAME_WEEKS = range(1, 17)
QUARTERBACKS = ("Joe Flacco", "Marcus Mariota", "Jameis Winston")


def format_passing_stats(
    year: int, week: int, players: nflgame.seq.GenPlayerStats, quarterbacks: Tuple[str]
):
    qb_list = list()
    for p in players.passing():
        player = " ".join(str(p.player).split(" ")[:2])
        if player in quarterbacks:
            qb_list.append([year, week, player, p.passing_tds, p.passing_yds])
    return qb_list


def collect_qb_stats() -> pd.DataFrame:
    qb_data = pd.DataFrame()
    for year, week in tqdm(product(GAME_YEARS, GAME_WEEKS)):
        games = nflgame.games(year, week)
        players = nflgame.combine_game_stats(games)
        qb_stats = format_passing_stats(year, week, players, quarterbacks=QUARTERBACKS)
        qb_data = qb_data.append(pd.DataFrame(qb_stats))
    qb_data.columns = ["year", "week", "player", "touchdowns", "passing_yds"]
    return qb_data