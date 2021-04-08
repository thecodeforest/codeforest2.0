import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class FcastDataFrame:
    """Use for pre-forecasting data transformations"""

    def __init__(
        self,
        df: pd.DataFrame,
        group_fields: List[str],
        date_field: str,
        yvar_field: str,
        ts_frequency: str,
    ):
        """
        Args:
            df (pd.DataFrame): dataframe with to be forecasted data
            group_fields (List[str]): grouping fields. These are often
                represented by attributes of each unit
                (e.g., store id, product id, etc.).
            date_field (str): date field
            yvar_field (str): outcome ("y") field
            ts_frequency (str): granularity of the data. For example,
                data that is recorded on a weekly basis, every Friday will
                be "W-FRI". Note that sub-day level (e.g, hourly, minute)
                data is not supported.
        """
        self.df = df
        self.group_fields = group_fields
        self.date_field = date_field
        self.yvar_field = yvar_field
        self.ts_frequency = ts_frequency

    def filter_groups_min_obs(self, min_obs_threshold: int):
        """Filters groups based on some minimum number of observations 
           required for forecasting

        Args:
            min_obs_threshold (int): removes all groups with less obsevations than 
                                     this threshold
        """
        n_unique_groups = self.df[self.group_fields].drop_duplicates().shape[0]
        min_obs_filter_df = (
            self.df.groupby(self.group_fields)[self.yvar_field]
            .count()
            .reset_index()
            .rename(columns={self.yvar_field: "obs_count"})
            .query(f"obs_count > {str(min_obs_threshold)}")
            .drop(columns="obs_count")
        )
        n_remaining_groups = min_obs_filter_df.shape[0]
        df = pd.merge(self.df, min_obs_filter_df, how="inner", on=self.group_fields)
        self.df = df
        logger.info("N groups dropped: {}".format(n_unique_groups - n_remaining_groups))

    def replace_negative_value_with_zero(self):
        """Replaces all negative values with zero"""
        logger.info("Replacing negative values with nan")
        self.df[self.yvar_field] = [0 if x < 0 else x for x in self.df[self.yvar_field]]

    def pad_missing_values(self):
        """Fills in missing date values depending on min/max of date by group"""

        logger.info("Padding missing values")
        min_max_dt_df = (
            self.df.groupby(self.group_fields)
            .agg(dt_min=(self.date_field, "min"), dt_max=(self.date_field, "max"))
            .reset_index()
        )
        df_padded_by_group = pd.DataFrame(None)

        for index, row in enumerate(min_max_dt_df.itertuples()):
            date_range = [
                x.strftime("%Y-%m-%d")
                for x in pd.date_range(row.dt_min, row.dt_max, freq=self.ts_frequency)
            ]
            group_values = min_max_dt_df.iloc[index][self.group_fields]
            group_values_format = [[x] * len(date_range) for x in group_values]
            df_padded = pd.DataFrame(group_values_format + [date_range]).transpose()
            df_padded_by_group = df_padded_by_group.append(df_padded)

        df_padded_by_group.columns = self.group_fields + [self.date_field]
        df_padded_by_group = pd.merge(
            df_padded_by_group,
            self.df,
            how="left",
            on=self.group_fields + [self.date_field],
        )
        logger.info(
            f"Total missing rows added: {df_padded_by_group.shape[0] - self.df.shape[0]}"
        )
        self.df = df_padded_by_group

    @staticmethod
    def _calc_missing_streak(df: pd.DataFrame) -> pd.DataFrame:
        """Helper method for filter_groups_max_missing_streak. 
           Calculates the longest consecutive streak of missing values (NAs)
           for a time series. 

        Args:
            df (pd.DataFrame): dataframe with possible missing values. 

        Returns:
            pd.DataFrame: [description]
        """
        df['streak2'] = (df['non_nan'] == 0).cumsum()
        df['cumsum'] = np.nan
        df.loc[df['non_nan'] == 1, 'cumsum'] = df['streak2']
        df['cumsum'] = df['cumsum'].fillna(method='ffill')
        df['cumsum'] = df['cumsum'].fillna(0)
        df['streak'] = df['streak2'] - df['cumsum']
        df.drop(['streak2', 'cumsum'], axis=1, inplace=True)
        return df


    def filter_groups_max_missing_streak(self, max_streak: int):
        """Filters groups groups based on minimum number of consecutive missing values

        Args:
            max_streak (int): Maximum number of consecutive missing values allowed
        """
         
        current_rows = self.df[self.group_fields].drop_duplicates().shape[0]
        self.df['non_nan'] = [1 if not x else 0 for x in self.df[self.yvar_field].isnull()]
        missing_streaks = self.df.groupby(self.group_fields).apply(self._calc_missing_streak)
        longest_streak_df = (missing_streaks
                             .groupby(self.group_fields)['streak'].max()
                             .reset_index()
                             .query(f"streak <= {str(max_streak)}")
                             .drop(columns='streak')
        )
        del self.df['non_nan']
        updated_rows = longest_streak_df.shape[0]
        logging.info(f'N groups excluded from analysis: {current_rows - updated_rows}')
        self.df = pd.merge(self.df, longest_streak_df, on=self.group_fields, how='inner')



    def fill_missing_values(self):
        """Fills missing values with the group average"""
        total_na = sum(self.df[self.yvar_field].isnull())
        logger.info(f"Total NAs found: {total_na}") 
        if total_na == 0:
            logger.info("No missing Values to interpolate")
            return None       
        self.df[self.date_field] = pd.to_datetime(self.df[self.date_field])
        self.df.index = self.df[self.date_field]
        df_yvar = (self.df
                   .groupby(self.group_fields)
                   .resample(self.ts_frequency)
                   .mean()
                   )
        self.df = self.df.drop(columns=self.date_field).reset_index()       
        self.df[self.yvar_field] = df_yvar[self.yvar_field].interpolate().reset_index()[self.yvar_field]


    def add_forecast_bounds(self, floor_multiplier: float, cap_multiplier: float):
        """Creates a minimum (floor) and maximum (cap) that forecasts cannot exceed when
           creating predictions with Prophet.

        Args:
            floor_multiplier (float): Multiplier for minimum value. For example, "0.5"
                will
            cap_multiplier (float): [description]
        """
        logger.info("Forecast floor added with {floor_multiplier} * min_value added")
        logger.info("Forecast cap added with {cap_multiplier} * max_value added ")
        min_max_yvar_df = (
            self.df.groupby(self.group_fields)
            .agg(
                floor_value=(self.yvar_field, "min"), cap_value=(self.yvar_field, "max")
            )
            .reset_index()
        )
        min_max_yvar_df["floor_value"] = (
            min_max_yvar_df["floor_value"] * floor_multiplier
        )
        min_max_yvar_df["cap_value"] = min_max_yvar_df["cap_value"] * cap_multiplier
        self.df = pd.merge(self.df, min_max_yvar_df, how="inner", on=self.group_fields)

    def log_transform_values(self, *fields: str):
        """Log transforms all fields provided. Creates a 
           field with '_log1p' as a suffix to the orignal
           field name
        """
        for field in fields:
            logger.info(f"Log transforming {field}")
            self.df[field + "_log1p"] = np.log1p(self.df[field])
            del self.df[field]

    def return_transformed_df(self):
        """Returns the fully transformed dataframe"""
        return self.df
