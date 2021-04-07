import pandas as pd
import numpy as np
from typing import List
import logging

import logging

logger = logging.getLogger(__name__)

class FcastDataFrame:
    """[summary]
    """
    def __init__(self, df: pd.DataFrame, group_fields: List[str], date_field: str, yvar_field: str, ts_frequency: str):
        """[summary]

        Args:
            df (pd.DataFrame): [description]
            group_fields (List[str]): [description]
            date_field (str): [description]
            yvar_field (str): [description]
            ts_frequency (str): [description]
        """
        self.df = df
        self.group_fields = group_fields
        self.date_field = date_field
        self.yvar_field = yvar_field
        self.ts_frequency = ts_frequency
    

    def replace_negative_value_with_zero(self):
        logger.info('Replacing negative values with nan')
        self.df[self.yvar_field] = [0 if x < 0 else x for x in self.df[self.yvar_field]]


    def pad_missing_values(self):
        """[summary]
        """

        logger.info('Padding missing values')
        min_max_dt_df = (self.df
                        .groupby(self.group_fields)
                        .agg(dt_min=(self.date_field, 'min'), 
                            dt_max=(self.date_field, 'max'))
                        .reset_index()
                        )
        df_padded_by_group = pd.DataFrame(None)                 
        for index, row in enumerate(min_max_dt_df.itertuples()):
            date_range = [x.strftime('%Y-%m-%d') 
                        for x in pd.date_range(row.dt_min, row.dt_max, freq=self.ts_frequency)]
            group_values = min_max_dt_df.iloc[index][self.group_fields]
            group_values_format = [[x] * len(date_range) for x in group_values]    
            df_padded = pd.DataFrame(group_values_format + [date_range]).transpose()
            df_padded_by_group = df_padded_by_group.append(df_padded)  
        df_padded_by_group.columns = self.group_fields + [self.date_field]
        df_padded_by_group = pd.merge(df_padded_by_group, self.df, how='left', on = self.group_fields + [self.date_field]) 
        logger.info(f'Total missing rows added: {df_padded_by_group.shape[0] - self.df.shape[0]}')
        self.df = df_padded_by_group


    def fill_missing_values(self, method = 'interpolate'):
        total_na = sum(self.df[self.yvar_field].isnull())
        logger.info(f'Total NAs found: {total_na}')
        if  total_na == 0:
            logger.info('No missing Values to interpolate')
            return None
        logger.info(f'Interpolating via method: {method}')
        if method == 'interpolate':
            yvar_interpolated = (self.df
                                .groupby(self.group_fields)[self.yvar_field]
                                .apply(lambda x : x.interpolate(method="spline", 
                                                                order = 1, 
                                                                limit_direction = "both"))                   
                                )
        else:
            yvar_interpolated = self.df[self.yvar_field].fillna(0)                   
        self.df[self.yvar_field + 'interp'] = yvar_interpolated
        self.df[self.yvar_field + '_prep'] = self.df[self.yvar_field].combine_first(self.df[self.yvar_field + 'interp'])
        self.df[self.yvar_field] = self.df[self.yvar_field + '_prep']

    def add_forecast_bounds(self, floor_multiplier: float, cap_multiplier:float):
        logger.info('Forecast floor added with {floor_multiplier} * min_value added')
        logger.info('Forecast cap added with {cap_multiplier} * max_value added ')
        min_max_yvar_df = (self.df
                    .groupby(self.group_fields)
                    .agg(floor_value = (self.yvar_field, 'min'),
                         cap_value = (self.yvar_field, 'max'))
                    .reset_index()
                    )
        min_max_yvar_df['floor_value'] = min_max_yvar_df['floor_value'] * floor_multiplier
        min_max_yvar_df['cap_value'] = min_max_yvar_df['cap_value'] * cap_multiplier
        self.df = pd.merge(self.df, min_max_yvar_df, how='inner', on=self.group_fields)

    def log_transform_values(self, *fields):    
        for field in fields:
            logger.info(f'Log transforming {field}')
            self.df[field + '_log1p'] = np.log1p(self.df[field])
            del self.df[field]


    def return_transformed_df(self):
        return self.df
        