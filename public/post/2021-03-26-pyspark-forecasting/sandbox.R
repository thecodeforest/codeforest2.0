library(tidyverse)
library(timetk)
library(here)
base_dir = '/Users/MLeBo1/Desktop/codeforest2.0/content/post/2021-03-26-pyspark-forecasting'
df = read_csv(file.path(base_dir, 'data', 'output', 'sales_data_raw.csv'))
df %>% 
  distinct(store, dept)
df %>% 
  filter(store == 1, dept %in% c(1, 2, 3, 4)) %>% 
  mutate(store_id = paste(store, dept, sep='_')) %>% 
  select(-part) %>% 
  select(date, store_id, contains('weekly')) %>% 
  pivot_longer(contains('weekly')) %>% 
  plot_time_series(.date_var = date, 
                   .value = value, 
                   .color_var = name, 
                   .facet_vars = store_id,
                   .smooth = FALSE)

df_input = read_csv(file.path(base_dir, 'input_df.csv'))
df_input %>%
  mutate(y = expm1(y)) %>% 
  filter(store == 1, dept == 1) %>%
  plot_time_series(.date_var = ds, .value = y)
