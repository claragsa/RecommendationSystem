import pandas as pd
from pathlib import Path

def load_data(data_path=Path('data/raw')):
    df_events = pd.read_csv(data_path / 'events.csv')
    df_items_props_part1 = pd.read_csv(data_path / 'item_properties_part1.csv')
    df_items_props_part2 = pd.read_csv(data_path / 'item_properties_part2.csv')
    df_category = pd.read_csv(data_path / 'category_tree.csv')

    return df_events, df_items_props_part1, df_items_props_part2, df_category
