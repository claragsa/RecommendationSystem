##-----------Data Cleaning
import pandas as pd

def clean_data(df_events, df_item_props_part1, df_item_props_part2):
    # Identificar produtos únicos
    products_event = set(df_events['itemid'].unique())
    products_prop1 = set(df_item_props_part1['itemid'].unique()) 
    products_prop2 = set(df_item_props_part2['itemid'].unique())
    products_catalog = products_prop1.union(products_prop2)
    
    # Produtos que aparecem apenas em eventos ou apenas em propriedades
    products_only_props = products_catalog - products_event
    products_only_events = products_event - products_catalog
    
    # Remover produtos que não estão em ambos os conjuntos
    df_events = df_events[~df_events['itemid'].isin(products_only_events)]
    df_item_props_part1 = df_item_props_part1[~df_item_props_part1['itemid'].isin(products_only_props)]
    df_item_props_part2 = df_item_props_part2[~df_item_props_part2['itemid'].isin(products_only_props)]

    # Manter apenas usuários ativos com pelo menos N interações
    user_interactions = df_events['visitorid'].value_counts()
    interactions_limit = 5
    active_visitors = user_interactions[user_interactions >= interactions_limit].index
    df_events = df_events[df_events['visitorid'].isin(active_visitors)]

    # Converter timestamp para datetime
    df_events['timestamp'] = pd.to_datetime(df_events['timestamp'], unit='ms')

    # Coluna 1: número de itens comprados por usuário
    itens_purchased = df_events[df_events['event'] == 'transaction'].groupby('visitorid')['itemid'].count()
    df_events['n_itens_purchased'] = None
    mask = df_events['event'] == 'transaction'
    df_events.loc[mask, 'n_itens_purchased'] = df_events.loc[mask, 'visitorid'].map(itens_purchased)

    # Coluna 2: data da última compra do usuário
    last_purchase = df_events[df_events['event'] == 'transaction'].groupby('visitorid')['timestamp'].max()
    df_events['last_purchase'] = pd.NaT
    df_events.loc[mask, 'last_purchase'] = df_events.loc[mask, 'visitorid'].map(last_purchase)

    return df_events, df_item_props_part1, df_item_props_part2