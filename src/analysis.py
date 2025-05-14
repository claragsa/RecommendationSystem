import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import FuncFormatter

def data_analysis(df_events, df_item_props_part1, df_item_props_part2):
    # 1. Contagem de eventos por tipo
    n_events = sns.countplot(data=df_events, x='event', hue='event')
    for container in n_events.containers:
        n_events.bar_label(container)
    plt.title('Quantidade de ocorrência de cada evento')
    plt.xlabel('Eventos')
    plt.ylabel('Número de ocorrências')
    plt.tight_layout()
    plt.show()

    # 2. Heatmap de interações por usuário
    user_event_matrix = pd.crosstab(df_events['visitorid'], df_events['event'])
    first_20_users = df_events['visitorid'].value_counts().head(20).index
    user_event_matrix = user_event_matrix.loc[first_20_users]
    user_event_heatmap = sns.heatmap(user_event_matrix, annot=True, cmap='Blues', fmt='d')
    plt.title('Heatmap de interações por usuário e tipo de evento')
    plt.ylabel('Visitor ID')
    plt.xlabel('Tipo de Evento')
    plt.tight_layout()
    plt.show()

    # 3. Top 20 itens com mais eventos
    item_interactions = df_events['itemid'].value_counts()
    top_itens = item_interactions.head(20).index
    top_item_interactions = df_events[df_events['itemid'].isin(top_itens)]
    most_viewed = sns.countplot(data=top_item_interactions, x='itemid', hue='event')
    for container in most_viewed.containers:
        most_viewed.bar_label(container)
    plt.title('Top 20 itens com mais eventos')
    plt.xlabel('Item ID')
    plt.ylabel('Número de interações')
    plt.tight_layout()
    plt.show()

    # 4. Vendas ao longo do tempo
    df_transactions = df_events[df_events['event'] == 'transaction'].copy()
    df_transactions.loc[:, 'date'] = df_transactions['timestamp'].dt.date
    daily_counts = df_transactions.groupby('date').size().reset_index(name='num_transactions')

    sns.lineplot(data=daily_counts, x='date', y='num_transactions', marker='o')
    plt.title('Número de vendas ao longo do tempo')
    plt.xlabel('Tempo')
    plt.ylabel('Vendas')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 5. Taxa de conversão por produto
    view_df = df_events[df_events['event'] == 'view']
    transaction_df = df_events[df_events['event'] == 'transaction']

    top_itens = transaction_df['itemid'].value_counts().head(70).index

    views_df_top = view_df[view_df['itemid'].isin(top_itens)]
    transactions_df_top = transaction_df[transaction_df['itemid'].isin(top_itens)]

    view_counts = views_df_top.groupby('itemid').size()
    transaction_counts = transactions_df_top.groupby('itemid').size()

    conversion_rate = (transaction_counts / view_counts).dropna() * 100

    conversion_df = conversion_rate.reset_index()
    conversion_df.columns = ['itemid', 'conversion_rate']

    conversion_graph = sns.barplot(data=conversion_df, x='itemid', y='conversion_rate', palette='viridis', hue='itemid', legend=False)
    plt.title('Taxa de conversão dos 70 itens mais vendidos')
    plt.ylabel('Taxa de conversão (%)')
    plt.xlabel('Item ID')
    plt.xticks(rotation=45)
    formatter = FuncFormatter(lambda y, _: f'{y:.0f}%')
    conversion_graph.yaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.show()

    return n_events, user_event_heatmap, most_viewed, daily_counts, conversion_graph