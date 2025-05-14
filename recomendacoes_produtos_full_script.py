##-----------Required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
from implicit.evaluation import train_test_split, precision_at_k, ndcg_at_k

##-----------Load e Extraction
df_events = pd.read_csv(r'events.csv')
df_item_props_part1 = pd.read_csv(r'item_properties_part1.csv')
df_item_props_part2 = pd.read_csv(r'item_properties_part2.csv')
df_category = pd.read_csv(r'category_tree.csv')


##-----------First look into the data
print('Primeiras informações (head e info):')
print('Events:')
print(df_events.head())
print(df_events.info())
print('\nItem properties:')
print(df_item_props_part1.head())
print(df_item_props_part1.info())
print(df_item_props_part2.head())
print(df_item_props_part2.info())
print('\nProducts category:')
print(df_category.head())
print(df_category.info())

print('\nValores únicos por coluna:')
print('Events:')
print(df_events[['timestamp', 'visitorid', 'itemid']].nunique())
print('\nItem Properties:')
print(df_item_props_part1[['timestamp', 'itemid']].nunique())
print(df_item_props_part2[['timestamp', 'itemid']].nunique())
print('\nProducts category:')
print(df_category.nunique())

print('\nValores nulos:')
print('Events:')
print(df_events.isnull().sum())
print('\nItem Properties:')
print(df_item_props_part1.isnull().sum())
print(df_item_props_part2.isnull().sum())
print('\nProducts category:')
print(df_category.isnull().sum())

print('\nValores duplicados:')
print('Events:')
print(df_events.duplicated().sum())
print('\nItem Properties:')
print(df_item_props_part1.duplicated().sum())
print(df_item_props_part2.duplicated().sum())
print('\nProducts category:')
print(df_category.duplicated().sum())

#Interações por usuário e produto
##Interações por usuário
user_interactions = df_events['visitorid'].value_counts()
print('\nEventos por cada usuário', '\n',user_interactions.head())

##Interações por item
item_interactions = df_events['itemid'].value_counts()
print('\nEventos com cada item:', '\n',item_interactions.head())

##Interações agrupadas por tipo de evento
#usuario
user_event_counts = df_events.groupby(['visitorid', 'event']).size().unstack(fill_value=0)
print('\nInterações de cada usuário por tipo de evento:')
print(user_event_counts.head())
#item
item_event_counts = df_events.groupby(['itemid', 'event']).size().unstack(fill_value=0)
print('\nInterações de cada item por tipo de evento:')
print(item_event_counts.head())

##Produtos sem eventos
products_event = set(df_events['itemid'].unique())

products_prop1 = set(df_item_props_part1['itemid'].unique()) 
products_prop2 = set(df_item_props_part2['itemid'].unique())
products_catalog = products_prop1.union(products_prop2)

products_only_props = products_catalog - products_event
print('\nProdutos com propriedas mas sem eventos:', len(products_only_props))

products_only_events = products_event - products_catalog
print('\nProdutos apenas com eventos:', len(products_only_events))

##-----------Data Cleaning
#Remover produtos com poucos registros e remover produtos que estao em eventos mas nao estão catalogados
df_events = df_events[~df_events['itemid'].isin(products_only_events)]

df_item_props_part1 = df_item_props_part1[~df_item_props_part1['itemid'].isin(products_only_props)]
df_item_props_part2 = df_item_props_part2[~df_item_props_part2['itemid'].isin(products_only_props)]

#Filtrar os usuários com base em um limite minimo de interações
user_interactions = df_events['visitorid'].value_counts()
interactions_limit = 5
active_visitors = user_interactions[user_interactions>=interactions_limit].index
df_events = df_events[df_events['visitorid'].isin(active_visitors)]

#Converter timestamp para datetime
df_events['timestamp'] = pd.to_datetime(df_events['timestamp'], unit = 'ms')
print('\n',df_events['timestamp'].head())

#Colunas auxiliares
##1. numero de item comprados
itens_purchased = df_events[df_events['event'] == 'transaction'].groupby('visitorid')['itemid'].count()
df_events['n_itens_purchased'] = None
mask1 = df_events['event'] == 'transaction'
df_events.loc[mask1, 'n_itens_purchased'] = df_events.loc[mask1, 'visitorid'].map(itens_purchased)

##2. data da ultima compra
last_purchase = df_events[df_events['event'] == 'transaction'].groupby('visitorid')['timestamp'].max()
df_events['last_purchase'] = pd.NaT
mask = df_events['event'] == 'transaction'
df_events.loc[mask, 'last_purchase'] = df_events.loc[mask, 'visitorid'].map(last_purchase)

##-----------Data analysis
#1.Gráficos
#### número de eventos por tipo
n_events = sns.countplot(df_events, x= 'event', hue='event' )
for container in n_events.containers:
    n_events.bar_label(container) 
plt.title('Quantidade de ocorrencia de cada evento')
plt.xlabel('Eventos')
plt.ylabel('Número de ocorrências')
plt.tight_layout()
plt.show()

#### heatmap de cada interação
user_event_matrix = pd.crosstab(df_events['visitorid'], df_events['event'])
first_20_users = df_events['visitorid'].value_counts().head(20).index
user_event_matrix = user_event_matrix.loc[first_20_users]
sns.heatmap(user_event_matrix, annot=True, cmap= 'Blues', fmt='d')
plt.title('Heatmap de interações por usuário e tipo de evento')
plt.ylabel('Visitor ID')
plt.xlabel('Tipo de Evento')
plt.tight_layout()
plt.show()

#### produtos com mais interações
item_interactions = df_events['itemid'].value_counts() #Refiltrar após o data cleaning
top_itens = item_interactions.head(20).index
top_item_interactions = df_events[df_events['itemid'].isin(top_itens)]
most_viewed = sns.countplot(top_item_interactions, x= 'itemid', hue='event')
for container in most_viewed.containers:
    most_viewed.bar_label(container)
plt.title('Top 20 itens com mais eventos')
plt.xlabel('Item ID')
plt.ylabel('Número de interações')
plt.tight_layout()
plt.show()

#### comparação de tempo e número de vendas
df_transactions = df_events[df_events['event'] == 'transaction'].copy()
df_transactions.loc[:, 'date'] = df_transactions['timestamp'].dt.date
daily_counts = df_transactions.groupby('date').size().reset_index(name= 'num_transactions')

sns.lineplot(data= daily_counts, x= 'date', y = 'num_transactions', marker = 'o')
plt.title('Número de vendas ao longo do tempo')
plt.xlabel('Tempo')
plt.ylabel('Vendas')
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()

#### taxa de conversão por produto
view_df = df_events[df_events['event'] == 'view']
transaction_df = df_events[df_events['event'] == 'transaction']

top_itens = transaction_df['itemid'].value_counts().head(70).index

views_df_top = view_df[view_df['itemid'].isin(top_itens)]
transactions_df_top = transaction_df[transaction_df['itemid'].isin(top_itens)]

view_counts = views_df_top.groupby('itemid').size()
transaction_counts = transactions_df_top.groupby('itemid').size()

conversion_rate = (transaction_counts/view_counts).dropna()*100

conversion_df = conversion_rate.reset_index()
conversion_df.columns= ['itemid', 'conversion_rate']

conversion_graph = sns.barplot(conversion_df, x= 'itemid', y='conversion_rate', palette='viridis', hue= 'itemid', legend= False)
plt.title('Taxa de conversão dos 70 itens mais vendidos')
plt.ylabel('Taxa de conversão %')
plt.xlabel('Item ID')
plt.xticks(rotation= 45)
formatter = FuncFormatter(lambda y, _: f'{y:.0f}%')
conversion_graph.yaxis.set_major_formatter(formatter)
plt.tight_layout()
plt.show()


##-----------Model - collaborative filtering
##### preparação de dados
def prepare_data(df_events):
    '''
    1. Filtra os eventos relevantes e remove os dados faltantes
    2. Converte os IDs para ints
    3. Atribui pesos diferentes para cada tipo de evento
    '''
    df_events=df_events[df_events['event'].isin(['view', 'addtocart', 'transaction'])]
    df_events = df_events.dropna(subset = ['visitorid', 'itemid'])

    visitorid_cat = df_events['visitorid'].astype('category')
    itemid_cat = df_events['itemid'].astype('category')

    visitorid_map = dict(enumerate(visitorid_cat.cat.categories))
    itemid_map = dict(enumerate(itemid_cat.cat.categories))

    visitorid_reverse_map = {v: k for k, v in visitorid_map.items()}
    itemid_reverse_map = {v: k for k, v in itemid_map.items()}

    df_events['visitorid'] = visitorid_cat.cat.codes
    df_events['itemid'] = itemid_cat.cat.codes

    event_weight = {
        'view': 1,
        'addtocart': 6,
        'transaction': 20
    }
    df_events['weight'] = df_events['event'].map(event_weight)
    return df_events, visitorid_reverse_map, itemid_map

df_events, visitorid_reverse_map, itemid_map = prepare_data(df_events)

##### matriz usuario-item
user_item_matrix = sparse.csr_matrix(
    (df_events['weight'].astype(float),
     (df_events['visitorid'], df_events['itemid']))
)


##### pesos para normalização dos dados
user_item_matrix = bm25_weight(user_item_matrix, K1=1.2, B=0.75)
user_item_matrix = user_item_matrix.tocsr()

##### modelo ALS
train_df, test_df = train_test_split(user_item_matrix, train_percentage= 0.8)

model = AlternatingLeastSquares(factors=512, regularization=0.3, iterations=10)
model.fit(train_df)

###### avaliação do modelo com Precision@K e NDCG@K
precision = precision_at_k(model, train_df, test_df, K=10)
print(f"Precision@10: {precision:.4f}")

ndcg = ndcg_at_k(model, train_df, test_df, K=10)
print(f"NDCG@10: {ndcg:.4f}")

##-----------Recomendação de produtos

def recommend_to_visitor(visitor_original_id, model, user_item_matrix, visitorid_reverse_map, itemid_map, N=10):

    if visitor_original_id not in visitorid_reverse_map:
        print(f"Visitor ID {visitor_original_id} não encontrado.")
        return None

    visitor_id = visitorid_reverse_map[visitor_original_id]

    try:
        recommendations = model.recommend(
            userid = visitor_id,
            user_items = user_item_matrix[visitor_id],
            N = N,
            filter_already_liked_items = True,
        )
        if isinstance(recommendations, tuple) and len(recommendations) == 2:
            item_ids, scores = recommendations
            resultado = [(itemid_map[item_id], score) for item_id, score in zip(item_ids, scores)]
        else:
            resultado = [(itemid_map[item_id], score) for item_id, score in recommendations]
    
    except Exception as e:
        print(f"Erro ao recomendar itens: {e}")
        return None
    return resultado

visitor_id = 54
recs = recommend_to_visitor(visitor_id, model, user_item_matrix, visitorid_reverse_map, itemid_map) #Visitor ID aleatório dentro do arquivo events.csv
if recs:
    print('\nRecomendações:')
    for item, score in recs:
        print(f"Item ID: {item}, Score: {score:.4f}")
else:
    print("Nenhuma recomendação encontrada.")
