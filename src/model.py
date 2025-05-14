import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import train_test_split, precision_at_k, ndcg_at_k
from implicit.nearest_neighbours import bm25_weight

##-----------Model - collaborative filtering
##### preparação de dados
def prepare_data(df_events):
    df_events = df_events.copy()
    df_events = df_events[df_events['event'].isin(['view', 'addtocart', 'transaction'])]
    df_events = df_events.dropna(subset=['visitorid', 'itemid'])

    visitorid_cat = df_events['visitorid'].astype('category')
    itemid_cat = df_events['itemid'].astype('category')

    visitorid_map = dict(enumerate(visitorid_cat.cat.categories))
    itemid_map = dict(enumerate(itemid_cat.cat.categories))

    visitorid_to_index = {v: k for k, v in visitorid_map.items()}
    item_index_to_id = itemid_map

    df_events['visitorid'] = visitorid_cat.cat.codes
    df_events['itemid'] = itemid_cat.cat.codes

    event_weight = {
        'view': 1,
        'addtocart': 6,
        'transaction': 20
    }
    df_events['weight'] = df_events['event'].map(event_weight)
    return df_events, visitorid_to_index, item_index_to_id

#### matriz de interação
def get_matrix(df_events):
    user_item_matrix = sparse.csr_matrix(
        (df_events['weight'].astype(float),
         (df_events['visitorid'], df_events['itemid']))
    )

    user_item_matrix = bm25_weight(user_item_matrix, K1=1.2, B=0.75)
    user_item_matrix = user_item_matrix.tocsr()

    return user_item_matrix

#### treinamento do modelo
def train_model(user_item_matrix):
    train_df, test_df = train_test_split(user_item_matrix, train_percentage= 0.8)

    model = AlternatingLeastSquares(factors=512, regularization=0.3, iterations=10)
    model.fit(train_df)

    return model, train_df, test_df

#### avaliação do modelo
def evaluate_model(model, train_df, test_df):
    precision = precision_at_k(model, train_df, test_df, K=10)
    print(f"Precision@10: {precision:.4f}")

    ndcg = ndcg_at_k(model, train_df, test_df, K=10)
    print(f"NDCG@10: {ndcg:.4f}")

    return precision, ndcg

#### recomendação para o visitante
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