from src.load import load_data
from src.cleaning import clean_data
from src.model import prepare_data, get_matrix, train_model, evaluate_model, recommend_to_visitor

def main():
    df_events, df_items_props_part1, df_items_props_part2, df_category = load_data()
    df_events_clean, df_item_props1_clean, df_item_props2_clean = clean_data(df_events, df_items_props_part1, df_items_props_part2)
    df_prepared, visitor_map, item_map = prepare_data(df_events_clean)
    user_item_matrix = get_matrix(df_prepared)
    model, train_df, test_df = train_model(user_item_matrix)
    precision, ndcg = evaluate_model(model, train_df, test_df)
    recs = recommend_to_visitor(54, model, user_item_matrix, visitor_map, item_map, N=10)
    if recs:
        print('\nRecomendações para o usuário:')
        for item, score in recs:
            print(f"Item ID: {item}, Score: {score:.4f}")
    else:
        print("Nenhuma recomendação disponível.")
if __name__ == "__main__":
    main()



