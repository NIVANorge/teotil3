import pandas as pd


def get_teotil3_results(st_yr, end_yr, regine_list, agri_loss_model, nve_data_yr):
    df = pd.read_csv(
        f"/home/jovyan/shared/common/teotil3/evaluation/teo3_results_nve{nve_data_yr}_{st_yr}-{end_yr}_agri-{agri_loss_model}-loss.csv"
    )
    df = df.query(
        "(regine in @regine_list) and (year >= @st_yr) and (year <= @end_yr)"
    ).copy()
    df["År"] = df["year"]
    cols = [i for i in df.columns if i.split("_")[0] == "accum"]
    df = df[["regine", "År"] + cols]
    for col in df.columns:
        if col.endswith("_kg"):
            new_col = col.replace("_kg", "_tonnes")
            df[new_col] = df[col] / 1000
            del df[col]

    return df


def get_aggregation_dict_for_columns(par):
    agg_dict = {
        "Akvakultur": [f"accum_aquaculture_{par}_tonnes"],
        "Jordbruk": [f"accum_agriculture_{par}_tonnes"],
        "Avløp": [
            f"accum_large-wastewater_{par}_tonnes",
            f"accum_spredt_{par}_tonnes",
        ],
        "Industri": [f"accum_industry_{par}_tonnes"],
        "Bebygd": [f"accum_urban_{par}_tonnes"],
        "Bakgrunn": [
            f"accum_agriculture-background_{par}_tonnes",
            f"accum_upland_{par}_tonnes",
            f"accum_wood_{par}_tonnes",
            f"accum_lake_{par}_tonnes",
        ],
    }

    return agg_dict


def aggregate_parameters(df, par):
    agg_dict = get_aggregation_dict_for_columns(par)
    for group, cols in agg_dict.items():
        df[group] = df[cols].sum(axis=1)

    df = df[["regine", "År"] + list(agg_dict.keys())]

    return df