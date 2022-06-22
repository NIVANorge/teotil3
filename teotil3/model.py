# import os
# from collections import defaultdict

# import geopandas as gpd
# import graphviz
# import matplotlib.pyplot as plt
import networkx as nx

# import numpy as np
import pandas as pd


def build_graph(df, id_col="regine", next_down_col="regine_down"):
    """Build a Directed Acyclic Graph (DAG) from an adjacency matrix. Data 
    from columns other than 'id_col' and 'next_down_col' will be added to 
    each node as 'local' attributes.

    Args
        df:            Dataframe. Adjacency matrix
        id_col:        Str. Default 'regine'. Column in 'df' with unique
                       IDs for catchments
        next_down_col: Str. Default 'regine_down'. Column in 'df' with
                       ID of catchment immediately downstream

    Returns
        Object. networkX graph
    """
    g = nx.DiGraph()

    # Add nodes
    for idx, row in df.iterrows():
        nd = row[id_col]
        g.add_node(nd, local=row.to_dict(), accum={})

    # Add edges
    for idx, row in df.iterrows():
        fr_nd = row[id_col]
        to_nd = row[next_down_col]
        g.add_edge(fr_nd, to_nd)

    return g


def run_model(data):
    """Run the TEOTIL2 model with the specified inputs. 'data' must either be a
    dataframe or a file path to a CSV in the correct format e.g. the dataframe
    or CSV returned by make_input_file(). See below for format details.

    Quantities specified in 'data' are assigned to the regine catchment network
    and accumulated downstream, allowing for retention.

    Args
        data: Str or dataframe e.g. as returned by make_input_file(). The following
              columns are mandatory:

                  ["regine", "regine_down", "a_reg_land_km2", "runoff_mm/yr",
                   "q_reg_m3/s", "vol_lake_m3"]

              Additional columns to be accumulated must be named '{source}_{par}_{unit}',
              all in lowercase e.g. 'ind_cd_tonnes' for industrial point inputs of
              cadmium in tonnes. In addition, there must be a corresponding column
              named 'trans_{par}' containing transmission factors
              (floats between 0 and 1)

    Returns
        NetworkX graph object with results added as node attributes.
    """
    # Parse input
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, str):
        df = pd.read_csv(data)
    else:
        raise ValueError('"data" must be either a "raw" string or a Pandas dataframe.')

    # Check required cols are present
    req_cols = [
        "regine",
        "regine_ned",
        "a_reg_km2",
        "runoff_mm/yr",
        "q_reg_m3/s",
        "vol_lake_m3",
    ]
    for col in req_cols:
        assert col in df.columns, f"'data' must contain a column named '{col}'."

    # Identify cols to accumulate
    acc_cols = [
        i for i in df.columns if (i not in req_cols) and (i.split("_")[0] != "trans")
    ]

    # Check 'trans' cols are present
    par_list = list(set([i.split("_")[-2] for i in acc_cols]))
    for par in par_list:
        assert (
            f"trans_{par}" in df.columns
        ), f"Column 'trans_{par}' not present in input.'"
        assert (
            df[f"trans_{par}"].between(0, 1, inclusive=True).all()
        ), f"Column 'trans_{par}' contains values outside of range [0, 1]"

    # Run model
    g = build_graph(df, id_col="regine", next_down_col="regine_down")
    g = accumulate_loads(g, acc_cols)

    return g