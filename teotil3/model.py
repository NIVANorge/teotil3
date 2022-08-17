# import os
from collections import defaultdict

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


def run_model(data, id_col="regine", next_down_col="regine_down"):
    """Run the TEOTIL2 model with the specified inputs. 'data' must either be a
    dataframe or a file path to a CSV in the correct format e.g. the dataframe
    or CSV returned by make_input_file(). See below for format details.

    Quantities specified in 'data' are assigned to the regine catchment network
    and accumulated downstream, allowing for retention.

    Args
        data:          Str or dataframe e.g. as returned by make_input_file().
                       The following columns are mandatory:

                           [id_col, next_down_col, "a_cat_land_km2",
                            "runoff_mm/yr", "q_cat_m3/s"]

                       Additional columns to be accumulated must be named
                       '{source}_{par}_{unit}', all in lowercase e.g.
                       'ind_cd_tonnes' for industrial point inputs of cadmium in
                       tonnes. In addition, there must be a corresponding column
                       named 'trans_{par}' containing transmission factors
                       (floats between 0 and 1)
        id_col:        Str. Default 'regine'. Column in 'df' with unique IDs for
                       catchments
        next_down_col: Str. Default 'regine_down'. Column in 'df' with ID of
                       catchment immediately downstream

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
        id_col,
        next_down_col,
        "a_cat_land_km2",
        "runoff_mm/yr",
        "q_cat_m3/s",
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
            df[f"trans_{par}"].between(0, 1, inclusive="both").all()
        ), f"Column 'trans_{par}' contains values outside of range [0, 1]"

    # Run model
    g = build_graph(df, id_col=id_col, next_down_col=next_down_col)
    g = accumulate_loads(g, acc_cols)

    return g


def accumulate_loads(g, acc_cols):
    """Perform accumulation over a TEOTIL3 hydrological network. Usually called by run_model().
       Local inputs for the sources and parameters specified by 'acc_cols' are accumulated
       downstream, allowing for parameter-specific retention.

    Args
        g          Pre-built NetworkX graph. Must be a directed tree/forest and each node must
                   have properties 'local' (internal load) and 'accum' (empty dict).
        acc_cols:  List of str. Columns to accumulate (in addition to the standard/required ones).
                   Must be named '{source}_{par}_{unit}' - see docstring for run_model() for
                   further details
    Returns
        NetworkX graph object. g is modifed by adding the property 'accum_XXX' to each node.
        This is the total amount of substance flowing out of the node.
    """
    assert nx.is_tree(g), "g is not a valid tree."
    assert nx.is_directed_acyclic_graph(g), "g is not a valid DAG."

    # Process nodes in topo order from headwaters down
    for nd in list(nx.topological_sort(g))[:-1]:
        # Get catchments directly upstream
        preds = list(g.predecessors(nd))

        if len(preds) > 0:
            # Accumulate total input from upstream
            # Counters default to 0
            a_up = 0
            q_up = 0
            tot_dict = defaultdict(int)

            # Loop over upstream catchments
            for pred in preds:
                a_up += g.nodes[pred]["accum"]["upstr_area_km2"]
                q_up += g.nodes[pred]["accum"]["q_m3/s"]

                # Loop over quantities of interest
                for col in acc_cols:
                    tot_dict[col] += g.nodes[pred]["accum"][col]

            # Assign outputs
            # Area and flow
            g.nodes[nd]["accum"]["upstr_area_km2"] = (
                a_up + g.nodes[nd]["local"]["a_cat_land_km2"]
            )
            g.nodes[nd]["accum"]["q_m3/s"] = q_up + g.nodes[nd]["local"]["q_cat_m3/s"]

            # Calculate output. Oi = ti(Li + Ii)
            for col in acc_cols:
                par = col.split("_")[-2]
                g.nodes[nd]["accum"][col] = (
                    g.nodes[nd]["local"][col] + tot_dict[col]
                ) * g.nodes[nd]["local"]["trans_%s" % par]

        else:
            # Area and flow
            g.nodes[nd]["accum"]["upstr_area_km2"] = g.nodes[nd]["local"][
                "a_cat_land_km2"
            ]
            g.nodes[nd]["accum"]["q_m3/s"] = g.nodes[nd]["local"]["q_cat_m3/s"]

            # No upstream inputs. Oi = ti * Li
            for col in acc_cols:
                par = col.split("_")[-2]
                g.nodes[nd]["accum"][col] = (
                    g.nodes[nd]["local"][col] * g.nodes[nd]["local"]["trans_%s" % par]
                )

    return g


def model_to_dataframe(g, id_col="regine", next_down_col="regine_down", out_path=None):
    """Convert a TEOTIL3 graph to a Pandas dataframe. If a path is supplied, the dataframe
       will also be written to CSV format.

    Args
        g              NetworkX graph object returned by teo.model.run_model()
        id_col:        Str. Default 'regine'. Column in 'df' with unique IDs for
                       catchments
        next_down_col: Str. Default 'regine_down'. Column in 'df' with ID of
                       catchment immediately downstream
        plot_path:     Raw Str. Optional. CSV path to which df will be saved

    Returns
        Dataframe
    """
    # Container for data
    out_dict = defaultdict(list)

    # Loop over data
    for nd in list(nx.topological_sort(g))[:-1]:
        for stat in ["local", "accum"]:
            for key in g.nodes[nd][stat]:
                out_dict["%s_%s" % (stat, key)].append(g.nodes[nd][stat][key])

    # Convert to df
    df = pd.DataFrame(out_dict)

    # Reorder cols
    key_cols = [f"local_{id_col}", f"local_{next_down_col}"]
    cols = [i for i in df.columns if not i in key_cols]
    cols.sort()
    df = df[key_cols + cols]
    cols = list(df.columns)
    cols[:2] = [id_col, next_down_col]
    df.columns = cols

    # Write output
    if out_path:
        df.to_csv(out_path, index=False, encoding="utf-8")

    return df
