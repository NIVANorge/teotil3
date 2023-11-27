from collections import defaultdict

import networkx as nx
import pandas as pd


def build_graph(df, id_col="regine", next_down_col="regine_down"):
    """
    Build a Directed Acyclic Graph (DAG) from an adjacency matrix. Data from columns other than
    'id_col' and 'next_down_col' will be added to each node as 'local' attributes.

    Args
        df: Dataframe. Adjacency matrix
        id_col: Str. Default 'regine'. Column in 'df' with unique IDs for catchments 
        next_down_col: Str. Default 'regine_down'. Column in 'df' with ID of catchment immediately
            downstream

    Returns
        Object. networkX graph
    """
    g = nx.DiGraph()

    # Add nodes
    for idx, row in df.iterrows():
        g.add_node(row[id_col], local=row.to_dict(), accum={})

    # Add edges
    for idx, row in df.iterrows():
        g.add_edge(row[id_col], row[next_down_col])

    return g


def validate_input(data, req_cols, acc_cols):
    """
    Validate the input data for the TEOTIL3 model.

    Args
        data: DataFrame or str. The input data for the model.
        req_cols: List[str]. The required columns in the input data.
        acc_cols: List[str]. The columns to be accumulated in the model.

    Returns
        DataFrame. The validated input data.

    Raises
        ValueError: If the input data is not a DataFrame or a raw string.
        ValueError: If a required column is not in the input data.
        ValueError: If a 'trans' column is not present in the input data.
        ValueError: If a 'trans' column contains values outside of range [0, 1].
    """
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, str):
        df = pd.read_csv(data)
    else:
        raise ValueError('"data" must be either a "raw" string or a Pandas dataframe.')

    for col in req_cols:
        if col not in df.columns:
            raise ValueError(f"'data' must contain a column named '{col}'.")

    par_list = list(set([i.split("_")[-2] for i in acc_cols]))
    for par in par_list:
        if f"trans_{par}" not in df.columns:
            raise ValueError(f"Column 'trans_{par}' not present in input.'")
        if not df[f"trans_{par}"].between(0, 1, inclusive="both").all():
            raise ValueError(
                f"Column 'trans_{par}' contains values outside of range [0, 1]"
            )

    return df


def run_model(
    data, id_col="regine", next_down_col="regine_down", totals_from_subfracs=True
):
    """
    Run the TEOTIL2 model with the specified inputs.

    Args
        data: DataFrame or str. The input data for the model.
        id_col: Str. The column in 'df' with unique IDs for catchments.
        next_down_col: Str. The column in 'df' with ID of catchment immediately downstream.
        totals_from_subfracs: Bool. Whether to sum subfractions of N and P for each source to give
            results for TOTN and TOTP.

    Returns
        DiGraph. The networkX graph object with results added as node attributes.
    """
    req_cols = [id_col, next_down_col, "a_cat_land_km2", "runoff_mm/yr", "q_cat_m3/s"]
    acc_cols = [
        i for i in data.columns if (i not in req_cols) and (i.split("_")[0] != "trans")
    ]

    df = validate_input(data, req_cols, acc_cols)

    g = build_graph(df, id_col=id_col, next_down_col=next_down_col)
    g = accumulate_loads(g, acc_cols)
    if totals_from_subfracs:
        g = sum_subfractions(g, acc_cols)

    return g


def accumulate_loads(g, acc_cols):
    """
    Perform accumulation over a TEOTIL3 hydrological network. Usually called by run_model(). Local
    inputs for the sources and parameters specified by 'acc_cols' are accumulated downstream, 
    allowing for parameter-specific retention.

    Args
        g: Pre-built NetworkX graph. Must be a directed tree/forest and each node must have 
            properties 'local' (internal load) and 'accum' (empty dict).
        acc_cols: List of str. Columns to accumulate (in addition to the standard/required ones).
            Must be named '{source}_{par}_{unit}' - see docstring for run_model() for further 
            details

    Returns
        NetworkX graph object. g is modifed by adding the property 'accum_XXX' to each node. This 
        is the total amount of substance flowing out of the node.

    Raises
        ValueError: If the graph is not a valid tree or a valid DAG.
    """
    if not nx.is_tree(g):
        raise ValueError("The graph is not a valid tree.")
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("The graph is not a valid DAG.")

    # Process nodes in topo order from headwaters down
    for nd in list(nx.topological_sort(g))[:-1]:
        # Get catchments directly upstream
        preds = list(g.predecessors(nd))

        # Initialize counters
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

    return g


def sum_subfractions(g, acc_cols):
    """
    Checks whether nodes in 'g' contain results for ('din_kg' or 'ton_kg'), or ('tdp_kg' or 
    'tpp_kg'). If they do, new 'local' and 'accum' attributes are added by summing the subfractions
    for each parameter from each source.

    Args
        g: Networkx graph object of model results
        acc_cols: List of str. Parameter named ('{source}_{par}_{unit}') accumulated in 'g'

    Returns
        'g' is updated with new node attributes.
    """
    # Get sources for which N & P totals can be calculated
    subfrac_list = ["din", "ton", "tdp", "tpp"]
    sources = set(
        [
            "_".join(col.split("_")[:-2])
            for col in acc_cols
            if col.split("_")[-2] in subfrac_list
        ]
    )

    # Loop over graph
    for nd in g.nodes():
        for source in sources:
            if "local" in g.nodes[nd]:
                for substance, fractions in [
                    ("n", ["din", "ton"]),
                    ("p", ["tdp", "tpp"]),
                ]:
                    cols = [
                        f"{source}_{frac}_kg"
                        for frac in fractions
                        if f"{source}_{frac}_kg" in acc_cols
                    ]
                    g.nodes[nd]["local"][f"{source}_tot{substance}_kg"] = sum(
                        g.nodes[nd]["local"][col] for col in cols
                    )
                    g.nodes[nd]["accum"][f"{source}_tot{substance}_kg"] = sum(
                        g.nodes[nd]["accum"][col] for col in cols
                    )

    return g


def model_to_dataframe(g, id_col="regine", next_down_col="regine_down", out_path=None):
    """
    Convert a TEOTIL3 graph to a Pandas dataframe. If a path is supplied, the dataframe will also
    be written to CSV format.

    Args
        g: NetworkX graph object returned by teo.model.run_model()
        id_col: Str. Default 'regine'. Column in 'df' with unique IDs for catchments
        next_down_col: Str. Default 'regine_down'. Column in 'df' with ID of catchment immediately 
            downstream
        out_path: Raw Str. Optional. CSV path to which df will be saved

    Returns
        Dataframe
    """
    # Container for data
    out_dict = defaultdict(list)

    # Loop over data
    for nd in list(nx.topological_sort(g))[:-1]:
        for stat in ["local", "accum"]:
            for key, value in g.nodes[nd][stat].items():
                out_dict[f"{stat}_{key}"].append(value)

    # Convert to df
    df = pd.DataFrame(out_dict)

    # Reorder cols
    key_cols = [f"local_{id_col}", f"local_{next_down_col}"]
    other_cols = sorted(col for col in df.columns if col not in key_cols)
    df = df[key_cols + other_cols]

    # Rename key columns
    df.rename(columns={key_cols[0]: id_col, key_cols[1]: next_down_col}, inplace=True)

    # Write output
    if out_path:
        df.to_csv(out_path, index=False, encoding="utf-8")

    return df
