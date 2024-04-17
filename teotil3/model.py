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


def validate_input(data, id_col, next_down_col):
    """
    Validate the input data for the TEOTIL3 model.

    Args
        data: DataFrame or str. The input data for the model.
        id_col: Str. The column in 'df' with unique IDs for catchments.
        next_down_col: Str. The column in 'df' with ID of catchment immediately downstream.

    Returns
        Tuple of input data (input_dataframe, cols_to_accumulate).

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

    req_cols = [id_col, next_down_col, "a_cat_land_km2", "runoff_mm/yr", "q_cat_m3/s"]
    acc_cols = [
        i for i in df.columns if (i not in req_cols) and (i.split("_")[0] != "trans")
    ]

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

    return (df, acc_cols)


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
    df, acc_cols = validate_input(data, id_col, next_down_col)
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


def _remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def get_source_contributions(reg_id, res_df, par, stat, type="pct"):
    """Calculate the contributioon from each source for the selected regine. Note
    that if 'res_df' contains data for multiple years, the proportion over all
    years will be calculated.

    Args
        reg_id: Str. The regine ID of interest.
        res_df: DataFrame of model results for several years.
        par: Str. Parameter of interest. Must be one of
            ('TOTN', 'DIN', 'TON', 'TOTP', 'TDP', 'TPP', 'TOC', 'SS').
        stat: Str. Either 'accum' or 'local'.
        type: Str. Either 'pct' for percentage contributions or 'abs' for
            absolute contribution. Default 'pct'.

    Returns
        A dataframe of source contributions.

    Raises
        ValueError if 'par' not in ('TOTN', 'DIN', 'TON', 'TOTP', 'TDP', 'TPP', 'TOC', 'SS').
        ValueError if 'stat' not in ('accum', 'stat').
        ValueError if 'type' not in ('pct', 'abs').
        ValueError if units in 'res_df' are inconsistent.
    """
    if par not in ("TOTN", "DIN", "TON", "TOTP", "TDP", "TPP", "TOC", "SS"):
        raise ValueError(
            f"'par' must be one of ('TOTN', 'DIN', 'TON', 'TOTP', 'TDP', 'TPP', 'TOC', 'SS'), not '{par}'."
        )
    if stat not in ("accum", "local"):
        raise ValueError("'stat' must be either 'accum' or 'local'.")
    if type not in ("pct", "abs"):
        raise ValueError("'type' must be either 'pct' or 'abs'.")

    # Get data of interest
    df = res_df.query("regine == @reg_id")
    cols = [col for col in df.columns if col.startswith(stat)]
    df = df[cols]
    cols = [_remove_prefix(col, f"{stat}_") for col in cols]
    df.columns = cols

    par_cols = [
        col
        for col in df.columns
        if (par.lower() in (i.lower() for i in col.split("_")))
        and not col.startswith("trans_")
    ]
    units = [col.split("_")[-1] for col in par_cols]
    if len(set(units)) != 1:
        raise ValueError(f"Results for {par} use inconsistent units ({set(units)}).")
    unit = units[0]
    df = df[par_cols]

    # Get totals for each source
    df = df.sum().to_frame()
    if type == "abs":
        col = "abs_contribution"
    else:
        df = 100 * df / df.sum()
        col = "pct_contribution"
    df.columns = [col]
    df.sort_values(col, ascending=False, inplace=True)
    df.index = df.index.str.split("_").str[0]

    return df


def get_outflow_concs_fluxes(
    reg_id,
    res_df,
    index_col="year",
    pars=["TOTN", "DIN", "TON", "TOTP", "TDP", "TPP", "TOC", "SS"],
    include="both",
):
    """Calculate outflow concentrations for the specified regine.

    Args
        reg_id: Str. The regine ID of interest.
        res_df: DataFrame of model results for several years.
        index_col: Str. The name of the index column in the DataFrame. Default 'year'.
        pars: List. Parameters to consider. Default
            ["TOTN", "DIN", "TON", "TOTP", "TDP", "TPP", "TOC", "SS"].
        include: Str. One of ('concs', 'fluxes', 'both'). Default both. Data to include
            in returned dataframe

    Returns
        A dataframe of outflow concentrations.

    Raises
        KeyError if 'index_col' not in 'res_df'.
        ValueError if units in 'res_df' are inconsistent.
        ValueError if input units are not 'kg' or 'tonnes'.
    """
    if index_col not in res_df.columns:
        raise KeyError(f"Index column '{index_col}' not found in 'res_df'.")
    if include not in ("concs", "fluxes", "both"):
        raise ValueError(
            f"'include' must be one of ('concs', 'fluxes', 'both'), not '{include}'."
        )

    # Get accumulated fluxes for regine of interest
    df = res_df.query("regine == @reg_id").set_index(index_col)
    cols = [col for col in df.columns if col.startswith("accum_")]
    df = df[cols].copy()
    cols = [_remove_prefix(col, "accum_") for col in cols]
    df.columns = cols

    # Annual flow volume
    df["vol_m3"] = df["q_m3/s"] * 365.25 * 24 * 60 * 60

    keep_cols = []
    for par in pars:
        par_cols = [col for col in df.columns if f"_{par.lower()}_" in col]
        units = [col.split("_")[-1] for col in par_cols]
        if len(set(units)) != 1:
            raise ValueError(
                f"Results for {par} use inconsistent units ({set(units)})."
            )
        unit = units[0]
        if unit not in ("kg", "tonnes"):
            raise ValueError(
                f"Input units must be either 'kg' or 'tonnes', not '{unit}'."
            )

        # Factors to convert SS and TOC to mg/l and other pars to ug/l
        unit_factor_dict = {
            ("C", "kg"): (1e3, "mg/l"),
            ("C", "tonnes"): (1e6, "mg/l"),
            ("N", "kg"): (1e6, "ug/l"),
            ("N", "tonnes"): (1e9, "ug/l"),
            ("P", "kg"): (1e6, "ug/l"),
            ("P", "tonnes"): (1e9, "ug/l"),
            ("S", "kg"): (1e3, "mg/l"),
            ("S", "tonnes"): (1e6, "mg/l"),
        }
        fac, conc_unit = unit_factor_dict[(par[-1], unit)]
        df[f"{par}_{unit}"] = df[par_cols].sum(axis="columns")
        df[f"{par}_{conc_unit}"] = fac * df[f"{par}_{unit}"] / df["vol_m3"]

        if include == "concs":
            keep_cols.append(f"{par}_{conc_unit}")
        elif include == "fluxes":
            keep_cols.append(f"{par}_{unit}")
        else:
            keep_cols.append(f"{par}_{conc_unit}")
            keep_cols.append(f"{par}_{unit}")

    df = df[sorted(keep_cols)]

    return df


def get_avlastningsbehov(
    reg_id,
    par,
    ges_conc,
    res_df,
    st_yr=2013,
    end_yr=2022,
    index_col="year",
):
    """
    This function calculates the load reduction target for a given regine and parameter.

    Args
        reg_id: Str. Regine of interest
        par: Str. Parameter to consider. Must be one of "TOTP", "TOTN", "TOC" (details subfractions
            are included if TOTN or TOTP area chosen).
        ges_conc: Float. The concentration target for Good Ecological Status (GES) in ug/l.
        res_df: DataFrame of model results for multiple years.
        st_yr: Int. Start year for period of interest. Default 2013.
        end_yr Int. End year for period of interest. Defaults 2022.
        index_col: Str. The name of the index column in the DataFrame. Default 'year'.

    Returns
        Dict of summary results.

    Raises
        ValueError is 'par' not in ("TOTP", "TOTN", "TOC").
        KeyError if 'index_col' not in 'res_df'.
        ValueError if units are not in ('tonnes', 'kg').

    """
    if par not in ("TOTP", "TOTN", "TOC"):
        raise ValueError("'par' must be one of 'TOTN', 'TOTP', 'TOC', not '{par}'.")
    if index_col not in res_df.columns:
        raise KeyError(f"Index column '{index_col}' not found in 'res_df'.")

    # Find lakes in regine of interest
    url = r"https://raw.githubusercontent.com/NIVANorge/teotil3/main/data/lake_residence_times_10m_dem.csv"
    lake_df = pd.read_csv(url).query("regine == @reg_id")
    lake_id_list = lake_df["vatnLnr"].tolist()

    # Get model results for regine
    df = res_df.query(
        f"(regine == @reg_id) and ({index_col} >= @st_yr) and ({index_col} <= @end_yr)"
    )

    # Estimate current outflow concs and fluxes
    frac_dict = {
        "TOTP": ["TOTP", "TDP", "TPP"],
        "TOTN": ["TOTN", "DIN", "TON"],
        "TOC": ["TOC"],
    }
    fracs = frac_dict[par]
    conc_df = get_outflow_concs_fluxes(
        reg_id,
        df,
        index_col=index_col,
        pars=fracs,
        include="concs",
    )
    flux_df = get_outflow_concs_fluxes(
        reg_id,
        df,
        index_col=index_col,
        pars=fracs,
        include="fluxes",
    )

    # Get source contributions
    src_df = get_source_contributions(reg_id, df, par, "accum", type="pct")

    # Get transmission
    if par == "TOC":
        trans = df["local_trans_toc"].iloc[0]
    elif par == "TOTP":
        trans_tdp = df["local_trans_tdp"].iloc[0]
        trans_tpp = df["local_trans_tpp"].iloc[0]
        tdp_col = [col for col in flux_df.columns if col.startswith("TDP")]
        in_tdp = flux_df[tdp_col].sum().values / trans_tdp
        tpp_col = [col for col in flux_df.columns if col.startswith("TPP")]
        in_tpp = flux_df[tpp_col].sum().values / trans_tpp
        totp_col = [col for col in flux_df.columns if col.startswith("TOTP")]
        trans = flux_df[totp_col].sum().values / (in_tdp + in_tpp)
        trans = trans[0]
    else:
        trans_din = df["local_trans_din"].iloc[0]
        trans_ton = df["local_trans_ton"].iloc[0]
        din_col = [col for col in flux_df.columns if col.startswith("DIN")]
        in_din = flux_df[din_col].sum().values / trans_din
        ton_col = [col for col in flux_df.columns if col.startswith("TON")]
        in_ton = flux_df[ton_col].sum().values / trans_ton
        totn_col = [col for col in flux_df.columns if col.startswith("TOTN")]
        trans = flux_df[totn_col].sum().values / (in_din + in_ton)
        trans = trans[0]

    # Estimate acceptable outflow flux for GES
    avg_ann_vol = (df["accum_q_m3/s"] * 365.25 * 24 * 60 * 60).mean()
    max_out_flux_ges = ges_conc * 1000 * avg_ann_vol / 1e9  # kg

    # Estimate acceptable inputs for GES
    max_in_flux_ges = max_out_flux_ges / trans  # kg

    # Estimate current inputs
    par_col = [col for col in flux_df.columns if col.startswith(par)][0]
    if par_col.split("_")[-1] == "tonnes":
        est_in_flux = 1000 * flux_df[par_col].mean() / trans  # kg
    elif par_col.split("_")[-1] == "kg":
        est_in_flux = flux_df[par_col].mean() / trans
    else:
        raise ValueError("Could not parse units.")

    # Estimate load reduction target
    red_target = est_in_flux - max_in_flux_ges
    red_target_pct = 100 * red_target / est_in_flux

    res_dict = {
        "conc_df": conc_df,
        "flux_df": flux_df,
        "src_df": src_df,
        f"trans_{par.lower()}": trans,
        "avg_ann_vol_m3": avg_ann_vol,
        "max_out_flux_ges_kg": max_out_flux_ges,
        "max_in_flux_ges_kg": max_in_flux_ges,
        "est_cur_in_flux_kg": est_in_flux,
        "reduction_target_kg": red_target,
        "reduction_target_pct": red_target_pct,
    }

    # Print summary
    print(f"Selected regine:\t{reg_id}.")
    print(f"Selected parameter:\t{par}.")
    print(f"Time period:\t\t{st_yr} to {end_yr}.")
    print(f"Concentration for GES:\t{ges_conc} ug/l.")
    print("")
    print(f"The regine contains {len(lake_id_list)} lake(s):")
    print(f"\tvatnLnrs:\t{lake_id_list}")
    print("")
    print(f"Estimated mean outflow concentration(s) ({st_yr}-{end_yr}):")
    for frac in fracs:
        col = [col for col in conc_df.columns if col.startswith(frac)][0]
        unit = col.split("_")[-1]
        value = conc_df[col].mean()
        label = f"{frac} ({unit})"
        print(f"\t{label:<30}{value:>10.1f}")
    print("")
    print(f"Estimated mean annual outflow flux(es) ({st_yr}-{end_yr}):")
    for frac in fracs:
        col = [col for col in flux_df.columns if col.startswith(frac)][0]
        unit = col.split("_")[-1]
        value = flux_df[col].mean()
        label = f"{frac} ({unit})"
        print(f"\t{label:<30}{value:>10.0f}")
    print("")
    print(f"Source apportionment for outflow {par}:")
    for src, pct in src_df[src_df.columns[0]].items():
        src = src.split("_")[0] + " (%)"
        if pct > 0:
            print(f"\t{src.capitalize():<30}{pct:>10.1f}")
    print("")
    print(f"Maximum outflow flux for GES (kg/yr)\t{max_out_flux_ges:>8.0f}")
    print(f"Transmission factor (weighted) for {par:<4}{trans:>9.2f}")
    print(f"Maximum inflow flux for GES (kg/yr)\t{max_in_flux_ges:>8.0f}")
    print(f"Estimated inflow flux {st_yr}-{end_yr} (kg/yr)\t{est_in_flux:>8.0f}")
    print(
        f"Load reduction target for GES (kg/yr)\t{red_target:>8.0f} ({red_target_pct:.1f}%)"
    )

    return res_dict
