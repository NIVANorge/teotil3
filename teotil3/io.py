import os
import re

import geopandas as gpd
import numpy as np
import pandas as pd
from sqlalchemy import text


def geodataframe_to_geopackage(
    gdf, gpkg_path, layer, cols=None, attrib_tab_csv=None, index=False
):
    """Save a geodataframe as a layer within a geopackage. If the geopackage doesn't already
    exist, it is created. If it does exist, the layer is appended. Options are included to
    subset/reorder columns before saving, and to save the attribute table as a separate CSV if
    desired.

    Args
        gdf: Geodataframe to save
        gpkg_path: Str. Path to geopackage where 'gdf' will be saved. If the geopackage does not
            exist, it will be created
        layer: Str. Name of layer to create within the geopackage. If the layer already exists,
            it will be overwritten.
        cols: List of str or None. Default None. Optionally subset or reorder columns in 'gdf'
            before saving. If None, all columns are saved in the original order
        attrib_tab_csv: Str or None. Default None. Optional path to CSV where the "attribute
            table" should be saved. This is a table containing all the non-spatial information
            from the geodataframe
        index: Bool. Default False. Whether to save the (geo)dataframe index

    Returns
        None. Data are saved to disk.
    """
    if cols:
        if "geometry" not in cols:
            cols.append("geometry")
            print(
                "WARNING: 'geometry' was not included in 'cols', but is required. It will be added."
            )
        gdf = gdf[cols].copy()

    gdf.to_file(gpkg_path, layer=layer, driver="GPKG", index=index)

    if attrib_tab_csv:
        df = gdf.drop(columns="geometry")
        df.to_csv(attrib_tab_csv, index=index)

    return None


def is_num(s):
    """Test if a string represents a number.

    Args
        s: Str. String to test

    Returns
        Bool. True if 's' can be converted to an integer, otherwise False.
    """
    s = str(s)
    try:
        int(s)
        return True
    except ValueError:
        return False


def assign_regine_hierarchy(
    df,
    regine_col="regine",
    regine_down_col="regine_down",
    order_coastal=True,
    nan_to_vass=False,
    land_to_vass=False,
    add_offshore=False,
):
    """Determine hydrological ordering of regine catchments based on catchment IDs. See

    https://www.nve.no/media/2297/regines-inndelingssystem.pdf

    for an overview of the regine coding scheme.

    The default parameters aim to reproduce the behaviour of the original model, but are not
    necessarily the best choices in most cases.

    Args
        df: (Geo)Dataframe of regine data
        regine_col: Str. Default 'regine'. Name of column in 'df' containing the regine IDs
        regine_down_col: Str. Default 'regine_down'. Name of new column to be created to store
            the next down IDs
        order_coastal: Bool. Default True. If True, coastal catchments will be ordered
            hierarchically, as in the original TEOTIL model (e.g. '001.6' flows into '001.5', which
            flows into '001.4' etc.). In most cases this does not make sense, as the numbering of
            kystfelt does not reflect dominant circulation patterns. If False all kystfelt are
            assigned directly to the main vassdragsområde (e.g. '001.6', '001.5' and '001.4' would
            all flow into '001.')
        nan_to_vass: Bool. Default False. Whether catchments for which no downstream ID can be
            identified should be assigned directly to the parent vassdragsområde. In some areas -
            notably catchments in the east draining to Sweden - the regines are truncated
            mid-catchment. There is therefore no valid downstream catchment. For completeness, it is
            often useful to assign these catchments to the parent vassdragsområde, rather than
            leaving them as NaN. This behaviour can be disabled by setting 'nan_to_vass' to False.
            If True, a warning will be printed if NaNs are filled for any catchments other than those
            draining to Sweden (vassdragsområder 301 to 315)
        land_to_vass: Bool. Default False. Whether downstream non-coastal catchments should be linked
            directly to the parent vassdragsområde, or to the highest ranking kystfelt. In the
            original TEOTIL model, terrestrial catchments are linked to the highest-ranking coastal
            fields. However, these are not necessarily anywhere near the catchment outflows. An
            alternative is to link terrestrial outflows directly to the vassdragområder, which is
            simpler but may give misleading results for some kystfelt with large river mouths. If
            'order_coastal' is True, 'land_to_vass' must be False. Catchments will then be arranged
            as for the original model. If 'order_coastal' is False, use this kwarg to control whether
            the algorithm links terrestrial fluxes to kystfelt, or directly to the parent vassdragområde
        add_offshore: Bool. Default False. Whether to add additional rows extending the hierarchy
            offshore to OSPAR areas (based on data hosted on GitHub)

    Returns
        (Geo)Dataframe. Copy of 'df' with two new columns added. The first is simply 'regine_col'
        cast to upper case; the second is named 'regine_down' and contains the regine ID of the next
        catchment downstream.

    Raises
        ValeError if 'regine_col' not in 'df'.
        ValueError if 'order_coastal', 'nan_to_vass' and 'add_offshore' are not Boolean.
        ValueError of 'order_coastal' is True AND 'land_to_vass' is True
        ValueError if if 'df' contains invalid column names.
    """
    if regine_col not in df.columns:
        raise ValueError(f"Column '{regine_col}' not found in 'df'.")
    for kwarg in [order_coastal, nan_to_vass, add_offshore]:
        if not isinstance(kwarg, bool):
            raise ValueError("Boolean keyword argument must be either True or False.")
    if order_coastal:
        if land_to_vass:
            raise ValueError(
                "'land_to_vass' must be False when 'order_coastal' is True."
            )
    for col in ["vassomx", "codex", regine_down_col]:
        if col in df.columns:
            raise ValueError(
                f"'df' cannot already contain column '{col}'. Please rename and try again."
            )

    df = df.copy()
    df[regine_col] = df[regine_col].str.upper()
    df[["vassomx", "codex"]] = df[regine_col].str.split(".", expand=True)
    df.sort_values(regine_col, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Sequence of alphanumeric codes for searching
    seq = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"

    # Process each vassdragsområde separately
    all_regines = []
    all_next_down = []
    for vassom in df["vassomx"].unique():
        vassom_df = df.query("vassomx == @vassom").reset_index(drop=True).copy()
        regines = vassom_df[regine_col].tolist()
        codes = vassom_df["codex"].tolist()
        next_down = []

        # Process each regine in the vassom
        for idx, reg in enumerate(regines):
            if idx == 0:
                # The first entry in the list is always the lowest in the vassom.
                # => Assign directly to vassom
                next_down.append(f"{vassom}.")
            else:
                # Walk hierarchy
                code = codes[idx]
                matches = []

                if is_num(code) and order_coastal is False:
                    next_down.append(f"{vassom}.")
                else:
                    while len(matches) < 1:
                        if code.endswith("0"):
                            # Already at the start of the sequence, so move one level up and continue
                            new_code = code[:-1]
                        else:
                            # Step down the sequence
                            last_char = code[-1]
                            last_char_pos = seq.find(last_char)
                            prev_char = seq[last_char_pos - 1]
                            new_code = code[:-1] + prev_char

                        if new_code == "":
                            # No match possible
                            break

                        # Try to match 'new_code' (which may be optionally sub-divided with additional numbers)
                        pattern = re.compile(f"^{new_code}([0-9]+)?$")
                        matches = list(filter(pattern.match, codes[:idx]))
                        code = new_code

                    if len(matches) == 0:
                        # Can only happen if new_code == '' and we broke out of the while loop
                        next_down.append(np.nan)
                    else:
                        # Assign the "highest-sorted" code that matches
                        match = matches[-1]
                        if is_num(match) and land_to_vass is True:
                            next_down.append(f"{vassom}.")
                        else:
                            next_down.append(f"{vassom}.{match}")

        assert len(regines) == len(next_down)
        all_regines += regines
        all_next_down += next_down

    assert len(all_regines) == len(all_next_down)
    next_down_df = pd.DataFrame(
        {regine_col: all_regines, regine_down_col: all_next_down}
    )
    del df["codex"]
    cols = [i for i in df.columns if i not in (regine_col)]
    df = pd.merge(df, next_down_df, how="left", on=regine_col)
    df = df[[regine_col, regine_down_col] + cols]

    if add_offshore:
        csv_path = r"https://raw.githubusercontent.com/NIVANorge/teotil3/main/data/offshore_hierarchy.csv"
        off_df = pd.read_csv(csv_path)
        df = pd.concat([df, off_df], axis="rows")

    if nan_to_vass:
        # Check all regines with NaN for next down ID drain to Sweden
        vassoms = df.query("regine_down != regine_down")["vassomx"].astype(int)
        if not ((301 <= vassoms) & (vassoms <= 315)).all():
            print(
                "WARNING: Not all NaNs drain to Sweden. Are you sure setting 'nan_to_vass' as True is OK?"
            )
            print(vassoms)
        df[regine_down_col].fillna(df["vassomx"] + ".", inplace=True)

    del df["vassomx"]

    pct_filled = 100 * df[regine_down_col].count() / len(df)
    print(f"{pct_filled:.2f} % of regines assigned.")

    # Fill NaN
    for col in df.columns:
        if col not in (regine_col, regine_down_col, "geometry"):
            if col.split("_")[0] == "trans":
                df[col].fillna(1, inplace=True)
            else:
                df[col].fillna(0, inplace=True)

    return df


def get_regine_geodataframe(engine, year):
    """Get the regine catchment polygons and basic attributes from PostGIS as a geodataframe.

    Args
        engine: SQL-Alchemy 'engine' object already connected to the 'teotil3' database
        year: Int. Year of interest

    Returns
        Geodataframe.

    Raises
        ValueError if 'year' is not an integer.
    """
    if not isinstance(year, int):
        raise ValueError("'year' must be an integer.")

    sql = text("SELECT * FROM teotil3.regines ORDER BY regine")
    gdf = gpd.read_postgis(sql, engine)

    # Get the year to use for administrative boundaries
    admin_year = 2014 if year < 2015 else year

    # Delete irrelevant columns and tidy
    del gdf["upstr_a_km2"]
    for col in gdf.columns:
        if col.startswith("fylnr") or col.startswith("komnr"):
            if not col.endswith(str(admin_year)):
                del gdf[col]
    gdf.rename(
        {
            f"fylnr_{admin_year}": "fylnr",
            f"komnr_{admin_year}": "komnr",
            "geom": "geometry",
        },
        axis="columns",
        inplace=True,
    )
    gdf.set_geometry("geometry", drop=False, inplace=True)

    return gdf


def get_annual_vassdrag_flow_factors(data_supply_year, year, engine):
    """Calculate flow factors for each vassdragområde based on modelled output from HBV. For each
    vassdragsområde, a correction factor is calculated as

        mean_flow_year / mean_flow_1991-2020

    These factors can then be multipled by the regine-level averages for 1991-2020 from NVE's
    new runoff dataset to estimate regine-level annual runoff and flow.

    Args
        data_supply_year: Int. Year of data delivery from NVE. NVE supplies complete data from
            1990 to present each year. The historic values change slightly as NVE update their
            models and input data. This argument specifies the NVE dataset to be used
        year: Int. Year of interest
        engine: SQL-Alchemy 'engine' object already connected to the 'teotil3' database

    Returns
        Dataframe

    Raises
        ValueError if 'year' or 'data_supply_year' are not integers.
        AssertionError if data are missing for some vassdragsområder.
    """
    if not isinstance(data_supply_year, int):
        raise ValueError("'data_supply_year' must be an integer.")
    if not isinstance(year, int):
        raise ValueError("'year' must be an integer.")

    sql = text(
        """
        SELECT a.vassom,
            (a."ann_flow_m3/s" / b."long_term_flow_m3/s") AS flow_factor
        FROM
            (SELECT vassom,
                AVG("flow_m3/s") AS "ann_flow_m3/s"
                FROM teotil3.nve_hbv_discharge
                WHERE data_supply_year = :data_supply_year
                AND DATE_PART('year', date) = :year
                GROUP BY vassom) a
        JOIN
            (SELECT vassom,
                AVG("flow_m3/s") AS "long_term_flow_m3/s"
                FROM teotil3.nve_hbv_discharge
                WHERE data_supply_year = :data_supply_year
                AND DATE_PART('year', date) > 1990
                AND DATE_PART('year', date) < 2021
                GROUP BY vassom) b
        ON a.vassom = b.vassom
        ORDER BY a.vassom
        """
    )
    param_dict = {"data_supply_year": data_supply_year, "year": year}
    df = pd.read_sql(sql, engine, params=param_dict)

    assert len(df) == 261, "Data are missing for some vassdragsområder."

    # NVE data for vassom 183 are very strange. For now, assign flow_factor=1
    # See issue here
    df.loc[df["vassom"] == "183", "flow_factor"] = 1

    return df


def rescale_annual_flows(reg_gdf, data_supply_year, year, engine):
    """Rescales flows for each regine based on annual HBV data from NVE. 'reg_gdf' must include
    columns named 'runoff_mm/yr' and 'q_cat_m3/s', each containing the average mean values for
    the period from 1991 to 2020 from NVE. It must also include a column named 'vassom' defining
    the vassdragsområder.

    Args
        reg_gdf: Geodataframe. Regine catchments with long-term mean flow values
        data_supply_year: Int. Year of data delivery from NVE. NVE supplies complete data from
            1990 to present each year. The historic values change slightly as NVE update their
            models and input data. This argument specifies the NVE dataset to be used
        year: Int. Year of interest
        engine: SQL-Alchemy 'engine' object already connected to the 'teotil3' database

    Returns
        Copy of 'reg_gdf' with flow-related columns modified to reflect the current year. Columns
        modified are ['runoff_mm/yr', 'q_cat_m3/s']

    Raises
        ValueError if 'year' or 'data_supply_year' are not integers.
    """
    if not isinstance(data_supply_year, int):
        raise ValueError("'data_supply_year' must be an integer.")
    if not isinstance(year, int):
        raise ValueError("'year' must be an integer.")

    reg_gdf = reg_gdf.copy()
    q_fac_df = get_annual_vassdrag_flow_factors(data_supply_year, year, engine)
    reg_gdf = pd.merge(reg_gdf, q_fac_df, how="left", on="vassom")
    for col in ["runoff_mm/yr", "q_cat_m3/s"]:
        reg_gdf[col] = reg_gdf[col] * reg_gdf["flow_factor"]
        reg_gdf[col].fillna(value=0, inplace=True)
    reg_gdf["runoff_mm/yr"] = reg_gdf["runoff_mm/yr"].round(0).astype(int)
    reg_gdf["q_cat_m3/s"] = reg_gdf["q_cat_m3/s"].round(6)
    del reg_gdf["flow_factor"]

    return reg_gdf


def get_retention_transmission_factors(year, keep="trans", csv_path=None):
    """Read regine-level retention and transmission coefficients.

    Args
        year: Int. Year of interest
        keep: Str. Default 'trans'. One of ['trans', 'ret', 'both']. Whether to return only
            transmission factors, only retention factors, or both
        csv_path: Str or None. Default None. Path to CSV file containing the retention and
            transmission factors. If None, the latest data will be read from GitHub

    Returns
        Dataframe.

    Raises
        ValueError if 'year' is not an integer.
        ValueError if 'keep' is not in ['trans', 'ret', 'both'].
    """
    if not isinstance(year, int):
        raise ValueError("'year' must be an integer.")
    if keep not in ["trans", "ret", "both"]:
        raise ValueError("'keep' must be one of ['trans', 'ret', 'both'].")

    url = r"https://raw.githubusercontent.com/NIVANorge/teotil3/main/data/regine_retention_transmission_10m_dem.csv"
    if csv_path:
        url = csv_path
    df = pd.read_csv(url)

    if keep != "both":
        factor_type = "trans_" if keep == "trans" else "ret_"
        cols = ["regine"] + [col for col in df.columns if col.startswith(factor_type)]
        df = df[cols]

    return df


def get_background_coefficients(year):
    """Read the static, spatially variable and spatio-temporally variable background coefficents
    and join to a single dataframe.

    Args
        year: Int. Year of interest

    Returns
        Dataframe.

    Raises
        ValueError if 'year' is not an integer.
    """
    if not isinstance(year, int):
        raise ValueError("'year' must be an integer.")

    # Define URLs for background coefficients
    base_url = "https://raw.githubusercontent.com/NIVANorge/teotil3/main/data/"
    urls = {
        "static": f"{base_url}spatially_static_background_coefficients.csv",
        "spatial": f"{base_url}spatially_variable_background_coefficients.csv",
        "spat_temp": f"{base_url}spatiotemporally_variable_background_coefficients.csv",
    }

    # Read background coefficients
    static_df = pd.read_csv(urls["static"])
    spatial_df = pd.read_csv(urls["spatial"])
    spat_temp_df = pd.read_csv(urls["spat_temp"])

    # Rename column for the specified year
    spat_temp_df.rename(
        columns={f"{year}_lake_din_kg/km2": "lake_din_kg/km2"}, inplace=True
    )
    spat_temp_df = spat_temp_df[["regine", "lake_din_kg/km2"]]

    # Merge dataframes
    df = pd.merge(spatial_df, spat_temp_df, on="regine", how="inner")
    for idx, row in static_df.iterrows():
        df[row["variable"]] = row["value"]

    return df


def calculate_background_inputs(reg_gdf):
    """Estimates non-agricultural diffuse inputs for each regine based on land cover, annual
    runoff and background coefficients.

    Args
        reg_gdf: (Geo)Dataframe

    Returns
        (Geo)Dataframe. Copy of 'reg_gdf' with new columns representing inputs for each
        parameter added and the background coefficients removed.

    Raises
        ValueError if units cannot be identified correctly from column names in 'reg_gdf'.
    """
    reg_gdf = reg_gdf.copy()
    reg_gdf["q_sp_l/km2"] = reg_gdf["runoff_mm/yr"] * 1e6

    # Define the parameters for concentration-based and area-based calculations
    concentration_based_pars = [
        "wood_totn_µg/l",
        "wood_din_µg/l",
        "wood_ton_µg/l",
        "wood_totp_µg/l",
        "wood_tdp_µg/l",
        "wood_tpp_µg/l",
        "wood_toc_mg/l",
        "upland_totn_µg/l",
        "upland_din_µg/l",
        "upland_ton_µg/l",
        "upland_totp_µg/l",
        "upland_tdp_µg/l",
        "upland_tpp_µg/l",
        "upland_toc_mg/l",
        "urban_totn_µg/l",
        "urban_din_µg/l",
        "urban_ton_µg/l",
        "urban_totp_µg/l",
        "urban_tdp_µg/l",
        "urban_tpp_µg/l",
        "urban_toc_µg/l",
        "urban_ss_mg/l",
    ]
    area_based_pars = [
        "lake_din_kg/km2",
        "wood_ss_kg/km2",
        "upland_ss_kg/km2",
        "glacier_ss_kg/km2",
    ]

    # Perform calculations for concentration-based parameters
    for col in concentration_based_pars:
        lc_class, par, unit = col.split("_")
        if unit[0] == "m":
            unit_fac = 1e6
        elif unit[0] == "µ":
            unit_fac = 1e9
        else:
            raise ValueError(f"Could not identify correct unit factor for {unit}.")

        reg_gdf[f"{lc_class}_{par}_kg"] = (
            reg_gdf[col]
            * reg_gdf[f"a_{lc_class}_km2"]
            * reg_gdf["q_sp_l/km2"]
            / unit_fac
        ).round(1)
        del reg_gdf[col]
    del reg_gdf["q_sp_l/km2"]

    # Perform calculations for area-based parameters
    for col in area_based_pars:
        lc_class, par, unit = col.split("_")
        if unit[0] == "k":
            unit_fac = 1
        else:
            raise ValueError(f"Could not identify correct unit factor for {unit}.")

        reg_gdf[f"{lc_class}_{par}_kg"] = (
            reg_gdf[col] * reg_gdf[f"a_{lc_class}_km2"]
        ).round(1)
        del reg_gdf[col]

    return reg_gdf


def make_input_file(
    year,
    nve_data_year,
    engine,
    out_csv_fold=None,
    nan_to_vass=True,
    add_offshore=True,
    order_coastal=False,
    land_to_vass=True,
    agri_loss_model="annual",
):
    """Builds an input file for the specified year. All the required data must be uploaded to
    the database.

    Args
        year: Int. Year of interest
        nve_data_year: Int. Specifies the discharge dataset to use from NVE
        engine: SQL-Alchemy 'engine' object already connected to the 'teotil3' database
        out_csv_fold: None or Str. Default None. Path to folder where output CSV will be created
        nan_to_vass: Bool. Default True. Additional kwarg passed to 'assign_regine_hierarchy'
            to determine hydrological connectivity
        add_offshore: Bool. Default True. Additional kwarg passed to 'assign_regine_hierarchy'
            to determine hydrological connectivity
        order_coastal: Bool. Default False. Additional kwarg passed to 'assign_regine_hierarchy'
            to determine hydrological connectivity
        land_to_vass: Bool. Default True. Additional kwarg passed to 'assign_regine_hierarchy'
            to determine hydrological connectivity
        agri_loss_model: Str. Default 'annual'. Either 'annual' or 'risk'. NIBIO's models estimate
            losses in two modes: 'annual' and 'risk'. 'annual' attempts to account for weather
            variability from year to year and should give the most accurate simulations of observed
            losses; 'risk' assumes a typical fixed year for weather and hydrology and is useful for
            exploring effects of changing agricultural management. 'annual' is usually the most
            appropriate choice for TEOTIL.

    Returns
        Dataframe. The CSV is written to the specified folder.

    Raises
        ValueError: If 'year' or 'nve_data_year' is not an integer.
        ValueError: If 'out_csv_fold' is not None or a string.
        ValueError: If 'nan_to_vass', 'add_offshore', 'order_coastal' or 'land_to_vass' are not
            boolean.
        ValueError: If 'agri_loss_model' is not 'annual' or 'risk'.
    """
    if not isinstance(year, int):
        raise ValueError("'year' must be an integer")
    if not isinstance(nve_data_year, int):
        raise ValueError("'nve_data_year' must be an integer")
    if out_csv_fold is not None and not isinstance(out_csv_fold, str):
        raise ValueError("'out_csv_fold' must be a directory path.")
    if not isinstance(nan_to_vass, bool):
        raise ValueError("'nan_to_vass' must be a boolean")
    if not isinstance(add_offshore, bool):
        raise ValueError("'add_offshore' must be a boolean")
    if not isinstance(order_coastal, bool):
        raise ValueError("'order_coastal' must be a boolean")
    if not isinstance(land_to_vass, bool):
        raise ValueError("'land_to_vass' must be a boolean")
    if agri_loss_model not in ["annual", "risk"]:
        raise ValueError("'agri_loss_model' must be either 'annual' or 'risk'")

    # Get basic datasets from database
    reg_gdf = get_regine_geodataframe(engine, year)
    ret_df = get_retention_transmission_factors(year, keep="trans")
    back_df = get_background_coefficients(year)
    spr_df = get_annual_spredt_data(engine, year)
    aqu_df = get_annual_point_data(engine, year, "aquaculture")
    ind_df = get_annual_point_data(engine, year, "industry")
    ww_df = get_annual_point_data(engine, year, "large wastewater")
    agri_df = get_annual_agriculture_data(engine, year, loss_type=agri_loss_model)

    # Rescale long-term flow averages to year of interest
    reg_gdf = rescale_annual_flows(reg_gdf, nve_data_year, year, engine)

    # Add retention factors
    reg_gdf = pd.merge(reg_gdf, ret_df, how="left", on="regine")

    # Estimate non-agricultural diffuse ("background") inputs
    reg_gdf = pd.merge(reg_gdf, back_df, how="left", on="regine")
    reg_gdf = calculate_background_inputs(reg_gdf)

    # Allocate outputs from "små renseanlegg"/spredt
    reg_gdf = assign_spredt_to_regines(reg_gdf, spr_df)

    # Join point inputs
    df_list = [reg_gdf, aqu_df, ind_df, ww_df]
    df_list = [i for i in df_list if i is not None]
    for df in df_list:
        df.set_index("regine", inplace=True)
    reg_gdf = pd.concat(df_list, axis="columns").reset_index()

    # Join agricultural inputs
    reg_gdf = pd.merge(reg_gdf, agri_df, how="left", on="regine")
    for col in agri_df.columns:
        if col != "regine":
            reg_gdf[col].fillna(0, inplace=True)

    # Determine hydrological connectivity
    reg_gdf = assign_regine_hierarchy(
        reg_gdf,
        regine_col="regine",
        regine_down_col="regine_down",
        nan_to_vass=nan_to_vass,
        add_offshore=add_offshore,
        order_coastal=order_coastal,
        land_to_vass=land_to_vass,
    )

    if out_csv_fold:
        # Save relevant cols to output
        cols_to_ignore = [
            "a_cat_poly_km2",
            "vassom",
            "ospar_region",
            "a_agri_km2",
            "a_glacier_km2",
            "a_lake_km2",
            "a_other_km2",
            "a_sea_km2",
            "a_upland_km2",
            "a_urban_km2",
            "a_wood_km2",
            "ar50_tot_a_km2",
            "a_lake_nve_km2",
            "komnr",
            "fylnr",
            "geometry",
        ]
        cols = [col for col in reg_gdf.columns if col not in cols_to_ignore]
        reg_df = reg_gdf[cols]
        csv_name = f"teotil3_input_data_nve{nve_data_year}_{year}.csv"
        csv_path = os.path.join(out_csv_fold, csv_name)
        reg_df.to_csv(csv_path, index=False)

    return reg_gdf


def get_annual_spredt_data(
    engine,
    year,
    par_list=[
        "totn_kg",
        "din_kg",
        "ton_kg",
        "totp_kg",
        "tdp_kg",
        "tpp_kg",
        "toc_kg",
    ],
):
    """Get annual 'spredt' data for each kommune. 'spredt' comprises wastewater treatment
    discharges from sources not connected to the main sewerage network (septic tanks etc.) or
    from 'små anlegg' sites that are too small (< 50 p.e.) to be reported individually.

    Args
        engine: SQL-Alchemy 'engine' object already connected to the 'teotil3' database
        year: Int. Year of interest
        par_list: List of parameters to consider. If None, returns data for all parameters in
            the database. The default is the basic set of parameters considered by TEOTIL3.

    Returns
        Dataframe of spredt data

    Raises
        ValueError if 'year' is not an integer.
        AssertionError if the spredt dataframe contains missing values.
    """
    if not isinstance(year, int):
        raise ValueError("'year' must be an integer.")

    sql = text(
        """
        SELECT a.komnr, 
            CONCAT_WS('_', c.name, c.unit) AS name, 
            (a.value*b.factor) AS value 
        FROM teotil3.spredt_inputs a, 
            teotil3.input_output_param_conversion b, 
            teotil3.output_param_definitions c 
        WHERE a.in_par_id = b.in_par_id 
        AND b.out_par_id = c.out_par_id 
        AND a.year = :year
        """
    )
    df = pd.read_sql(sql, engine, params={"year": year})

    if len(df) == 0:
        print(f"    No spredt data for {year}.")

        return None

    else:
        df = df.pivot(index="komnr", columns="name", values="value").copy()
        cols = [f"spredt_{col.lower()}" for col in df.columns]
        df.columns = cols
        df.columns.name = ""
        if par_list:
            cols = [
                f"spredt_{col}" for col in par_list if f"spredt_{col}" in df.columns
            ]
            df = df[cols]
        df.reset_index(inplace=True)

        assert pd.isna(df).sum().sum() == 0

        return df


def get_annual_point_data(
    engine,
    year,
    sector,
    par_list=[
        "totn_kg",
        "din_kg",
        "ton_kg",
        "totp_kg",
        "tdp_kg",
        "tpp_kg",
        "toc_kg",
        "ss_kg",
    ],
):
    """Get annual data for aquaculture, industry or wasterwater treatment from the database.

    Args
        year: Int. Year of interest
        sector: Str. One of ['aquaculture', 'industry', 'large wastewater', 'small wastewater']
        engine: SQL-Alchemy 'engine' object already connected to the 'teotil3' database
        par_list: List of parameters to consider. If None, returns data for all parameters in
            the database. The default is the basic set of parameters considered by TEOTIL3.

    Returns
        Dataframe of point data assigned to regines.

    Raises
        ValueError if 'sector' is not one of
            ['aquaculture', 'industry', 'large wastewater', 'small wastewater']
    """
    valid_sectors = [
        "aquaculture",
        "industry",
        "large wastewater",
        "small wastewater",
    ]
    sector = sector.lower()
    if sector not in valid_sectors:
        raise ValueError(
            "'sector' must be one of ['aquaculture', 'industry', 'large wastewater', 'small wastewater']."
        )

    sql = text(
        """
        SELECT d.regine,
            CONCAT_WS('_', c.name, c.unit) AS name,
            SUM(a.value * b.factor) AS value
        FROM teotil3.point_source_values a,
            teotil3.input_output_param_conversion b,
            teotil3.output_param_definitions c,
            (SELECT a.site_id,
                a.sector,
                a.year,
                b.regine
            FROM teotil3.point_source_locations a,
                teotil3.regines b
            WHERE ST_WITHIN(a.outlet_geom, b.geom)
            ) d
        WHERE a.in_par_id = b.in_par_id
            AND b.out_par_id = c.out_par_id
            AND a.site_id = d.site_id
            AND a.year = :year
            AND d.year = :year
            AND d.sector = :sector
        GROUP BY d.regine,
            c.name,
            c.unit
    """
    )
    df = pd.read_sql(sql, engine, params={"year": year, "sector": sector.capitalize()})

    if len(df) == 0:
        print(f"    No {sector} data for {year}.")

        return None

    else:
        df = df.pivot(index="regine", columns="name", values="value").copy()
        cols = [f"{sector.replace(' ', '-')}_{col.lower()}" for col in df.columns]
        df.columns = cols
        df.columns.name = ""
        if par_list:
            cols = [
                f"{sector.replace(' ', '-')}_{col}"
                for col in par_list
                if f"{sector.replace(' ', '-')}_{col}" in df.columns
            ]
            df = df[cols].copy()
        df = df.round(1)
        df.reset_index(inplace=True)

        return df


def assign_spredt_to_regines(
    reg_gdf,
    spr_df,
    par_list=[
        "totn_kg",
        "din_kg",
        "ton_kg",
        "totp_kg",
        "tdp_kg",
        "tpp_kg",
        "toc_kg",
    ],
):
    """Kommune level totals for spredt in 'spr_df' area are assigned to regines in 'reg_gdf'. Spredt
    is distributed evenly over all agricultural land in each kommune (if agricultural land
    exists) and otherwise it is simply distributed evenly over all land.

    Args
        reg_gdf: Geodataframe of regine data.
        spr_df: Dataframe of kommune level spredt data
        par_list: List of parameters to consider. If None, returns data for all parameters in
            the database. The default is the basic set of parameters considered by TEOTIL3.

    Returns
        Geodataframe. New columns are added to 'reg_gdf' indicting the spredt inputs to each
        regine.
    """
    kom_df = reg_gdf[["komnr", "a_cat_land_km2", "a_agri_km2"]].copy()
    kom_df = kom_df.groupby("komnr").sum().reset_index()
    kom_df.columns = ["komnr", "a_kom_km2", "a_agri_kom_km2"]

    if spr_df is not None:
        kom_df = pd.merge(kom_df, spr_df, how="left", on="komnr")
    else:  # Create cols of zeros
        for par in par_list:
            kom_df[f"spredt_{par}"] = 0

    # Join back to main df
    reg_gdf = pd.merge(reg_gdf, kom_df, how="left", on="komnr")

    # Distribute loads
    for par in par_list:
        # Over agri
        reg_gdf["spredt_agri"] = (
            reg_gdf[f"spredt_{par}"] * reg_gdf["a_agri_km2"] / reg_gdf["a_agri_kom_km2"]
        )
        # Over all area
        reg_gdf["spredt_all"] = (
            reg_gdf[f"spredt_{par}"] * reg_gdf["a_cat_land_km2"] / reg_gdf["a_kom_km2"]
        )

        # Use agri if > 0, else all
        reg_gdf[f"spredt_{par}"] = np.where(
            reg_gdf["a_agri_kom_km2"] > 0, reg_gdf["spredt_agri"], reg_gdf["spredt_all"]
        )
    reg_gdf.drop(
        ["spredt_agri", "spredt_all", "a_kom_km2", "a_agri_kom_km2"],
        inplace=True,
        axis="columns",
    )
    for par in par_list:
        reg_gdf[f"spredt_{par}"] = reg_gdf[f"spredt_{par}"].fillna(value=0).round(1)

    return reg_gdf


def get_point_time_series(
    site_list,
    st_yr,
    end_yr,
    eng,
    par_list=[
        "totn_kg",
        "din_kg",
        "ton_kg",
        "totp_kg",
        "tdp_kg",
        "tpp_kg",
        "toc_kg",
        "ss_kg",
    ],
):
    """Get annual discharge time series for a list of sites.

    Args
        site_list: List of str. List of site IDs ('anleggsnr') of interest
        st_yr: Int. Start year of interest
        end_yr: Int. End year of interest
        eng: Obj. Active database engine connected to the TEOTIL3 database
        par_list. List of str or None. List of parameters to consider. If None, returns data for
            all parameters in the database. The default is the basic set of parameters considered
            by TEOTIL3.

    Returns
        Geodataframe of point data, or None if not data are available.

    Raises
        TypeError if 'site_list' or 'par_list' are not lists.
        TypeError if 'st_yr' and 'end_yr' are not integers.
    """
    if not isinstance(site_list, list):
        raise TypeError("'site_list' must be a list.")
    if not isinstance(par_list, list):
        raise TypeError("'par_list' must be a list.")
    if not isinstance(st_yr, int):
        raise TypeError("'st_yr' must be an integer.")
    if not isinstance(end_yr, int):
        raise TypeError("'end_yr' must be an integer.")

    # Get point data
    sql = text(
        "SELECT a.site_id, "
        "  a.name, "
        "  a.sector, "
        "  a.type, "
        "  a.year, "
        "  a.outlet_geom AS geometry, "
        "  CONCAT_WS('_', d.name, d.unit) AS par_name, "
        "  (b.value * c.factor) AS value "
        "FROM teotil3.point_source_locations a, "
        "  teotil3.point_source_values b, "
        "  teotil3.input_output_param_conversion c, "
        "  teotil3.output_param_definitions d "
        "WHERE b.in_par_id = c.in_par_id "
        "  AND c.out_par_id = d.out_par_id "
        "  AND a.site_id = b.site_id "
        "  AND a.year >= :st_yr "
        "  AND a.year <= :end_yr "
        "  AND a.year = b.year "
        "  AND a.site_id IN :site_list"  # Modified to use IN clause
    )
    gdf = gpd.read_postgis(
        sql,
        eng,
        geom_col="geometry",
        params={"st_yr": st_yr, "end_yr": end_yr, "site_list": tuple(site_list)},
    )

    if len(gdf) == 0:
        print(f"No data for selected sites within the period of interest.")
        return None
    else:
        id_cols = ["site_id", "name", "sector", "type", "year", "geometry"]
        gdf = gdf.pivot(index=id_cols, columns="par_name", values="value").copy()
        if par_list:
            cols = [col for col in gdf.columns if col.lower() in par_list]
            gdf = gdf[cols].copy()
        gdf = gdf.round(1)
        gdf.reset_index(inplace=True)
        gdf = gdf.sort_values(["site_id", "year"])

        return gdf


def get_annual_agriculture_data(eng, year, loss_type="annual"):
    """Get annual agriculture data for regines based on modelling undertaken by NIBIO.

    Args
        eng: Obj. Active SQL-Alchemy 'engine' object connected to the 'teotil3' database
        year: Int. Year of interest
        loss_type: Str. Default 'annual'. Either 'annual' or 'risk'. NIBIO's models estimate losses
            in two modes: 'annual' and 'risk'. 'annual' attempts to account for weather variability
            from year to year and should give the most accurate simulations of observed losses;
            'risk' assumes a typical fixed year for weather and hydrology and is useful for
            exploring effects of changing agricultural management. 'annual' is usually the most
            appropriate choice for TEOTIL.

    Returns
        Dataframe of annual agricultural inputs.

    Raises
        TypeError if 'year' is not an integer.
        ValueError if 'loss_type' is not in ['annual', 'risk'].
    """
    if not isinstance(year, int):
        raise TypeError("'year' must be an integer.")
    if loss_type not in ("annual", "risk"):
        raise ValueError("'loss_type' must be either 'annual' or 'risk'.")

    sql = text(
        "SELECT * FROM teotil3.agri_inputs "
        "WHERE year = :year "
        "AND loss_type = :loss_type"
    )
    df = pd.read_sql(sql, eng, params={"year": year, "loss_type": loss_type})
    df.drop(columns=["loss_type", "year"], inplace=True)

    if len(df) == 0:
        print(f"    No agriculture data for {year}.")

        return None

    else:
        df.columns = [
            col.replace("agriculture_background", "agriculture-background")
            for col in df.columns
        ]
        return df
