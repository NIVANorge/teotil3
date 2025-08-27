import itertools
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
from rasterstats import zonal_stats
from sqlalchemy import exc, text


def read_raw_regine_data(geodatabase_path, layer_name):
    """Read basic regine properties from an NVE geodatabase. Calculates accurate polygon areas
    and removes catchments on Svalbard.

    Args
        geodatabase_name: Str. Path to geodatabase
        layer_name: Str. Name of regine layer in geodatabase

    Returns
        Geodataframe with columns 'regine', 'vassom', 'a_cat_poly_km2' and 'upstr_a_km2'.
    """
    # Read the geodatabase file
    reg_gdf = gpd.read_file(geodatabase_path, driver="fileGDB", layer=layer_name)

    # Rename the columns
    reg_gdf.rename(
        {
            "vassdragsnummer": "regine",
            "nedborfeltOppstromAreal_km2": "upstr_a_km2",
        },
        axis="columns",
        inplace=True,
    )

    # Remove Svalbard
    reg_gdf["vassom"] = reg_gdf["regine"].str.split(".", n=1).str[0].astype(int)
    reg_gdf = reg_gdf.query("vassom < 400").copy()
    reg_gdf["vassom"] = reg_gdf["vassom"].apply(lambda x: f"{x:03}")

    # Calculate polygon area
    reg_gdf["a_cat_poly_km2"] = reg_gdf.to_crs({"proj": "cea"})["geometry"].area / 1e6

    # Get columns of interest
    reg_cols = [
        "regine",
        "vassom",
        "a_cat_poly_km2",
        "upstr_a_km2",
        "geometry",
    ]
    reg_gdf = reg_gdf[reg_cols]

    # Sort by 'regine' and reset index
    reg_gdf.sort_values("regine", inplace=True)
    reg_gdf.reset_index(inplace=True, drop=True)

    return reg_gdf


def summarise_regine_hydrology(reg_gdf, ro_grid_path, all_touched=True):
    """Summarise basic, regine-level hydrology information based on NVE's latest "runoff normal"
    for 1991 to 2020.

    Args
        reg_gdf: Geodataframe of regines. Must contain column named 'a_cat_land_km2'
            representing the land area in each regine
        ro_grid_path: Str. Path to 1 km2 runoff grid for 1991-2020 from NVE
        all_touched: Bool. Default True. Defines the rasterisation strategy. See
            https://pythonhosted.org/rasterstats/manual.html#rasterization-strategy

    Returns
        Geodataframe. Copy of the original geodataframe with columns 'runoff_mm/yr' and
        'q_cat_m3/s' added.
    """
    reg_gdf = reg_gdf.copy()

    stats = ["mean"]

    # Check the coordinate reference system
    if reg_gdf.crs.to_epsg() != 25833:
        stats_gdf = reg_gdf.to_crs("epsg:25833")
    else:
        stats_gdf = reg_gdf.copy()

    # Calculate zonal statistics
    df = pd.DataFrame(
        zonal_stats(
            vectors=stats_gdf,
            raster=ro_grid_path,
            stats=stats,
            all_touched=all_touched,
        )
    )

    assert len(reg_gdf) == len(df)

    # Add new columns to the geodataframe
    reg_gdf["runoff_mm/yr"] = df["mean"].round(0)
    reg_gdf["runoff_mm/yr"].fillna(0, inplace=True)
    reg_gdf["runoff_mm/yr"] = reg_gdf["runoff_mm/yr"].astype(int)
    reg_gdf["q_cat_m3/s"] = (
        reg_gdf["runoff_mm/yr"] * reg_gdf["a_cat_land_km2"] * 1e6
    ) / (1000 * 60 * 60 * 24 * 365.25)
    reg_gdf["q_cat_m3/s"] = reg_gdf["q_cat_m3/s"].round(5)

    # Move 'geometry' to end
    cols = reg_gdf.columns.tolist()
    cols.remove("geometry")
    cols.append("geometry")
    reg_gdf = reg_gdf[cols]

    return reg_gdf


def assign_regines_to_administrative_units(reg_gdf, admin_gpkg, admin_year):
    """Assign each regine in 'reg_gdf' to a single fylke and kommune.

    Args
        reg_gdf: Geodataframe of regine boundaries from NVE. Must contain a column named
            'regine' with the regine IDs
        admin_gpkg: Str. Path to geopackage containing administrative data
        admin_year: Int. Year for administrative boundaries read from 'admin_gpkg'

    Returns
        Geodataframe. A copy of the original geodataframe, but with new columns named 'fylnr' and
        'komnr' added. Regines that cannot be assigned to administrative units (e.g. because they
        lie over the border in Sweden or Finland) are assigned a value of -1.
    """
    reg_gdf = reg_gdf.copy()

    for admin_unit in ["kommuner", "fylker"]:
        print(f"Processing {admin_unit}")

        col_name = admin_unit[:3] + "nr"
        adm_gdf = gpd.read_file(
            admin_gpkg, layer=f"{admin_unit}{admin_year}"
        )
        adm_gdf = adm_gdf[[col_name, "geometry"]]

        # Intersect
        int_gdf = gpd.overlay(reg_gdf, adm_gdf, how="intersection", keep_geom_type=True)
        print(f"   {len(adm_gdf)} {admin_unit}.")
        print(f"   {len(reg_gdf)} regines.")
        print(f"   {len(int_gdf)} intersected polygons.")

        # Find dominant admin unit for each regine
        int_gdf["area_km2"] = int_gdf.to_crs({"proj": "cea"})["geometry"].area / 1e6
        int_gdf = int_gdf[["regine", col_name, "area_km2"]]
        int_gdf.sort_values("area_km2", inplace=True)
        int_gdf.drop_duplicates("regine", keep="last", inplace=True)
        del int_gdf["area_km2"]
        reg_gdf = pd.merge(reg_gdf, int_gdf, how="left", on="regine")
        reg_gdf[col_name].fillna("-1", inplace=True)

        print(f"   {len(int_gdf)} regines assigned to {admin_unit}.")

    # Move 'geometry' to end
    cols = reg_gdf.columns.tolist()
    cols.remove("geometry")
    cols.append("geometry")
    reg_gdf = reg_gdf[cols]

    return reg_gdf


def assign_regines_to_ospar_regions(
    reg_gdf,
    ospar_csv=r"https://raw.githubusercontent.com/NIVANorge/teotil3/main/data/vassdragsomrader_ospar_regions.csv",
):
    """Assign each regine in 'reg_gdf' to a single OSPAR region.

    Args
        reg_gdf: Geodataframe of regine boundaries from NVE. Must contain a column named 'vassom'
            storing the vassdragsområde for each regine
        ospar_csv: Str. Default is a data table hosted on GitHub. Path to CSV mapping
            vassdragsområder to OSPAR regions

    Returns
        Geodataframe. A copy of the original geodataframe with a new column named 'ospar_region'
        appended.
    """
    reg_gdf = reg_gdf.copy()
    osp_df = pd.read_csv(ospar_csv)
    osp_df["vassom"] = osp_df["vassom"].apply(lambda x: f"{x:03}")
    reg_gdf = pd.merge(reg_gdf, osp_df, how="left", on="vassom")

    # Move 'geometry' to end
    cols = reg_gdf.columns.tolist()
    cols.remove("geometry")
    cols.append("geometry")
    reg_gdf = reg_gdf[cols]

    return reg_gdf


def calculate_ar50_land_cover_proportions(
    reg_gdf,
    ar50_gdf,
    land_class_csv=r"https://raw.githubusercontent.com/NIVANorge/teotil3/main/data/ar50_artype_classes.csv",
):
    """Calculate land cover proportions for each regine based on NIBIO's AR50 dataset.

    Args
        reg_gdf: Geodataframe of regine boundaries from NVE. Must contain a column named 'regine'
            with the regine IDs
        ar50_gdf: Geodataframe of AR50 data from NIBIO. Must contain a column named 'artype'
        land_class_csv: Str. Default is a data table hosted on GitHub. Path to CSV mapping AR50
            classes to those used by TEOTIL

    Returns
        Geodataframe with additional columns showing the area of each class in 'land_class_csv'
        in each regine.
    """
    reg_gdf = reg_gdf.copy()

    print("Reclassifying...")
    artype_df = pd.read_csv(land_class_csv)
    ar50_gdf = pd.merge(ar50_gdf, artype_df, how="left", on="artype")
    ar50_gdf = ar50_gdf[["teotil", "geometry"]]

    print("Reprojecting to equal area...")
    reg_gdf_cea = reg_gdf.to_crs({"proj": "cea"})
    ar50_gdf_cea = ar50_gdf.to_crs({"proj": "cea"})

    print("Intersecting polygons...")
    int_gdf = gpd.overlay(
        reg_gdf_cea, ar50_gdf_cea, how="intersection", keep_geom_type=True
    )
    int_gdf["area_km2"] = int_gdf["geometry"].area / 1e6

    print("Aggregating...")
    lc_df = int_gdf.groupby(["regine", "teotil"]).sum(numeric_only=True)["area_km2"]
    lc_df = lc_df.unstack("teotil")
    lc_df.columns = [f"a_{i}_km2" for i in lc_df.columns]
    lc_df.reset_index(inplace=True)
    lc_df.columns.name = ""
    reg_gdf = pd.merge(reg_gdf, lc_df, on="regine", how="left")
    cols = [
        "a_agri_km2",
        "a_glacier_km2",
        "a_lake_km2",
        "a_sea_km2",
        "a_upland_km2",
        "a_urban_km2",
        "a_wood_km2",
    ]
    for col in cols:
        reg_gdf[col].fillna(0, inplace=True)

    # Some regines lie wholly or partially outside Norway/AR50. These parts should
    # be considered 100% 'other'
    reg_gdf["a_other_km2"] = (
        reg_gdf["a_cat_poly_km2"]
        - reg_gdf["a_agri_km2"]
        - reg_gdf["a_glacier_km2"]
        - reg_gdf["a_lake_km2"]
        - reg_gdf["a_sea_km2"]
        - reg_gdf["a_upland_km2"]
        - reg_gdf["a_urban_km2"]
        - reg_gdf["a_wood_km2"]
    ).clip(lower=0)

    # Sum the total AR50 area for classes that are not 'sea' or 'other'
    reg_gdf["ar50_tot_a_km2"] = (
        reg_gdf["a_agri_km2"]
        + reg_gdf["a_glacier_km2"]
        + reg_gdf["a_lake_km2"]
        + reg_gdf["a_upland_km2"]
        + reg_gdf["a_urban_km2"]
        + reg_gdf["a_wood_km2"]
    )

    # Calculate land area for each regine
    reg_gdf["a_cat_land_km2"] = reg_gdf["a_cat_poly_km2"] - reg_gdf["a_sea_km2"]

    # Move 'geometry' to end
    cols = reg_gdf.columns.tolist()
    cols.remove("geometry")
    cols.append("geometry")
    reg_gdf = reg_gdf[cols]

    return reg_gdf


def calculate_nve_regine_lake_areas(reg_gdf, lake_gdf):
    """Calculate the total area of lakes in each regine based on NVE's innsjø database.

    Args
        reg_gdf:  Geodataframe of regine boundaries from NVE. Must contain a column named 'regine'
            with the regine IDs
        lake_gdf: Geodataframe of lake polygons from NVE

    Returns
        Geodataframe with an additional column named 'a_lake_nve_km2'.
    """
    reg_gdf = reg_gdf.copy()
    reg_gdf_cea = reg_gdf.to_crs({"proj": "cea"})
    lake_gdf_cea = lake_gdf.to_crs({"proj": "cea"})
    int_gdf = gpd.overlay(
        reg_gdf_cea, lake_gdf_cea, how="intersection", keep_geom_type=True
    )
    int_gdf["a_lake_nve_km2"] = int_gdf["geometry"].area / 1e6
    int_gdf = (
        int_gdf.groupby("regine")["a_lake_nve_km2"].sum(numeric_only=True).reset_index()
    )
    reg_gdf = pd.merge(reg_gdf, int_gdf, on="regine", how="left")
    reg_gdf["a_lake_nve_km2"].fillna(0, inplace=True)

    # Move 'geometry' to end
    cols = [col for col in reg_gdf.columns if col != "geometry"] + ["geometry"]
    reg_gdf = reg_gdf[cols]

    return reg_gdf


def transmission_sigma_constant(tau, sigma):
    """Estimate lake transmission from water residence time using a "basic" Vollenweider model:

        T = 1 / (1 + sigma * tau)

    Args
        tau: Array-like. Lake water residence times in years
        sigma: Float. First-order rate constant for removal processes (units per year)

    Returns
        Array of transmission factors.
    """
    return 1 / (1 + (sigma * tau))


def transmission_sigma_from_tau(tau, k, p):
    """Estimate lake transmission from water residence time using a model where sigma is a
    function of tau.

        T = 1 / (1 + k * tau ** p)

    Args
        tau: Array-like. Lake water residence times in years
        k: Float. Model parameter
        p: Float. Model parameter

    Returns
        Array of transmission factors.
    """
    return 1 / (1 + (k * (tau**p)))


def transmission_sigma_from_depth(H, s):
    """Estimate lake transmission from water residence time using a model where sigma is a
    function of mean lake depth.

        T = 1 / (1 + (s / H))

    Args
        H: Array-like. Lake hydraulic load (m/year)
        s: Float. Apparent settling velocity (m/year)

    Returns
        Array of transmission factors.
    """
    return 1 / (1 + (s / H))


def calculate_lake_retention_vollenweider(df, par_name, params):
    """Calculate retention and transmission factors for individual lakes according to various
    Vollenweider-like models.

    Args
        df: Dataframe of lake data
        par_name: Str. Name for parameter
        params: Dict. Must contains the following keys:
                    ind_var_col: Str. Column name in 'df' containing the independent variable
                        (e.g. 'tau' or 'H')
                    model: Str. One of ['sigma_constant', 'sigma_from_tau', 'sigma_from_depth']
                    <other>: Passed as kwargs to the relevant 'model' function

    Returns
        Dataframe. 'df' is returned with two new columns added: 'trans_par' and 'ret_par', where
        'par' is the 'par_name' provided.

    Raises
        ValueError is 'model' is not valid.
        ValueError if 'ind_var_col' not in columns of 'df'.
    """
    ind_var_col = params.pop("ind_var_col")
    model = params.pop("model")

    model_funcs = {
        "sigma_constant": transmission_sigma_constant,
        "sigma_from_tau": transmission_sigma_from_tau,
        "sigma_from_depth": transmission_sigma_from_depth,
    }

    if model not in model_funcs:
        raise ValueError(f"'model' must be one of {list(model_funcs.keys())}.")

    if ind_var_col not in df.columns:
        raise ValueError(f"'{ind_var_col}' not found in 'df'.")

    trans = model_funcs[model](df[ind_var_col], **params)

    df[f"trans_{par_name}"] = trans
    df[f"ret_{par_name}"] = 1 - trans

    return df


def calculate_regine_retention(df, regine_col, pars):
    """
    Aggregate lake retention factors to regine level by combining in series.

    Args
        df: Dataframe of lake retention data. Must contain cols 'trans_par' for all 'pars'
        regine_col: Str. Column name in 'df' with regine codes for aggregation
        pars: List of str. Parameters to aggregate

    Returns
        DataFrame: Dataframe with columns [regine, trans_par, ret_par] for all parameters in
        'pars'.

    Raises
        ValueError: If 'regine_col' not found in 'df' or if any 'trans_{par}' not found in
        'df'.
    """
    if regine_col not in df.columns:
        raise ValueError(f"'{regine_col}' not found in 'df'.")

    trans_cols = [f"trans_{par}" for par in pars]
    for col in trans_cols:
        if col not in df.columns:
            raise ValueError(f"'{col}' not found in 'df'.")

    reg_df = df.groupby(regine_col).prod()[trans_cols].round(6).reset_index()
    for par in pars:
        reg_df[f"ret_{par}"] = (1 - reg_df[f"trans_{par}"]).round(6)

    return reg_df


def assign_regine_retention(reg_gdf, regine_col="regine", dtm_res=10, voll_dict=None):
    """
    Assign retention and transmission coefficients to each regine.

    Args
        reg_gdf: Geodataframe of regine boundaries from NVE.
        regine_col: Str. Name of column in reg_gdf with regine codes. Default is "regine".
        dtm_res: Int. Resolution in file name of CSV with residence times. Default is 10.
        voll_dict: Dict or None. Dictionary of Vollenweider parameters to use for estimating
            retention and transmission for individual lakes. If None, default values based on
            literature analysis are used.

    Returns
        GeoDataFrame: Copy of 'reg_gdf' with retention and transmission columns added for each
        parameter.

    Raises
        ValueError: If 'regine_col' not found in 'reg_gdf' or if any required column not found
        in 'df'.
    """
    if regine_col not in reg_gdf.columns:
        raise ValueError(f"'{regine_col}' not found in 'reg_gdf'.")

    reg_gdf = reg_gdf.copy()

    # Get lake residence times
    res_csv = f"https://raw.githubusercontent.com/NIVANorge/teotil3/main/data/lake_residence_times_{dtm_res}m_dem.csv"
    df = pd.read_csv(res_csv)

    # Vollenweider parameters for individual lakes
    if voll_dict is None:
        # Original parameters derived from literature data
        # voll_dict = {
        #     "totp": {
        #         "ind_var_col": "res_time_yr",
        #         "model": "sigma_from_tau",
        #         "k": 1,
        #         "p": 0.5,
        #     },
        #     "tdp": {
        #         "ind_var_col": "res_time_yr",
        #         "model": "sigma_from_tau",
        #         "k": 0.5,
        #         "p": 0.5,
        #     },
        #     "tpp": {
        #         "ind_var_col": "res_time_yr",
        #         "model": "sigma_from_tau",
        #         "k": 2,
        #         "p": 0.5,
        #     },
        #     "totn": {"ind_var_col": "hyd_load_mpyr", "model": "sigma_from_depth", "s": 4.5},
        #     "din": {"ind_var_col": "hyd_load_mpyr", "model": "sigma_from_depth", "s": 6.0},
        #     "ton": {"ind_var_col": "hyd_load_mpyr", "model": "sigma_from_depth", "s": 1.4},
        #     "ss": {"ind_var_col": "res_time_yr", "model": "sigma_constant", "sigma": 90},
        #     "toc": {
        #         "ind_var_col": "res_time_yr",
        #         "model": "sigma_from_tau",
        #         "k": 0.6,
        #         "p": 0.4,
        #     },
        # }
        # Current default parameters modified from literature values during calibration
        voll_dict = {
            "totp": {
                "ind_var_col": "res_time_yr",
                "model": "sigma_from_tau",
                "k": 1,
                "p": 0.5,
            },
            "tdp": {
                "ind_var_col": "res_time_yr",
                "model": "sigma_from_tau",
                "k": 0.2,
                "p": 0.5,
            },
            "tpp": {
                "ind_var_col": "res_time_yr",
                "model": "sigma_from_tau",
                "k": 3,
                "p": 0.5,
            },
            "totn": {
                "ind_var_col": "hyd_load_mpyr",
                "model": "sigma_from_depth",
                "s": 6,
            },
            "din": {
                "ind_var_col": "hyd_load_mpyr",
                "model": "sigma_from_depth",
                "s": 8,
            },
            "ton": {
                "ind_var_col": "hyd_load_mpyr",
                "model": "sigma_from_depth",
                "s": 3,
            },
            "ss": {
                "ind_var_col": "res_time_yr",
                "model": "sigma_constant",
                "sigma": 5,
            },
            "toc": {
                "ind_var_col": "res_time_yr",
                "model": "sigma_from_tau",
                "k": 0.4,
                "p": 0.4,
            },
        }
    for par, params in voll_dict.items():
        df = calculate_lake_retention_vollenweider(df, par, params)

    # Aggregate to regine level
    pars = [col[6:] for col in df.columns if col.startswith("trans_")]
    reg_df = calculate_regine_retention(df, regine_col=regine_col, pars=pars)

    reg_gdf = pd.merge(reg_gdf, reg_df, on=regine_col, how="left")

    for col in reg_gdf.columns:
        if col.startswith("ret_"):
            reg_gdf[col].fillna(0, inplace=True)
        elif col.startswith("trans_"):
            reg_gdf[col].fillna(1, inplace=True)
        else:
            pass

    return reg_gdf


def read_raw_aquaculture_data(xl_path, sheet_name, year):
    """Read the raw aquaculture data from Fiskeridirektoratet. Returns a dataframe of site
    locations, plus a dataframe of raw monthly data for further processing.

    Args
        xl_path: Str. Path to Excel file from Fiskeridirektoratet
        sheet_name: Str. Worksheet to read
        year: Int. Year being processed

    Returns
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple of (geo)dataframes (loc_gdf, data_df).

    Raises
        TypeError 'xl_path' or 'sheet_name' are not strings.
        TypeError if 'year' is not an integer.
    """
    if not isinstance(xl_path, str):
        raise TypeError("'xl_path' must be a valid file path.")
    if not isinstance(sheet_name, str):
        raise TypeError("'sheet_name' must be a string.")
    if not isinstance(year, int):
        raise TypeError("'year' must be an integer.")

    # Read raw file
    df = pd.read_excel(xl_path, sheet_name=sheet_name)
    df.dropna(how="all", inplace=True)
    df.rename(
        {
            "AAR": "year",
            "LOKNR": "site_id",
            "LOKNAVN": "name",
            "N_DESIMALGRADER_Y": "lat",
            "O_DESIMALGRADER_X": "lon",
        },
        axis="columns",
        inplace=True,
    )
    df = df.query("year == @year")
    df["site_id"] = df["site_id"].astype(str)
    df["sector"] = "Aquaculture"
    df["type"] = "Fiskeoppdrett i sjøvann"

    # Check for missing co-ords
    no_coords_df = df.query("(lat != lat) or (lon != lon)")[
        ["site_id", "name"]
    ].sort_values("site_id")
    if len(no_coords_df) > 0:
        print(
            f"{len(no_coords_df)} locations do not have co-ordinates in this year's data."
        )
        print(no_coords_df)

    df.dropna(subset=["lat", "lon"], how="any", inplace=True)

    # Build geodataframe of locations
    loc_gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"], crs="epsg:4326")
    )
    loc_gdf = loc_gdf.to_crs("epsg:25833")
    loc_gdf.rename({"geometry": "outlet_geom"}, axis="columns", inplace=True)
    loc_gdf["site_geom"] = loc_gdf["outlet_geom"].copy()
    loc_gdf.set_geometry("outlet_geom", inplace=True)
    loc_gdf.reset_index(drop=True, inplace=True)

    # The name sometimes changes even if the site is identical
    uniq_cols = ["site_id", "sector", "type", "year", "site_geom", "outlet_geom"]
    loc_gdf = loc_gdf[uniq_cols + ["name"]].drop_duplicates(subset=uniq_cols)

    return loc_gdf, df


def estimate_aquaculture_nutrient_inputs(
    df, year, eng, cu_tonnes=None, species_ids=[71401, 71101]
):
    """Estimate losses of nutrients from aquaculture.

    Args
        df: Dataframe of raw monthly data
        year: Int. Year of interest
        eng: Obj. Active database connection object connected to PostGIS
        cu_tonnes: Float. Optional. Total annual usage of copper by the aquaculture industry in
            tonnes. If supplied, 85% of this value is assumed to be lost to the environment.
            Losses are assigned to each aquaculture site in proportion to the total loss of P
        species_ids: List. Species to consider. The default is salmon and rainbow trout

    Returns
        Dataframe of estimated nutrient losses that can be added to teotil3.point_source_values.

    Raises
        TypeError if 'year' is not an integer.
        TypeError if 'cu_tonnes' is not a number.
        TypeError if 'species_ids' is not a list.
    """
    if not isinstance(year, int):
        raise TypeError("'year' must be an integer.")
    if not isinstance(species_ids, list):
        raise TypeError("'species_ids' must be a list.")
    if cu_tonnes:
        if not isinstance(cu_tonnes, (int, float, np.number)):
            raise TypeError("'cu_tonnes' must be a number.")

    # Read coefficients for aquaculture
    coeff_df = read_aquaculture_coefficients()

    # Fill NaN with 0 where necessary
    cols = [
        "FISKEBEHOLDNING_ANTALL",
        "FISKEBEHOLDNING_SNITTVEKT",
        "TAP_DOD",
        "TAP_UTKAST",
        "TAP_ROMT",
        "TAP_ANNET",
        "TELLEFEIL",
        "UTTAK_KILO",
    ]
    for col in cols:
        df[col].fillna(value=0, inplace=True)

    # Calculate biomass
    df = calculate_aquaculture_biomass(df)

    # Aggregate by month, location and species
    sum_df = aggregate_aquaculture_data(df)

    # Get biomass for previous month
    sum_df["biomass_prev_kg"] = sum_df.apply(
        get_aquaculture_biomass_previous_month, args=(sum_df,), axis=1
    )

    # Get productivity for each month
    fcr = coeff_df.loc["fcr"]["value"]
    sum_df["prod_kg"] = sum_df.apply(
        calculate_aquaculture_productivity, args=(fcr,), axis=1
    )

    # Calculate nutrient losses
    sum_df = calculate_aquaculture_nutrient_losses(sum_df, coeff_df)

    # Get just the data for species of interest
    sum_df.reset_index(inplace=True)
    sum_df = sum_df.query("FISKEARTID in @species_ids")

    # Aggregate by location
    agg_df = sum_df.groupby(by=["site_id", "sector", "type"])
    sum_df = agg_df.sum(numeric_only=True)[["TOTN_kg", "TOTP_kg", "TOC_kg"]]

    # Distribute Cu according to P production
    if cu_tonnes:
        sum_df = distribute_aquaculture_copper(sum_df, cu_tonnes)

    # Subdivide TOTN and TOTP
    sum_df.reset_index(inplace=True)
    sum_df = subdivide_point_source_n_and_p(sum_df, "Aquaculture", "TOTN_kg", "TOTP_kg")
    id_cols = ["site_id", "sector", "type"]
    sum_df = sum_df.melt(id_vars=id_cols)

    # Convert to db par_ids
    sum_df = map_par_names_to_db_ids(sum_df, eng, par_col="variable")
    sum_df["year"] = year
    sum_df = sum_df[["site_id", "in_par_id", "year", "value"]]

    return sum_df


def read_aquaculture_coefficients():
    """Read coefficients for aquaculture calculations.

    Args
        None.

    Returns
        Dataframe.
    """
    url = r"https://raw.githubusercontent.com/NIVANorge/teotil3/main/data/aquaculture_productivity_coefficients.csv"
    coeff_df = pd.read_csv(url, index_col=0)

    return coeff_df


def calculate_aquaculture_biomass(df):
    """Calculate biomass.

    Args
        df: Dataframe of aquaculture production

    Returns
        Dataframe.
    """
    df["biomass_kg"] = (
        (
            (df["FISKEBEHOLDNING_ANTALL"] * df["FISKEBEHOLDNING_SNITTVEKT"])
            + (df["TAP_DOD"] * df["FISKEBEHOLDNING_SNITTVEKT"])
            + (df["TAP_UTKAST"] * df["FISKEBEHOLDNING_SNITTVEKT"])
            + (df["TAP_ROMT"] * df["FISKEBEHOLDNING_SNITTVEKT"])
            + (df["TAP_ANNET"] * df["FISKEBEHOLDNING_SNITTVEKT"])
            + (df["TELLEFEIL"] * df["FISKEBEHOLDNING_SNITTVEKT"])
        )
        / 1000.0
    ) + df["UTTAK_KILO"]

    return df


def aggregate_aquaculture_data(df):
    """Aggregate by month, location and species.

    Args
        df: Dataframe of aquaculture production

    Returns
        Dataframe.
    """
    agg_df = df.groupby(by=["site_id", "sector", "type", "MAANED", "FISKEARTID"])
    sum_df = agg_df.sum(numeric_only=True)[["FORFORBRUK_KILO", "biomass_kg"]]

    return sum_df


def calculate_aquaculture_nutrient_losses(df, coeff_df):
    """Calculate aquaculture nutrient losses.

    Args
        df: Dataframe of aquaculture production
        coeff_df: Dataframe from 'read_aquaculture_coefficients'

    Returns
        New columns for 'TOTN_kg', 'TOTP_kg' and 'TOC_kg' are added to 'df'.
    """
    k_feed_n = coeff_df.loc["k_feed_n"]["value"]
    k_feed_p = coeff_df.loc["k_feed_p"]["value"]
    k_feed_c = coeff_df.loc["k_feed_c"]["value"]
    k_prod_n = coeff_df.loc["k_prod_n"]["value"]
    k_prod_p = coeff_df.loc["k_prod_p"]["value"]
    fcr = coeff_df.loc["fcr"]["value"]

    df["TOTN_kg"] = df.apply(
        calculate_aquaculture_n_and_p_loss,
        args=(k_feed_n, k_prod_n, fcr),
        axis=1,
    )
    df["TOTP_kg"] = df.apply(
        calculate_aquaculture_n_and_p_loss,
        args=(k_feed_p, k_prod_p, fcr),
        axis=1,
    )
    df["TOC_kg"] = df.apply(
        calculate_aquaculture_toc_loss,
        args=(k_feed_c, fcr),
        axis=1,
    )
    return df


def distribute_aquaculture_copper(df, cu_tonnes):
    """Distribute aquaculture Cu according to P production.

    Args
        df: Dataframe of aquaculture production
        cu_tonnes: Float. Annual copper usage in tonnes

    Returns
        A new column named 'Cu_kg' is added to 'df'.

    Raises
        TypeError if 'cu_tonnes' is not a number.
    """
    if not isinstance(cu_tonnes, (int, float)):
        raise TypeError("'cu_tonnes' must be a number.")

    cu_loss_tonnes = 0.85 * cu_tonnes
    print(
        f"The total annual copper lost to water from aquaculture is {cu_loss_tonnes:.1f} tonnes."
    )
    df["Cu_kg"] = 1000 * cu_loss_tonnes * df["TOTP_kg"] / df["TOTP_kg"].sum()

    return df


def map_par_names_to_db_ids(df, eng, par_col="variable"):
    """Takes a dataframe in 'long' format and maps parameters in the column named 'par_col' to
    database input parameter IDs. Parameters in 'par_col' must be named {par}_{unit}.

    Args
        df: Dataframe in 'long' format.
        eng: Obj. Active database connection object
        par_col: Str. Column in 'df'

    Returns
        Dataframe. New column named 'in_par_id' is added to 'df'.

    Raises
        ValueError if 'par_col' not in 'df'.
    """
    if par_col not in df.columns:
        raise ValueError(f"Column '{par_col}' not found in 'df'.")

    sql = text(
        """SELECT in_par_id,
             CONCAT_WS('_', name, unit) AS par_unit
           FROM teotil3.input_param_definitions
        """
    )
    input_par_df = pd.read_sql(sql, eng)
    par_map = input_par_df.set_index("par_unit").to_dict()["in_par_id"]

    # Get just vars of interest
    db_pars = list(par_map.keys())
    df = df.query(f"{par_col} in @db_pars").copy()

    df["in_par_id"] = df[par_col].map(par_map)

    return df


def get_aquaculture_biomass_previous_month(row, df):
    """Returns fish farm biomass for the previous month. If month = 1, or if data for the previous
    month are not available, returns 0.

    Args
        row: Obj. Dataframe row
        df: Obj. Original dataframe containing data for other months

    Returns
        Float. Biomass for previous month in kg.
    """
    # Get row props from multi-index
    loc, sec, typ, mon, spec = row.name

    if mon == 1:
        return 0
    else:
        try:
            # Returns a KeyError if data for (mon - 1) do not exist
            return df.loc[(loc, mon - 1, spec)]["biomass_kg"]

        except KeyError:
            return 0


def calculate_aquaculture_productivity(row, fcr):
    """Calculate fish farm productivity based on change in biomass compared to the previous month.
    If biomass has increased, the productivity is the increase in kg. If biomass has decreased, or
    if either of the biomasses are zero, the Feed Conversion Ratio (FCR) is used instead - see
    Section 3.3.6 here:

    https://niva.brage.unit.no/niva-xmlui/bitstream/handle/11250/2985726/7726-2022+high.pdf?sequence=1#page=29

    Args
        row: Obj. Dataframe row
        fcr: Float. Feed Conversion Ratio to use when biomass figures are not available

    Returns
        Float. Productivity for month in kg.

    Raises
        TyperError if 'fcr' is not a number.
    """
    if not isinstance(fcr, (int, float)):
        raise TypeError("'fcr' must be a number.")

    if (
        (row["biomass_kg"] == 0)
        or (row["biomass_prev_kg"] == 0)
        or (row["biomass_kg"] < row["biomass_prev_kg"])
    ):
        return row["FORFORBRUK_KILO"] / fcr

    else:
        return row["biomass_kg"] - row["biomass_prev_kg"]


def calculate_aquaculture_n_and_p_loss(row, k_feed, k_prod, fcr):
    """Calculate the balance of "nutrients in" versus "nutrients out" for aquaculture. For any
    parameter, X, (e.g. TOTN or TOTP) 'k_feed' is the proportion of X in the feed, and 'k_prod'
    is the proportion of X in exported fish. The default values used by TEOTIL are here:

    https://github.com/NIVANorge/teotil3/blob/main/data/aquaculture_productivity_coefficients.csv

    The balance is calculated as

        losses = inputs - outputs
               = (k_feed * feed_use) - (k_prod * productivity)

    If productivity data are not available, or if the apparent nutrient balance is negative, the
    Feed Conversion Ratio is used to estimate productivity.

    Args
        row: Obj. Dataframe row being processed
        k_feed: Float. Proportion of X in feed
        k_prod: Float. Proportion of X in exported fish
        fcr: Float. Feed Conversion Ratio to use when productivity figures are not available

    Returns
        Float. Nutrients lost in kg.

    Raises
        TypeError if 'fcr', 'k_feed' or 'k_prod' are not numbers.
    """
    if not isinstance(fcr, (int, float)):
        raise TypeError("'fcr' must be a number.")
    if not isinstance(k_feed, (int, float)):
        raise TypeError("'k_feed' must be a number.")
    if not isinstance(k_prod, (int, float)):
        raise TypeError("'k_prod' must be a number.")

    if row["FORFORBRUK_KILO"] == 0:
        return 0

    elif row["prod_kg"] == 0:
        return (k_feed * row["FORFORBRUK_KILO"]) - (
            k_prod * row["FORFORBRUK_KILO"] / fcr
        )

    elif ((k_feed * row["FORFORBRUK_KILO"]) - (k_prod * row["prod_kg"])) < 0:
        return (k_feed * row["FORFORBRUK_KILO"]) - (
            k_prod * row["FORFORBRUK_KILO"] / fcr
        )

    else:
        return (k_feed * row["FORFORBRUK_KILO"]) - (k_prod * row["prod_kg"])


def calculate_aquaculture_toc_loss(row, k_feed, fcr):
    """Estimate TOC losses from aquaculture based on feed use. The function implements the method
    described in section 6.2.4 here:

    https://niva.brage.unit.no/niva-xmlui/bitstream/handle/11250/2985726/7726-2022+high.pdf?sequence=1#page=43

    Args
         row: Obj. Dataframe row being processed
         k_feed: Float. Propoertion of TOC in feed
         fcr: Float. Feed Conversion Ratio to use when feed use figures are not available

    Returns
        Float. TOC loss in kg.

    Raises
        TypeError if 'fcr' or 'k_feed' are not numbers.
    """
    if not isinstance(k_feed, (int, float)):
        raise TypeError("'k_feed' must be a number.")
    if not isinstance(fcr, (int, float)):
        raise TypeError("'fcr' must be a number.")

    if (row["FORFORBRUK_KILO"] == 0) and (row["prod_kg"] == 0):
        return 0
    elif (row["FORFORBRUK_KILO"] == 0) and (row["prod_kg"] > 0):
        # Use FCR to estimate actual feed use
        feed_use = fcr * row["prod_kg"]
        return k_feed * feed_use * (0.97 * 0.3 + 0.03)
    else:
        # Use actual feed use values
        return k_feed * row["FORFORBRUK_KILO"] * (0.97 * 0.3 + 0.03)


def get_annual_copper_usage_aquaculture(year):
    """Get the total annual copper usage for the sepcified year from a file hosted on GitHub.
    Values are provided by Miljødirektoratet.

    Args
        year: Int. Year of interest

    Returns
        Float. Total copper usage in Aquaculture in tonnes. TEOTIL assumes 85% of this is
        lost to the environment

    Raises
        TypeError if 'year' is not an integer.
    """
    if not isinstance(year, int):
        raise TypeError("'year' must be an integer.")
    url = r"https://raw.githubusercontent.com/NIVANorge/teotil3/main/data/aquaculture_annual_copper_usage.csv"
    df = pd.read_csv(url, index_col=0)
    cu_tonnes = float(df.loc[year]["tot_cu_tonnes"])

    return cu_tonnes


def utm_to_wgs84_dd(utm_df, zone="utm_zone", east="utm_east", north="utm_north"):
    """Converts UTM co-ordinates to WGS84 decimal degrees, allowing for each row in 'utm_df' to
    have a different UTM Zone. (Note that if all rows have the same zone, this implementation is
    slow because it processes each row individually).

    Args
        utm_df: Dataframe containing UTM co-ords
        zone: Str. Column defining UTM zone
        east: Str. Column defining UTM Easting
        north: Str. Column defining UTM Northing

    Returns
        Copy of utm_df with 'lat' and 'lon' columns added.

    Raises
        ValueError: If any of the input column names are not strings.
    """
    # Check that input arguments are strings
    if not all(isinstance(arg, str) for arg in [zone, east, north]):
        raise ValueError("All input column names must be strings.")

    # Copy utm_df
    df = utm_df.copy()

    # Containers for data
    lats = []
    lons = []

    # Loop over df
    for idx, row in df.iterrows():
        # Only convert if UTM co-ords are available
        if pd.isnull(row[east]) or pd.isnull(row[north]) or pd.isnull(row[zone]):
            lats.append(np.nan)
            lons.append(np.nan)
        else:
            # Build projection
            p = pyproj.Proj(proj="utm", zone=row[zone], ellps="WGS84")

            # Convert
            lon, lat = p(row[east], row[north], inverse=True)
            lats.append(lat)
            lons.append(lon)

    # Add to df
    df["lat"] = lats
    df["lon"] = lons

    return df


def patch_coordinates(df, col_list, source="site", target="outlet"):
    """Patch data gaps in the target co-ordinates with the source co-ordinates.

    Args
        df: DataFrame containing co-ordinates.
        col_list: List of str. Co-ordinates to patch. For each name in 'col_list', the function
            looks for columns in 'df' named '{source}_{col}' and '{target}_{col}'. NaNs in the
            '{target}' column are filled with values from the '{source}' column.
        source: Str. Prefix for the source columns. Defaults to 'site'.
        target: Str. Prefix of the target columns. Defaults to 'outlet'.

    Returns
        DataFrame with target co-ordinates patched.

    Raises
        ValueError: If '{source}_{col}' and '{target}_{col}' columns are not present in 'df'.
    """
    for col in col_list:
        source_col = f"{source}_{col}"
        target_col = f"{target}_{col}"

        if source_col not in df.columns:
            raise ValueError(f"'{source_col}' not found in columns of df.")
        if target_col not in df.columns:
            raise ValueError(f"'{target_col}' not found in columns of df.")

        df[target_col].fillna(df[source_col], inplace=True)

    return df


def filter_valid_utm_zones(
    df, source_type, zone_cols=["site_zone", "outlet_zone"], zone_min=31, zone_max=36
):
    """Filters 'df' to only keep rows where columns in 'zone_cols' contain integers between
    'zone_min' and 'zone_max' (inclusive).

    Args
        df: Dataframe containing UTM data
        source_type: Str. One of ['Large wastewater', 'Miljøgifter']
        zone_cols: List of str. Columns in 'df' containing UTM zones to filter
        zone_min: Int. Minimum valid UTM zone. Default 31
        zone_max: Int. Maximum valid UTM zone. Default 36

    Returns
        Dataframe. Subset of rows in 'df' with valid UTM zones.

    Raises
        ValueError if columns in 'zone_cols' not found in 'df'.
        ValueError if 'source_type' not in
            ['Large wastewater', 'Miljøgifter']
        TypeError if 'zone_min' or 'zone_max' are not integers.
    """
    source_types = ["Large wastewater", "Miljøgifter"]
    if source_type not in source_types:
        raise ValueError(f"'{source_type}' must be one of {source_types}.")

    if not isinstance(zone_min, int) or not isinstance(zone_max, int):
        raise TypeError("'zone_min' and 'zone_max' must be integers.")

    for col in zone_cols:
        if col not in df.columns:
            raise ValueError(f"'{col}' not found in columns of df.")

        if df[col].min() < zone_min or df[col].max() > zone_max:
            print(
                f"'{col}' column in {source_type} contains values outside valid range [{zone_min}, {zone_max}]. These will be dropped."
            )
            df = df.query(f"@zone_min <= {loc}_zone <= @zone_max")

    return df


def read_raw_large_wastewater_data(data_fold, year):
    """Reads the raw, gap-filled data for TOTN, TOTP, BOF5 and KOF from "large" (>50 p.e.)
    wastewater treatment sites provided by SSB. Note that this dataset includes some data that is
    duplicated in the "miljøgifter" dataset.

    Args
        data_fold: Str. Folder containg raw data files, with the file structure as described above
        year: Int. Year being processed

    Returns
        Geodataframe in 'wide' format. A point geodataframe in EPSG 25833.

    Raises
        TypeError if 'data_fold' is not a string.
        TypeError if 'year' is not an integer.
    """
    if not isinstance(data_fold, str):
        raise TypeError("'data_fold' must be a valid file path.")
    if not isinstance(year, int):
        raise TypeError("'year' must be an integer.")

    column_mappings = {
        "ANLEGGSNR": "site_id",
        "ANLEGGSNAVN": "name",
        "Sone": "site_zone",
        "UTM_E": "site_east",
        "UTM_N": "site_north",
        "Sone_Utslipp": "outlet_zone",
        "UTM_E_Utslipp": "outlet_east",
        "UTM_N_Utslipp": "outlet_north",
        "MENGDE_P_UT_kg": "TOTP_kg",
        "MENGDE_N_UT_kg": "TOTN_kg",
    }
    val_cols = ["TOTP_kg", "TOTN_kg", "BOF5_kg", "KOF_kg"]

    # Read site locs and data for TOTN and TOTP
    stan_path = os.path.join(data_fold, f"avlop_stor_anlegg_{year}_raw.xlsx")
    df = pd.read_excel(stan_path, sheet_name=f"store_anlegg_{year}")
    df.dropna(how="all", inplace=True)
    df["sector"] = "Large wastewater"
    df["year"] = year
    df.rename(column_mappings, axis="columns", inplace=True)
    df.drop_duplicates(
        subset=[
            "site_id",
            "name",
            "site_zone",
            "site_east",
            "site_north",
        ],
        inplace=True,
    )

    # If the outlet co-ords aren't known, use the site co-ords instead
    df = patch_coordinates(
        df, ["zone", "east", "north"], source="site", target="outlet"
    )

    # Filter to valid UTM zones
    df = filter_valid_utm_zones(df, "Large wastewater")

    # Convert mixed UTM => lat/lon => EPSG 25833
    geom_df = pd.DataFrame()
    for loc in ["site", "outlet"]:
        # Convert UTM Zone to Pandas' nullable integer data type
        # (because proj. complains about float UTM zones)
        df[f"{loc}_zone"] = df[f"{loc}_zone"].astype(pd.Int64Dtype())
        df = utm_to_wgs84_dd(df, f"{loc}_zone", f"{loc}_east", f"{loc}_north")
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["lon"], df["lat"], crs="epsg:4326"),
        )
        gdf = gdf.to_crs("epsg:25833")
        df.drop(
            ["lon", "lat", f"{loc}_zone", f"{loc}_east", f"{loc}_north"],
            axis="columns",
            inplace=True,
        )
        gdf.rename({"geometry": f"{loc}_geom"}, axis="columns", inplace=True)
        geom_df = pd.concat([geom_df, gdf[f"{loc}_geom"]], axis=1)
    gdf = gpd.GeoDataFrame(pd.concat([df, geom_df], axis=1))
    gdf.set_geometry("outlet_geom", inplace=True)

    # Join treatment types, BOF5 and KOF
    types_path = os.path.join(
        data_fold, f"avlop_stor_anlegg_{year}_treatment_types_bof_kof.xlsx"
    )
    typ_df = pd.read_excel(types_path, sheet_name="data")
    typ_df = typ_df[["ANLEGGSNR", "RENSPRINS", "utslipp_BOF5", "utslipp_KOF"]]
    typ_df.columns = ["site_id", "type", "BOF5_kg", "KOF_kg"]
    typ_df.dropna(how="all", inplace=True)
    typ_df["type"].replace({"?": "Annen rensing"}, inplace=True)
    gdf = gdf.merge(typ_df, how="left", on="site_id")
    gdf["type"].fillna("Annen rensing", inplace=True)
    gdf["BOF5_kg"].fillna(0, inplace=True)
    gdf["KOF_kg"].fillna(0, inplace=True)
    gdf.reset_index(drop=True, inplace=True)
    id_cols = ["site_id", "name", "sector", "type", "year", "site_geom", "outlet_geom"]
    gdf = gdf[id_cols + val_cols]

    return gdf


def read_raw_miljogifter_data(data_fold, year):
    """Reads the raw, not-gap-filled data for "large" (>50 p.e.) wastewater treatment
    sites provided by SSB. Note that the 'miljøgifter' dataset includes some data
    that is duplicated in the "store anlegg" dataset.

    Args
        data_fold: Str. Folder containg raw data files, with the file structure as
                   described above
        year:      Int. Year being processed

    Returns
        Geodataframe in 'wide' format. A point geodataframe in EPSG 25833.

    Raises
        TypeError if 'data_fold' is not a string.
        TypeError if 'year' is not an integer.
    """
    if not isinstance(data_fold, str):
        raise TypeError("'data_fold' must be a valid file path.")
    if not isinstance(year, int):
        raise TypeError("'year' must be an integer.")

    column_mappings = {
        "ANLEGGSNR": "site_id",
        "ANLEGGSNAVN": "name",
        "SONEBELTE": "site_zone",
        "UTMOST": "site_east",
        "UTMNORD": "site_north",
        "RESIP2": "outlet_zone",
        "RESIP3": "outlet_east",
        "RESIP4": "outlet_north",
    }
    val_cols = [
        "KONSMENGDSS10",
        "MILJOGIFTAS2",
        "MILJOGIFTCD2",
        "MILJOGIFTCR2",
        "MILJOGIFTCU2",
        "MILJOGIFTHG2",
        "MILJOGIFTNI2",
        "MILJOGIFTPB2",
        "MILJOGIFTZN2",
    ]
    miljo_path = os.path.join(data_fold, f"avlop_miljogifter_{year}_raw.xlsx")
    df = pd.read_excel(miljo_path, sheet_name=f"miljogifter_{year}")
    df.dropna(how="all", inplace=True)
    df["sector"] = "Large wastewater"
    df["year"] = year
    df.rename(column_mappings, axis="columns", inplace=True)
    df.drop_duplicates(
        subset=[
            "site_id",
            "name",
            "site_zone",
            "site_east",
            "site_north",
        ],
        inplace=True,
    )

    # If the outlet co-ords aren't known, use the site co-ords instead
    df = patch_coordinates(
        df, ["zone", "east", "north"], source="site", target="outlet"
    )

    # Filter to valid UTM zones
    df = filter_valid_utm_zones(df, "Miljøgifter")

    # Convert mixed UTM => lat/lon => EPSG 25833
    geom_df = pd.DataFrame()
    for loc in ["site", "outlet"]:
        # Convert UTM Zone to Pandas' nullable integer data type
        # (because proj. complains about float UTM zones)
        df[f"{loc}_zone"] = df[f"{loc}_zone"].astype(pd.Int64Dtype())
        df = utm_to_wgs84_dd(df, f"{loc}_zone", f"{loc}_east", f"{loc}_north")
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["lon"], df["lat"], crs="epsg:4326"),
        )
        gdf = gdf.to_crs("epsg:25833")
        df.drop(
            ["lon", "lat", f"{loc}_zone", f"{loc}_east", f"{loc}_north"],
            axis="columns",
            inplace=True,
        )
        gdf.rename({"geometry": f"{loc}_geom"}, axis="columns", inplace=True)
        geom_df = pd.concat([geom_df, gdf[f"{loc}_geom"]], axis=1)
    gdf = gpd.GeoDataFrame(pd.concat([df, geom_df], axis=1))
    gdf.set_geometry("outlet_geom", inplace=True)

    # Join treatment types
    types_path = os.path.join(
        data_fold, f"avlop_stor_anlegg_{year}_treatment_types_bof_kof.xlsx"
    )
    typ_df = pd.read_excel(types_path, sheet_name="data")
    typ_df = typ_df[["ANLEGGSNR", "RENSPRINS"]]
    typ_df.columns = ["site_id", "type"]
    typ_df.dropna(how="all", inplace=True)
    typ_df["type"].replace({"?": "Annen rensing"}, inplace=True)
    gdf = gdf.merge(typ_df, how="left", on="site_id")
    gdf["type"].fillna("Annen rensing", inplace=True)
    gdf.reset_index(drop=True, inplace=True)
    id_cols = ["site_id", "name", "sector", "type", "year", "site_geom", "outlet_geom"]
    gdf = gdf[id_cols + val_cols]

    # Add units
    units_dict = {par: f"{par}_kg" for par in val_cols}
    gdf.rename(units_dict, axis="columns", inplace=True)

    return gdf


def read_raw_industry_data(data_fold, year):
    """Reads the raw industry data provided by Miljødirektoratet.

    Args
        data_fold: Str. Folder containg raw data files, with the file structure as
                   described above
        year:      Int. Year being processed

    Returns
        Geodataframe in 'wide' format. A point geodataframe in EPSG 25833.

    Raises
        TypeError if 'data_fold' is not a string.
        TypeError if 'year' is not an integer.
        ValueError if the 'Komm. nett' column contains NaNs.
    """
    if not isinstance(data_fold, str):
        raise TypeError("'data_fold' must be a valid file path.")
    if not isinstance(year, int):
        raise TypeError("'year' must be an integer.")

    column_mappings = {
        "Anleggsnr": "site_id",
        "Anleggsnavn": "name",
        "Anleggsaktivitet": "type",
        "Geografisk Longitude": "site_lon",
        "Geografisk Latitude": "site_lat",
        "Lon_Utslipp": "outlet_lon",
        "Lat_Utslipp": "outlet_lat",
        "Mengde": "value",
    }
    val_cols = ["variable", "value"]
    ind_path = os.path.join(data_fold, f"industri_{year}_raw.xlsx")
    df = pd.read_excel(ind_path, sheet_name=f"industri_{year}")
    df.dropna(how="all", inplace=True)

    # Remove sites that discharge to the kommunal network (included in wastewater dataset)
    if df["Komm. nett"].isna().sum() > 0:
        raise ValueError("Column 'Komm. nett' contains NaNs.")
    df = df.query("`Komm. nett` == False")

    if len(df["År"].unique()) > 1:
        print(
            f"WARNING: The industry dataset includes values for several years. Only data for {year} will be processed."
        )
        df = df.query("`År` == @year")
    df["sector"] = "Industry"
    df["year"] = year
    df.rename(column_mappings, axis="columns", inplace=True)

    # If the outlet co-ords aren't known, use the site co-ords instead
    df = patch_coordinates(df, ["lon", "lat"], source="site", target="outlet")

    # Convert lat/lon => EPSG 25833
    geom_df = pd.DataFrame()
    for loc in ["site", "outlet"]:
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(
                df[f"{loc}_lon"], df[f"{loc}_lat"], crs="epsg:4326"
            ),
        )
        gdf = gdf.to_crs("epsg:25833")
        df.drop(
            [f"{loc}_lon", f"{loc}_lat"],
            axis="columns",
            inplace=True,
        )
        gdf.rename({"geometry": f"{loc}_geom"}, axis="columns", inplace=True)
        geom_df = pd.concat([geom_df, gdf[f"{loc}_geom"]], axis=1)
    gdf = gpd.GeoDataFrame(pd.concat([df, geom_df], axis=1))
    gdf.set_geometry("outlet_geom", inplace=True)
    gdf.reset_index(drop=True, inplace=True)
    gdf["Enhet"].replace({"tonn": "tonnes"}, inplace=True)
    gdf["variable"] = gdf["Komp.kode"] + "_" + gdf["Enhet"]

    # Ignore some pars as they overlap with estimates based on TOTN and TOTP
    ignore_par_list = ["NH3", "NH4-N", "P-ORTO"]
    gdf = gdf.query("`Komp.kode` not in @ignore_par_list")

    id_cols = ["site_id", "name", "sector", "type", "year", "site_geom", "outlet_geom"]
    gdf = gdf[id_cols + val_cols]

    # Convert to wide format
    gdf = gdf.pivot(index=id_cols, columns="variable", values="value").reset_index()

    return gdf


def read_large_wastewater_and_industry_data(data_fold, year, eng):
    """Convenience function for processing the raw "store anlegg", "miljøgifter" and "industry"
    datasets. Assumes the raw files are named and arranged as follows:

        data_fold/
        ├─ avlop_stor_anlegg_{year}_raw.xlsx
        │  ├─ store_anlegg_{year} [worksheet]
        │
        ├─ avlop_stor_anlegg_{year}_treatment_types_bof_kof.xlsx
        │  ├─ data [worksheet]
        │
        ├─ avlop_miljogifter_{year}_raw.xlsx
        │  ├─ miljogifter_{year} [worksheet]
        │
        ├─ industri_{year}_raw.xlsx
        │  ├─ industri_{year} [worksheet]

    This function reads all the raw files and combines site locations and data values into a
    geodataframe and a dataframe, respectively. Parameter names are mapped to input parameter
    IDs in the database, and duplicates are removed. Sites without coordinates are highlighted.

    Args
        data_fold: Str. Folder containg raw data files, with the file structure as described
            above
        year: Int. Year being processed
        eng: Obj. Active database connection object connected to PostGIS

    Returns
        Tuple of (geo)dataframes (loc_gdf, df). 'loc_gdf' is a point geodataframe of site
        co-ordinates in EPSG 25833; 'df' is a dataframe of discharges from each site.

    Raises
        TypeError if 'data_fold' is not a string.
        TypeError if 'year' is not an integer.
    """
    if not isinstance(data_fold, str):
        raise TypeError("'data_fold' must be a valid file path.")
    if not isinstance(year, int):
        raise TypeError("'year' must be an integer.")

    # Read raw data
    stan_gdf = read_raw_large_wastewater_data(data_fold, year)
    miljo_gdf = read_raw_miljogifter_data(data_fold, year)
    ind_gdf = read_raw_industry_data(data_fold, year)

    # Estimate TOC from BOF and KOF
    stan_gdf = estimate_toc_from_bof_kof(stan_gdf, "Large wastewater")
    ind_gdf = estimate_toc_from_bof_kof(ind_gdf, "Industry")

    # Subdivide TOTN and TOTP
    stan_gdf = subdivide_point_source_n_and_p(
        stan_gdf, "Large wastewater", "TOTN_kg", "TOTP_kg"
    )
    ind_gdf = subdivide_point_source_n_and_p(
        ind_gdf, "Industry", "N-TOT_tonnes", "P-TOT_tonnes"
    )

    # Combine and split into locations and data (in 'long' format)
    gdf = pd.concat([stan_gdf, miljo_gdf, ind_gdf], axis="rows")
    loc_cols = ["site_id", "name", "sector", "type", "year", "site_geom", "outlet_geom"]
    val_cols = ["site_id", "year"] + [col for col in gdf.columns if col not in loc_cols]
    loc_gdf = gdf[loc_cols].copy()
    df = pd.DataFrame(gdf[val_cols]).melt(id_vars=["site_id", "year"])
    df = df.query("value > 0")

    # Convert to par_ids used in database
    df = map_par_names_to_db_ids(df, eng, par_col="variable")

    # Check for duplicates and drop if necessary
    dup_df = df[df.duplicated(subset=["site_id", "variable"], keep=False)].sort_values(
        by=["site_id", "variable"]
    )
    if len(dup_df) > 0:
        print(
            f"WARNING: {len(dup_df)} duplicates identified (see below). Only the first will be kept."
        )
        print(dup_df)
        df.drop_duplicates(subset=["site_id", "variable"], keep="first", inplace=True)

    # Process locations
    loc_gdf.drop_duplicates(
        subset=["site_id", "sector", "type"], keep="first", inplace=True
    )
    if not loc_gdf["site_id"].is_unique:
        dup_df = loc_gdf[loc_gdf.duplicated(subset="site_id", keep=False)].sort_values(
            "site_id"
        )
        print(
            "The same 'site_id' appears in both the industry and wastewater datasets. "
            "This may cause problems with double counting of input parameters."
        )
        print(dup_df)
        raise ValueError("Duplicated site IDs in 'industry' and 'wastewater' datasets.")

    # Check for missing co-ords. See
    # https://geopandas.org/en/stable/docs/user_guide/missing_empty.html
    no_coords_gdf = loc_gdf[
        loc_gdf["outlet_geom"].is_empty | loc_gdf["outlet_geom"].isna()
    ][["site_id", "name"]].sort_values("site_id")
    if len(no_coords_gdf) > 0:
        print(
            f"{len(no_coords_gdf)} locations do not have outlet co-ordinates in this year's data."
        )
        print(no_coords_gdf)

    # Drop values for sites without outlet co-ords
    no_coords_ids = no_coords_gdf["site_id"].tolist()
    df = df.query("site_id not in @no_coords_ids")
    df = df[["site_id", "in_par_id", "year", "value"]]
    df.reset_index(inplace=True, drop=True)
    loc_gdf = loc_gdf.query("site_id not in @no_coords_ids")

    assert set(df["site_id"]).issubset(set(loc_gdf["site_id"]))

    return loc_gdf, df


def read_raw_small_wastewater_data(xl_path, sheet_name, year, eng):
    """
    Reads raw small wastewater data from an Excel file and processes it.

    Args
        xl_path: Str. Path to the Excel file.
        sheet_name: Str. Name of the sheet in the Excel file.
        year: Int. Year of the data.
        eng: Obj. SQL engine connected to the database

    Returns
        pd.DataFrame: Processed data.
    """
    if not isinstance(xl_path, str):
        raise TypeError("'xl_path' must be a valid file path.")
    if not isinstance(sheet_name, str):
        raise TypeError("'sheet_name' must be a string.")
    if not isinstance(year, int):
        raise TypeError("'year' must be an integer.")

    # Check admin. boundaries for 'year' are assigned in PostGIS. Before 2015, we just use the
    # boundaries for 2014
    admin_year = 2014 if year < 2015 else year
    try:
        sql = text(f"SELECT komnr_{admin_year} AS komnr FROM teotil3.regines")
        reg_kom_df = pd.read_sql(sql, eng)
    except exc.ProgrammingError:
        raise ValueError(
            f"Administrative boundaries for {year} are not linked to regines in the TEOTIL database. "
            "Please download the latest boundaries from Geonorge and update the regine dataset."
        )

    df = pd.read_excel(xl_path, sheet_name=sheet_name).dropna(how="all")
    df = df.query("year == @year").drop(columns="year")
    df.rename({"KOMMUNENR": "komnr"}, axis="columns", inplace=True)

    # Komnr. should be a 4 char string, not a float
    df["komnr"] = df["komnr"].apply(lambda x: "%04d" % x)

    # Before 2020, there was a very small kommune (1245; Sund) that is not linked to any regines
    # "spredt" for this kommune can be aggregated with that for 1246
    assert df[
        "komnr"
    ].is_unique, f"Dataset contains duplicated kommuner: {list(df['komnr'].unique())}"
    df["komnr"].replace({"1245": "1246"}, inplace=True)
    df = df.groupby("komnr").sum(numeric_only=True).reset_index()

    # Tidy
    df = restructure_small_wastewater_data(df)

    # Estimate nutrient inputs
    df = subdivide_point_source_n_and_p(df, "Small wastewater", "TOTN_kg", "TOTP_kg")
    df["KOF_kg"] = np.nan
    df = estimate_toc_from_bof_kof(df, "Small wastewater")

    # Sum totals for each type back to kommune level
    df = df.groupby("komnr").sum(numeric_only=True).reset_index()

    # Check all SSB kommuner IDs are recognised
    check_komnrs_in_ssb_data(df, reg_kom_df)

    # Restructure for database
    df = pd.melt(df, id_vars="komnr")
    df = map_par_names_to_db_ids(df, eng, par_col="variable")
    df["year"] = year
    df = df[["komnr", "in_par_id", "year", "value"]]

    return df


def restructure_small_wastewater_data(df):
    """Basic restructuring of small wastewater data."""
    df = df.melt(id_vars="komnr")
    df["sector"] = "Small wastewater"
    df["variable"] = df["variable"].str.replace("TOTP", "TOTP_kg")
    df["variable"] = df["variable"].str.replace("TOTN", "TOTN_kg")
    df["variable"] = df["variable"].str.replace("BOF5", "BOF5_kg")
    df["variable"] = df["variable"].str.replace("SS", "SS_kg")
    df[["variable", "type"]] = df["variable"].str.split("-", n=1, expand=True)

    # Ignore 'Tett tank (for alt avløpsvann)' as it is always zero (it's transported to the
    # "large" plants)
    df = df.query("type != 'Tett tank (for alt avløpsvann)'")

    df.set_index(["komnr", "variable", "sector", "type"], inplace=True)
    df = df.unstack("variable")
    df.columns = df.columns.get_level_values(1)
    df.reset_index(inplace=True)
    df.index.name = ""

    return df


def check_komnrs_in_ssb_data(df, reg_kom_df):
    """Check that the kommuner IDs in the SSB data are all present in the regine spatial dataset."""
    not_in_db = set(df["komnr"].values) - set(reg_kom_df["komnr"].values)
    if len(not_in_db) > 0:
        print(len(not_in_db), 'kommuner are not in the TEOTIL "regine" dataset.')
        print(df[df["komnr"].isin(list(not_in_db))])


def subdivide_point_source_n_and_p(gdf, sector, totn_col, totp_col):
    """Subdivide TOTN and TOTP from wastewater, industry or aquaculture. Uses typical fractions
    for different types of treatment plant, based on Christian Vogelsang's literature review.
    See the file here for details:

    https://github.com/NIVANorge/teotil3/blob/main/data/point_source_treatment_types.csv

    Args
        gdf: Geodataframe of discharges. Must include columns for site or kommune ID, treatment
            type, year, TOTN and TOTP
        sector: Str. Type of sites being processed. Must be one of
            ["Large wastewater", "Small wastewater", "Industry", "Aquaculture"]
        totn_col: Str. Name of column in 'gdf' with TOTN data
        totp_col: Str. Name of column in 'gdf' with TOTP data

    Returns
        Geodataframe. 'gdf' is returned with columns for subfractions added.

    Raises
        TypeError: If 'totn_col' or 'totp_col' is not a string.
        ValueError: If 'sector' is not one of the expected sectors, if proportions for DIN and
        TON or TPP and TDP do not sum to one, if 'gdf' contains unknown treatment types, or if
        unit is not recognised.
    """
    if not isinstance(totn_col, str):
        raise TypeError("'totn_col' must be a string.")
    if not isinstance(totp_col, str):
        raise TypeError("'totp_col' must be a string.")
    sectors = ["Large wastewater", "Small wastewater", "Industry", "Aquaculture"]
    if sector not in sectors:
        raise ValueError(f"{sector} must be one of {sectors}.")

    url = r"https://raw.githubusercontent.com/NIVANorge/teotil3/main/data/point_source_treatment_types.csv"
    prop_df = pd.read_csv(url)
    prop_df = prop_df.query("sector == @sector")

    if not (prop_df["prop_din"] + prop_df["prop_ton"] == 1).all():
        raise ValueError("Proportions for DIN and TON do not sum to one.")
    if not (prop_df["prop_tpp"] + prop_df["prop_tdp"] == 1).all():
        raise ValueError("Proportions for TPP and TDP do not sum to one.")
    if not set(gdf["type"]).issubset(set(prop_df["type"])):
        raise ValueError(
            f"'gdf' contains unknown treatment types: {set(gdf['type']) - set(prop_df['type'])}."
        )

    gdf = gdf.merge(prop_df, how="left", on=["sector", "type"])

    fracs = ["DIN", "TON", "TPP", "TDP"]
    for frac in fracs:
        tot_col = totn_col if frac[-1] == "N" else totp_col
        unit = tot_col.split("_")[-1]
        unit_factor = 1000 if unit == "tonnes" else 1 if unit == "kg" else None
        if unit_factor is None:
            raise ValueError("Unit not recognised.")
        gdf[f"{frac}_kg"] = unit_factor * gdf[tot_col] * gdf[f"prop_{frac.lower()}"]

    # Delete unnecessary cols
    for col in prop_df.columns:
        if col not in ["sector", "type"]:
            del gdf[col]

    return gdf


def estimate_toc_from_bof_kof(gdf, sector):
    """Estimate TOC from KOF and/or BOF. Applies to the wastewater and industry datasets.
    For each treatment type, uses the best relationship identified by Christian
    Vogelsang to predict from TOC from either KOF or BOF. All relationships have the form

        TOC = k1 * KOF^k2 + k3 or TOC = b1 * BOF^b2 + b3

    where the k_i and b_i are taken from the CSV here (based on Chrtistian's literature review)

    https://github.com/NIVANorge/teotil3/blob/main/data/point_source_treatment_types.csv

    Args
        gdf: (Geo)dataframe of discharges. Must include columns for site/kommune, treatment type
            and BOF/KOF
        sector: Str. Type of sites being processed. Must be one of
            ["Large wastewater", "Small wastewater", "Industry"]

    Returns
        Geodataframe in same format as 'gdf', but with TOC added.

    Raises
        ValueError: If the sector is not one of
            ["Large wastewater", "Small wastewater", "Industry"]
        ValueError: If 'gdf' contains unknown treatment types
    """
    sectors = ["Large wastewater", "Small wastewater", "Industry"]
    if sector not in sectors:
        raise ValueError(f"{sector} must be one of {sectors}.")

    url = r"https://raw.githubusercontent.com/NIVANorge/teotil3/main/data/point_source_treatment_types.csv"
    prop_df = pd.read_csv(url)
    prop_df = prop_df.query("sector == @sector")

    if set(gdf["type"]).difference(set(prop_df["type"])):
        raise ValueError(
            f"'gdf' contains unknown treatment types: {set(gdf['type']) - set(prop_df['type'])}."
        )

    bof_col = "BOF5"
    kof_col = "KOF"
    unit = "tonnes" if sector == "Industry" else "kg"

    gdf = gdf.merge(prop_df, how="left", on=["sector", "type"])

    # Estimate TOC. Use KOF if available, otherwise BOF
    gdf[f"TOC_BOF_{unit}"] = gdf["bof_fac"] * gdf[f"{bof_col}_{unit}"]
    gdf[f"TOC_KOF_{unit}"] = gdf["kof_fac"] * gdf[f"{kof_col}_{unit}"]
    if f"TOC_{unit}" in gdf.columns:
        gdf[f"TOC_{unit}"] = gdf[f"TOC_{unit}"].combine_first(gdf[f"TOC_KOF_{unit}"])
    else:
        gdf[f"TOC_{unit}"] = gdf[f"TOC_KOF_{unit}"]
    gdf[f"TOC_{unit}"] = gdf[f"TOC_{unit}"].combine_first(gdf[f"TOC_BOF_{unit}"])
    del gdf[f"TOC_BOF_{unit}"], gdf[f"TOC_KOF_{unit}"]

    # Delete unnecessary cols
    gdf.drop(
        columns=[col for col in prop_df.columns if col not in ["sector", "type"]],
        inplace=True,
    )

    return gdf


def read_raw_agri_data(year, data_fold):
    """Read the raw agricultural data from a NIBIO template. Templates must be named
    "nibio_agri_data_{year}.xlsx" and have worksheets named 'Annual loss' and 'Risk loss'.

    Args
        year: Int. Year of interest.
        data_fold: Str. Path to folder containing templates.

    Returns
        Dataframe of agricultural data.

    Raises
        ValueError if estimated losses are nagative.
        ValueError if column names cannot be parsed correctly.
    """
    pars = ["totn", "din", "ton", "totp", "tdp", "tpp", "ss", "toc"]
    srcs = ["agriculture", "agriculture_background"]
    loss_types = ["annual", "risk"]

    valid_cols = ["loss_type", "year", "regine"] + [
        f"{src}_{par}_kg" for src, par in itertools.product(srcs, pars)
    ]

    xl_path = os.path.join(data_fold, f"nibio_agri_data_{year}.xlsx")
    df_list = []
    for loss_type in loss_types:
        df = pd.read_excel(
            xl_path, sheet_name=f"{loss_type.capitalize()} loss", header=[0, 1]
        )
        df.columns = df.columns.map("_".join).str.replace(" ", "")
        names_dict = get_agri_names_dict(loss_type)
        df.rename(columns=names_dict, inplace=True)
        convert_agri_units(df)
        calculate_derived_agri_parameters(df)

        if (numeric_cols := df.select_dtypes(include=np.number)).lt(0).any().any():
            raise ValueError(
                f"The template for {year} contains negative losses for {numeric_cols.columns.tolist()}."
            )

        if df.isna().sum().sum() > 0:
            print(
                f"WARNING: Template for {year} contains NaNs. These will be filled with zero."
            )
            df.fillna(0, inplace=True)

        df["year"] = year
        df["loss_type"] = loss_type

        if set(valid_cols) != set(df.columns):
            raise ValueError("Template contains invalid columns.")

        df = df[valid_cols]
        df_list.append(df)

    df = pd.concat(df_list, axis="rows")

    return df


def get_agri_names_dict(loss_type):
    """Generates a dictionary for renaming the columns of the agriculture dataframe.

    Args
        loss_type: The type of loss. Must be either 'annual' or 'risk'.

    Returns
        A dictionary mapping old to new column names.

    Raises
        Value error if 'loss_type' not in ['annual', 'risk'].
    """
    if loss_type not in ["annual", "risk"]:
        raise ValueError("'loss_type' must be either 'annual' or 'risk'.")

    names_dict = {
        "Regineunit_Unnamed:0_level_1": "regine",
        f"TNnetto{loss_type}loss_kg": "agriculture_totn_kg",
        f"TNbakgrunnsavr.{loss_type}loss_kg": "agriculture_background_totn_kg",
        f"NO3-Nnetto{loss_type}loss_kg": "agriculture_din_kg",
        f"NO3-Nbakgrunnsavr.{loss_type}loss_kg": "agriculture_background_din_kg",
        f"TPnetto{loss_type}loss_kg": "agriculture_totp_kg",
        f"TPbakgrunnsavr.{loss_type}loss_kg": "agriculture_background_totp_kg",
        f"PO4-Pnetto{loss_type}loss_kg": "agriculture_tdp_kg",
        f"PO4-Pbakgrunnsavr.{loss_type}loss_kg": "agriculture_background_tdp_kg",
        f"SSnetto{loss_type}loss_Tonn": "agriculture_ss_tonnes",
        f"SSbakgrunnsavr.{loss_type}loss_Tonn": "agriculture_background_ss_tonnes",
        f"TOCnetto{loss_type}loss_Tonn": "agriculture_toc_tonnes",
        f"TOCbakgrunnsavr.{loss_type}loss_Tonn": "agriculture_background_toc_tonnes",
    }

    return names_dict


def convert_agri_units(df):
    """Converts the units of the agriculture dataframe in-place.

    Args
        df: The dataFrame to convert.

    Returns
        None. 'df' is updated in-place.
    """
    for col in df.columns:
        parts = col.split("_")
        if parts[-1] == "tonnes":
            df["_".join(parts[:-1] + ["kg"])] = df[col] * 1000
            del df[col]


def calculate_derived_agri_parameters(df):
    """Calculate "derived" agricultural (TON and TPP) as the difference between the total
    and the reported sub-fractions. The dataFrame is modified in-place.

    Args
        df: The dataFrame to process.

    Returns
        None. 'df' is updated in-place.
    """
    df["agriculture_ton_kg"] = df["agriculture_totn_kg"] - df["agriculture_din_kg"]
    df["agriculture_background_ton_kg"] = (
        df["agriculture_background_totn_kg"] - df["agriculture_background_din_kg"]
    )
    df["agriculture_tpp_kg"] = df["agriculture_totp_kg"] - df["agriculture_tdp_kg"]
    df["agriculture_background_tpp_kg"] = (
        df["agriculture_background_totp_kg"] - df["agriculture_background_tdp_kg"]
    )
