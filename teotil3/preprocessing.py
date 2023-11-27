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
        layer_name:       Str. Name of regine layer in geodatabase

    Returns
        Geodataframe with columns 'regine', 'vassom', 'a_cat_poly_km2' and 'upstr_a_km2'.
    """
    reg_gdf = gpd.read_file(geodatabase_path, driver="fileGDB", layer=layer_name)
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

    # Get cols of interest
    reg_cols = [
        "regine",
        "vassom",
        "a_cat_poly_km2",
        "upstr_a_km2",
        "geometry",
    ]
    reg_gdf = reg_gdf[reg_cols]
    reg_gdf.sort_values("regine", inplace=True)
    reg_gdf.reset_index(inplace=True, drop=True)

    return reg_gdf


def summarise_regine_hydrology(reg_gdf, ro_grid_path, all_touched=True):
    """Summarise basic, regine-level hydrology information based on NVE's latest "runoff
    normal" for 1991 to 2020.

    Args
        reg_gdf:      Geodataframe of regines. Must contain column named 'a_cat_land_km2'
                      representing the land area in each regine
        ro_grid_path: Str. Path to 1 km2 runoff grid for 1991-2020 from NVE
        all_touched:  Bool. Default True. Defines the rasterisation strategy. See
                      https://pythonhosted.org/rasterstats/manual.html#rasterization-strategy

    Returns
        Geodataframe. Copy of the original geodataframe with columns 'runoff_mm/yr' and
        'q_cat_m3/s' added.
    """
    reg_gdf = reg_gdf.copy()

    stats = ["mean"]

    if reg_gdf.crs.to_epsg() != 25833:
        stats_gdf = reg_gdf.to_crs("epsg:25833")
    else:
        stats_gdf = reg_gdf.copy()

    df = pd.DataFrame(
        zonal_stats(
            vectors=stats_gdf,
            raster=ro_grid_path,
            stats=stats,
            all_touched=all_touched,
        )
    )

    assert len(reg_gdf) == len(df)

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
        reg_gdf:    Geodataframe of regine boundaries from NVE. Must contain
                    a column named 'regine' with the regine IDs
        admin_gpkg: Str. Path to geopackage containing administrative data
        admin_year: Int. Year for administrative boundaries read from
                    'admin_gpkg'

    Returns
        Geodataframe. A copy of the original geodataframe, but with new
        columns named 'fylnr' and 'komnr' added. Regines that cannot be
        assigned to administrative units (e.g. because they lie over the
        border in Sweden or Finland) are assigned a value of -1.
    """
    reg_gdf = reg_gdf.copy()
    for admin_unit in ["kommuner", "fylker"]:
        print("Processing", admin_unit)

        col_name = admin_unit[:3] + "nr"
        adm_gdf = gpd.read_file(
            admin_gpkg, layer=f"{admin_unit}{admin_year}", driver="GPKG"
        )
        adm_gdf = adm_gdf[[col_name, "geometry"]]

        # Intersect
        int_gdf = gpd.overlay(reg_gdf, adm_gdf, how="intersection", keep_geom_type=True)
        print("   ", len(adm_gdf), "kommuner.")
        print("   ", len(reg_gdf), "regines.")
        print("   ", len(int_gdf), "intersected polygons.")

        # Find dominant admin unit for each regine
        int_gdf["area_km2"] = int_gdf.to_crs({"proj": "cea"})["geometry"].area / 1e6
        int_gdf = int_gdf[["regine", col_name, "area_km2"]]
        int_gdf.sort_values("area_km2", inplace=True)
        int_gdf.drop_duplicates("regine", keep="last", inplace=True)
        del int_gdf["area_km2"]
        reg_gdf = pd.merge(reg_gdf, int_gdf, how="left", on="regine")
        reg_gdf[col_name].fillna("-1", inplace=True)

        print("   ", len(int_gdf), f"regines assigned to {admin_unit}.")

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
        reg_gdf:   Geodataframe of regine boundaries from NVE. Must contain
                   a column named 'vassom' storing the vassdragsområde for
                   each regine
        ospar_csv: Str. Default is a data table hosted on GitHub. Path to CSV
                   mapping vassdragsområder to OSPAR regions

    Returns
        Geodataframe. A copy of the original geodataframe with a new column
        named 'ospar_region' appended.
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
    """Calculate land cover proportions for each regine based on NIBIO's AR50
    dataset.

    Args
        reg_gdf:        Geodataframe of regine boundaries from NVE. Must
                        contain a column named 'regine' with the regine IDs
        ar50_gdf:       Geodataframe of AR50 data from NIBIO. Must contain a
                        column named 'artype'
        land_class_csv: Str. Default is a data table hosted on GitHub. Path
                        to CSV mapping AR50 classes to those used by TEOTIL

    Returns
        Geodataframe with additional columns showing the area of each class
        in 'land_class_csv' in each regine.
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
    """Calculate the total area of lakes in each regine based on NVE's innsjø
    database.

    Args
        reg_gdf:  Geodataframe of regine boundaries from NVE. Must contain a
                  column named 'regine' with the regine IDs
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
    cols = reg_gdf.columns.tolist()
    cols.remove("geometry")
    cols.append("geometry")
    reg_gdf = reg_gdf[cols]

    return reg_gdf


def transmission_sigma_constant(tau, sigma):
    """Estimate lake transmission from water residence time using a "basic"
    Vollenweider model:

        T = 1 / (1 + sigma * tau)

    Args
        tau:   Array-like. Lake water residence times in years
        sigma: Float. First-order rate constant for removal processes (units
               per year)

    Returns
        Array of transmission factors.
    """
    return 1 / (1 + (sigma * tau))


def transmission_sigma_from_tau(tau, k, p):
    """Estimate lake transmission from water residence time using a model
    where sigma is a function of tau.

        T = 1 / (1 + k * tau ** p)

    Args
        tau: Array-like. Lake water residence times in years
        k:   Float. Model parameter
        p:   Float. Model parameter

    Returns
        Array of transmission factors.
    """
    return 1 / (1 + (k * (tau**p)))


def transmission_sigma_from_depth(H, s):
    """Estimate lake transmission from water residence time using a model
    where sigma is a function of mean lake depth.

        T = 1 / (1 + (s / H))

    Args
        H: Array-like. Lake hydraulic load (m/year)
        s: Float. Apparent settling velocity (m/year)

    Returns
        Array of transmission factors.
    """
    return 1 / (1 + (s / H))


def calculate_lake_retention_vollenweider(df, par_name, params):
    """Calculate retention and transmission factors for individual lakes
    according to various Vollenweider-like models.

    Args
        df:       Dataframe of lake data
        par_name: Str. Name for parameter
        params:   Dict. Must contains the following keys:
                      ind_var_col: Str. Column name in 'df' containing the
                                   independent variable (e.g. 'tau' or 'H')
                      model        Str. One of [
                                       'sigma_constant',
                                       'sigma_from_tau',
                                       'sigma_from_depth',
                                       ]
                      <other>      Passed as kwargs to the relevant 'model'
                                   function.

    Returns
        Dataframe. 'df' is returned with two new columns added: 'trans_par'
        and 'ret_par', where 'par' is the 'par_name' provided.
    """
    ind_var_col = params.pop("ind_var_col")
    model = params.pop("model")

    models = ["sigma_constant", "sigma_from_tau", "sigma_from_depth"]
    assert model in models, f"'model' must be one of @models."
    assert ind_var_col in df.columns, f"'{ind_var_col}' not found in 'df'."

    if model == "sigma_constant":
        trans = transmission_sigma_constant(df[ind_var_col], **params)
    elif model == "sigma_from_tau":
        trans = transmission_sigma_from_tau(df[ind_var_col], **params)
    else:
        trans = transmission_sigma_from_depth(df[ind_var_col], **params)

    df[f"trans_{par_name}"] = trans
    df[f"ret_{par_name}"] = 1 - trans

    return df


def calculate_regine_retention(df, regine_col, pars):
    """Aggregate lake retention factors to regine level by combining in
    series. See

    https://niva.brage.unit.no/niva-xmlui/bitstream/handle/11250/2985726/7726-2022%2bhigh.pdf?sequence=1&isAllowed=y#page=23

    for details.

    Args
        df:         Dataframe of lake retention data. Must contain cols
                    'trans_par' for all 'pars'
        regine_col: Str. Column name in 'df' with regine codes for
                    aggregation
        pars:       List of str. Parameters to aggregate

    Retruns
        Dataframe with columns [regine, trans_par, ret_par] for all
        parameters in 'pars'.
    """
    assert regine_col in df.columns, "'regine_col' not found in 'df'."
    trans_cols = [f"trans_{par}" for par in pars]
    for col in trans_cols:
        assert col in df.columns, f"'{col}' not found in 'df'."

    reg_df = df.groupby(regine_col).prod()[trans_cols].round(6).reset_index()
    for par in pars:
        reg_df[f"ret_{par}"] = (1 - reg_df[f"trans_{par}"]).round(6)

    return reg_df


def assign_regine_retention(reg_gdf, regine_col="regine", dtm_res=10):
    """Assign retention and transmission coefficients to each regine.

    Args
        reg_gdf:    Geodataframe of regine boundaries from NVE.
        regine_col: Str. Name of column in reg_gdf with regine codes
        dtm_res:    Int. Resolution in file name of CSV with residence times

    Returns
        Copy of 'reg_gdf' with retention and transmission columns added
        for each parameter.
    """
    reg_gdf = reg_gdf.copy()

    # Get lake residence times
    res_csv = f"../../data/lake_residence_times_{dtm_res}m_dem.csv"
    df = pd.read_csv(res_csv)

    # Vollenweider parameters for individual lakes
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
            "k": 0.5,
            "p": 0.5,
        },
        "tpp": {
            "ind_var_col": "res_time_yr",
            "model": "sigma_from_tau",
            "k": 2,
            "p": 0.5,
        },
        "totn": {"ind_var_col": "hyd_load_mpyr", "model": "sigma_from_depth", "s": 4.5},
        "din": {"ind_var_col": "hyd_load_mpyr", "model": "sigma_from_depth", "s": 6.0},
        "ton": {"ind_var_col": "hyd_load_mpyr", "model": "sigma_from_depth", "s": 1.4},
        "ss": {"ind_var_col": "res_time_yr", "model": "sigma_constant", "sigma": 90},
        "toc": {
            "ind_var_col": "res_time_yr",
            "model": "sigma_from_tau",
            "k": 0.6,
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
    """Read the raw aquaculture data from Fiskeridirektoratet. Returns a dataframe of
    site locations, plus a dataframe of raw monthly data for further processing.

    Args
        xl_path:    Str. Path to Excel file from Fiskeridirektoratet
        sheet_name: Str. Worksheet to read
        year:       Int. Year being processed

    Returns
        Tuple of (geo)dataframes (loc_gdf, data_df).
    """
    assert isinstance(xl_path, str), "'xl_path' must be a valid file path."
    assert isinstance(sheet_name, str), "'sheet_name' must be a string."
    assert isinstance(year, int), "'year' must be an integer."

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
        df:          Dataframe of raw monthly data
        year:        Int. Year of interest
        eng:         Obj. Active database connection object connected to PostGIS
        cu_tonnes:   Float. Optional. Total annual usage of copper by the aquaculture industry
                     in tonnes. If supplied, 85% of this value is assumed to be lost to the
                     environment. Losses are assigned to each aquaculture site in proportion
                     to the total loss of P
        species_ids: List. Species to consider. The default is salmon and rainbow trout

    Returns
        Dataframe of estimated nutrient losses that can be added to teotil3.point_source_values.
    """
    assert isinstance(year, int), "'year' must be an integer."
    assert isinstance(species_ids, list), "'species_ids' must be a list."
    if cu_tonnes:
        assert isinstance(
            cu_tonnes, (int, float, np.number)
        ), "'cu_tonnes' must be a number."

    # Read coefficients for aquaculture calcs
    url = r"https://raw.githubusercontent.com/NIVANorge/teotil3/main/data/aquaculture_productivity_coefficients.csv"
    coeff_df = pd.read_csv(url, index_col=0)
    fcr = coeff_df.loc["fcr"]["value"]
    k_feed_n = coeff_df.loc["k_feed_n"]["value"]
    k_feed_p = coeff_df.loc["k_feed_p"]["value"]
    k_feed_c = coeff_df.loc["k_feed_c"]["value"]
    k_prod_n = coeff_df.loc["k_prod_n"]["value"]
    k_prod_p = coeff_df.loc["k_prod_p"]["value"]

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

    # Aggregate by month, location and species
    agg_df = df.groupby(by=["site_id", "sector", "type", "MAANED", "FISKEARTID"])
    sum_df = agg_df.sum(numeric_only=True)[["FORFORBRUK_KILO", "biomass_kg"]]

    # Get biomass for previous month
    sum_df["biomass_prev_kg"] = sum_df.apply(
        get_aquaculture_biomass_previous_month, args=(sum_df,), axis=1
    )

    # Get productivity for each month
    sum_df["prod_kg"] = sum_df.apply(
        calculate_aquaculture_productivity, args=(fcr,), axis=1
    )

    # Calculate nutrient losses
    sum_df["TOTN_kg"] = sum_df.apply(
        calculate_aquaculture_n_and_p_loss,
        args=(k_feed_n, k_prod_n, fcr),
        axis=1,
    )
    sum_df["TOTP_kg"] = sum_df.apply(
        calculate_aquaculture_n_and_p_loss,
        args=(k_feed_p, k_prod_p, fcr),
        axis=1,
    )
    sum_df["TOC_kg"] = sum_df.apply(
        calculate_aquaculture_toc_loss,
        args=(k_feed_c, fcr),
        axis=1,
    )

    # Get just the data for species of interest
    sum_df.reset_index(inplace=True)
    sum_df = sum_df.query("FISKEARTID in @species_ids")

    # Aggregate by location
    agg_df = sum_df.groupby(by=["site_id", "sector", "type"])
    sum_df = agg_df.sum(numeric_only=True)[["TOTN_kg", "TOTP_kg", "TOC_kg"]]

    # Distribute Cu according to P production
    if cu_tonnes:
        cu_loss_tonnes = 0.85 * cu_tonnes
        print(
            f"The total annual copper lost to water from aquaculture is {cu_loss_tonnes:.1f} tonnes."
        )
        sum_df["Cu_kg"] = (
            1000 * cu_loss_tonnes * sum_df["TOTP_kg"] / sum_df["TOTP_kg"].sum()
        )

    # Subdivide TOTN and TOTP
    sum_df.reset_index(inplace=True)
    sum_df = subdivide_point_source_n_and_p(sum_df, "Aquaculture", "TOTN_kg", "TOTP_kg")
    id_cols = ["site_id", "sector", "type"]
    sum_df = sum_df.melt(id_vars=id_cols)

    # Convert to db par_ids
    sql = text(
        """SELECT in_par_id,
             CONCAT_WS('_', name, unit) AS par_unit
           FROM teotil3.input_param_definitions
        """
    )
    input_par_df = pd.read_sql(sql, eng)
    par_map = input_par_df.set_index("par_unit").to_dict()["in_par_id"]

    sum_df["in_par_id"] = sum_df["variable"].map(par_map)
    sum_df["year"] = year
    sum_df = sum_df[["site_id", "in_par_id", "year", "value"]]

    return sum_df


def get_aquaculture_biomass_previous_month(row, df):
    """Returns fish farm biomass for the previous month. If month = 1, or if data for the
    previous month are not available, returns 0.

    Args
        row: Obj. Dataframe row
        df:  Obj. Original dataframe containing data for other months

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
    """Calculate fish farm productivity based on change in biomass compared to the previous
    month. If biomass has increased, the productivity is the increase in kg. If biomass
    has decreased, or if either of the biomasses are zero, the Feed Conversion Ratio (FCR) is
    used instead - see Section 3.3.6 here:

    https://niva.brage.unit.no/niva-xmlui/bitstream/handle/11250/2985726/7726-2022+high.pdf?sequence=1#page=29

    Args
        row: Obj. Dataframe row
        fcr: Float. Feed Conversion Ratio to use when biomass figures are not available

    Returns
        Float. Productivity for month in kg.
    """
    assert isinstance(fcr, (int, float)), "'fcr' must be a number."

    if (
        (row["biomass_kg"] == 0)
        or (row["biomass_prev_kg"] == 0)
        or (row["biomass_kg"] < row["biomass_prev_kg"])
    ):
        return row["FORFORBRUK_KILO"] / fcr

    else:
        return row["biomass_kg"] - row["biomass_prev_kg"]


def calculate_aquaculture_n_and_p_loss(row, k_feed, k_prod, fcr):
    """Calculate the balance of "nutrients in" versus "nutrients out" for aquaculture. For
    any parameter, X, (e.g. TOTN or TOTP) 'k_feed' is the proportion of X in the feed, and
    'k_prod' is the proportion of X in exported fish. The default values used by TEOTIL are
    here:

    https://github.com/NIVANorge/teotil3/blob/main/data/aquaculture_productivity_coefficients.csv

    The balance is calculated as

        losses = inputs - outputs
               = (k_feed * feed_use) - (k_prod * productivity)

    If productivity data are not available, or if the apparent nutrient balance is
    negative, the Feed Conversion Ratio is used to estimate productivity.

    Args
        row:    Obj. Dataframe row being processed
        k_feed: Float. Proportion of X in feed
        k_prod: Float. Proportion of X in exported fish
        fcr:    Float. Feed Conversion Ratio to use when productivity figures are not available

    Returns
        Float. Nutrients lost in kg.
    """
    assert isinstance(fcr, (int, float)), "'fcr' must be a number."
    assert isinstance(k_feed, (int, float)), "'fcr' must be a number."
    assert isinstance(k_prod, (int, float)), "'fcr' must be a number."

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
    """Estimate TOC losses from aquaculture based on feed use. The function implements the
    method described in section 6.2.4 here:

    https://niva.brage.unit.no/niva-xmlui/bitstream/handle/11250/2985726/7726-2022+high.pdf?sequence=1#page=43

    Args
         row:    Obj. Dataframe row being processed
         k_feed: Float. Propoertion of TOC in feed
         fcr:    Float. Feed Conversion Ratio to use when feed use figures are not available

    Returns
        Float. TOC loss in kg.
    """
    assert isinstance(k_feed, (int, float)), "'k_feed' must be a number."
    assert isinstance(fcr, (int, float)), "'fcr' must be a number."

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
    """
    url = r"https://raw.githubusercontent.com/NIVANorge/teotil3/main/data/aquaculture_annual_copper_usage.csv"
    df = pd.read_csv(url, index_col=0)
    cu_tonnes = df.loc[year]["tot_cu_tonnes"]

    return cu_tonnes


def utm_to_wgs84_dd(utm_df, zone="utm_zone", east="utm_east", north="utm_north"):
    """Converts UTM co-ordinates to WGS84 decimal degrees, allowing for each row in
    'utm_df' to have a different UTM Zone. (Note that if all rows have the same zone,
    this implememntation is slow because it processes each row individually).

    Args
        utm_df: Dataframe containing UTM co-ords
        zone:   Str. Column defining UTM zone
        east:   Str. Column defining UTM Easting
        north:  Str. Column defining UTM Northing

    Returns
        Copy of utm_df with 'lat' and 'lon' columns added.
    """
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


def read_raw_large_wastewater_data(data_fold, year):
    """Reads the raw, gap-filled data for TOTN, TOTP, BOF5 and KOF from "large" (>50 p.e.)
    wastewater treatment sites provided by SSB. Note that this dataset includes some data
    that is duplicated in the "miljøgifter" dataset.

    Args
        data_fold: Str. Folder containg raw data files, with the file structure as
                   described above
        year:      Int. Year being processed

    Returns
        Geodataframe in 'wide' format. A point geodataframe in EPSG 25833.
    """
    assert isinstance(data_fold, str), "'data_fold' must be a valid file path."
    assert isinstance(year, int), "'year' must be an integer."

    # Read site locs and data for TOTN and TOTP
    stan_path = os.path.join(data_fold, f"avlop_stor_anlegg_{year}_raw.xlsx")
    df = pd.read_excel(
        stan_path,
        sheet_name=f"store_anlegg_{year}",
    )
    df.dropna(how="all", inplace=True)
    df["sector"] = "Large wastewater"
    df["year"] = year
    df.rename(
        {
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
        },
        axis="columns",
        inplace=True,
    )
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
    for col in ["zone", "east", "north"]:
        df[f"outlet_{col}"].fillna(df[f"site_{col}"], inplace=True)

    # Only sites in valid UTM zones
    for loc in ["site", "outlet"]:
        if df[f"{loc}_zone"].min() < 31 or df[f"{loc}_zone"].max() > 36:
            print(
                f"'{loc}_zone' column in Large Wastewater contains values outside valid range [31, 36]. These will be dropped."
            )
            df = df.query(f"31 <= {loc}_zone <= 36")

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
    val_cols = ["TOTP_kg", "TOTN_kg", "BOF5_kg", "KOF_kg"]
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
    """
    assert isinstance(data_fold, str), "'data_fold' must be a valid file path."
    assert isinstance(year, int), "'year' must be an integer."

    miljo_path = os.path.join(data_fold, f"avlop_miljogifter_{year}_raw.xlsx")
    df = pd.read_excel(miljo_path, sheet_name=f"miljogifter_{year}")
    df.dropna(how="all", inplace=True)
    df["sector"] = "Large wastewater"
    df["year"] = year
    df.rename(
        {
            "ANLEGGSNR": "site_id",
            "ANLEGGSNAVN": "name",
            "SONEBELTE": "site_zone",
            "UTMOST": "site_east",
            "UTMNORD": "site_north",
            "RESIP2": "outlet_zone",
            "RESIP3": "outlet_east",
            "RESIP4": "outlet_north",
        },
        axis="columns",
        inplace=True,
    )
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
    for col in ["zone", "east", "north"]:
        df[f"outlet_{col}"].fillna(df[f"site_{col}"], inplace=True)

    # Only sites in valid UTM zones
    for loc in ["site", "outlet"]:
        if df[f"{loc}_zone"].min() < 31 or df[f"{loc}_zone"].max() > 36:
            print(
                f"'{loc}_zone' column in Miljøgifter dataset contains values outside valid range [31, 36]. These will be dropped."
            )
            df = df.query(f"31 <= {loc}_zone <= 36").copy()

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
    """
    assert isinstance(data_fold, str), "'data_fold' must be a valid file path."
    assert isinstance(year, int), "'year' must be an integer."

    ind_path = os.path.join(data_fold, f"industri_{year}_raw.xlsx")
    df = pd.read_excel(ind_path, sheet_name=f"industri_{year}")
    df.dropna(how="all", inplace=True)
    if len(df["År"].unique()) > 1:
        print(
            f"WARNING: The industry dataset includes values for several years. Only data for {year} will be processed."
        )
        df = df.query("`År` == @year")
    df["sector"] = "Industry"
    df["year"] = year
    df.rename(
        {
            "Anleggsnr": "site_id",
            "Anleggsnavn": "name",
            "Anleggsaktivitet": "type",
            "Geografisk Longitude": "site_lon",
            "Geografisk Latitude": "site_lat",
            "Lon_Utslipp": "outlet_lon",
            "Lat_Utslipp": "outlet_lat",
            "Mengde": "value",
        },
        axis="columns",
        inplace=True,
    )

    # If the outlet co-ords aren't known, use the site co-ords instead
    for col in ["lon", "lat"]:
        df[f"outlet_{col}"].fillna(df[f"site_{col}"], inplace=True)

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
    val_cols = ["variable", "value"]
    gdf = gdf[id_cols + val_cols]

    # Convert to wide format
    gdf = gdf.pivot(index=id_cols, columns="variable", values="value").reset_index()

    return gdf


def read_large_wastewater_and_industry_data(data_fold, year, eng):
    """Convenience function for processing the raw "store anlegg", "miljøgifter" and
    "industry" datasets. Assumes the raw files are named and arranged as follows:

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

    This function reads all the raw files and combines site locations and data values
    into a geodataframe and a dataframe, respectively. Parameter names are mapped to
    input parameter IDs in the database, and duplicates are removed. Sites without
    coordinates are highlighted.

    Args
        data_fold: Str. Folder containg raw data files, with the file structure as
                   described above
        year:      Int. Year being processed
        eng:       Obj. Active database connection object connected to PostGIS

    Returns
        Tuple of (geo)dataframes (loc_gdf, df). 'loc_gdf' is a point geodataframe of
        site co-ordinates in EPSG 25833; 'df' is a dataframe of discharges from each
        site.
    """
    assert isinstance(data_fold, str), "'data_fold' must be a valid file path."
    assert isinstance(year, int), "'year' must be an integer."

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
    df = df.query("variable in @db_pars").copy()

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

    # Convert to db IDs
    df["in_par_id"] = df["variable"].map(par_map)

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
    """ """
    assert isinstance(xl_path, str), "'xl_path' must be a valid file path."
    assert isinstance(sheet_name, str), "'sheet_name' must be a string."
    assert isinstance(year, int), "'year' must be an integer."

    # Check admin. boundaries for 'year' are assigned in PostGIS. Before 2015, we
    # just use the boundaries for 2014
    if year < 2015:
        admin_year = 2014
    else:
        admin_year = year
    try:
        sql = text(f"SELECT komnr_{admin_year} AS komnr FROM teotil3.regines")
        reg_kom_df = pd.read_sql(sql, eng)
    except exc.ProgrammingError:
        raise ValueError(
            f"Administrative boundaries for {year} are not linked to regines in the TEOTIL database. "
            "Please download the latest boundaries from Geonorge and update the regine dataset."
        )

    df = pd.read_excel(xl_path, sheet_name=sheet_name)
    df.dropna(how="all", inplace=True)
    df = df.query("year == @year")
    del df["year"]
    df.rename({"KOMMUNENR": "komnr"}, axis="columns", inplace=True)

    # Komnr. should be a 4 char string, not a float
    fmt = lambda x: "%04d" % x
    df["komnr"] = df["komnr"].apply(fmt)

    # Before 2020, there was a very small kommune (1245; Sund) that is not linked to any regines
    # "spredt" for this kommune can be aggregated with that for 1246
    assert df[
        "komnr"
    ].is_unique, f"Dataset contains duplicated kommuner: {list(df['komnr'].unique())}"
    df["komnr"].replace({"1245": "1246"}, inplace=True)
    df = df.groupby("komnr").sum(numeric_only=True).reset_index()

    # Restructure data
    df = df.melt(id_vars="komnr")
    df["sector"] = "Small wastewater"
    df["variable"] = df["variable"].str.replace("FOSFOR ", "TOTP_kg;")
    df["variable"] = df["variable"].str.replace("NITROGEN ", "TOTN_kg;")
    df["variable"] = df["variable"].str.replace("BOF ", "BOF5_kg;")
    df[["variable", "type"]] = df["variable"].str.split(";", n=1, expand=True)
    # Ignore 'Tett tank (for alt avløpsvann)' as it is always zero (it's transported to the "large" plants)
    df = df.query("type != 'Tett tank (for alt avløpsvann)'")
    df.set_index(["komnr", "variable", "sector", "type"], inplace=True)
    df = df.unstack("variable")
    df.columns = df.columns.get_level_values(1)
    df.reset_index(inplace=True)
    df.index.name = ""

    # Subdivide TOTN and TOTP
    df = subdivide_point_source_n_and_p(df, "Small wastewater", "TOTN_kg", "TOTP_kg")

    # Estimate TOC from BOF5
    df["KOF_kg"] = np.nan
    df = estimate_toc_from_bof_kof(df, "Small wastewater")

    # Sum totals for each type back to kommune level
    df = df.groupby("komnr").sum(numeric_only=True).reset_index()

    # Check komnrs in SSB data are found in the admin data for this year
    not_in_db = set(df["komnr"].values) - set(reg_kom_df["komnr"].values)
    if len(not_in_db) > 0:
        print(len(not_in_db), 'kommuner are not in the TEOTIL "regine" dataset.')
        print(df[df["komnr"].isin(list(not_in_db))])

    # Convert to par_ids used in database
    sql = text(
        """SELECT in_par_id,
             CONCAT_WS('_', name, unit) AS par_unit
           FROM teotil3.input_param_definitions
        """
    )
    input_par_df = pd.read_sql(sql, eng)
    par_map = input_par_df.set_index("par_unit").to_dict()["in_par_id"]
    df.rename(par_map, axis="columns", inplace=True)
    df = pd.melt(df, id_vars="komnr", var_name="in_par_id", value_name="value")
    df["year"] = year
    df = df[["komnr", "in_par_id", "year", "value"]]

    return df


def subdivide_point_source_n_and_p(gdf, sector, totn_col, totp_col):
    """Subdivide TOTN and TOTP from wastewater, industry or aquaculture. Uses typical
    fractions for different types of treatment plant, based on Christian Vogelsang's
    literature review. See the file here for details:

    https://github.com/NIVANorge/teotil3/blob/main/data/point_source_treatment_types.csv

    Args
        gdf:      Geodataframe of discharges. Must include columns for site or kommune ID,
                  treatment type, year, TOTN and TOTP
        sector :  Str. Type of sites being processed. Must be one of
                  ["Large wastewater", "Small wastewater", "Industry", "Aquaculture"]
        totn_col: Str. Name of column in 'gdf' with TOTN data
        totp_col: Str. Name of column in 'gdf' with TOTP data

    Returns
        Geodataframe. 'gdf' is returned with columns for subfractions added.
    """
    assert isinstance(totn_col, str), "'totn_col' must be a string."
    assert isinstance(totp_col, str), "'totp_col' must be a string."
    sectors = ["Large wastewater", "Small wastewater", "Industry", "Aquaculture"]
    assert sector in sectors, f"{sector} must be one of {sectors}."

    url = r"https://raw.githubusercontent.com/NIVANorge/teotil3/main/data/point_source_treatment_types.csv"
    prop_df = pd.read_csv(url)
    prop_df = prop_df.query("sector == @sector")

    assert (
        prop_df["prop_din"] + prop_df["prop_ton"] == 1
    ).all(), "Proportions for DIN and TON do not sum to one."
    assert (
        prop_df["prop_tpp"] + prop_df["prop_tdp"] == 1
    ).all(), "Proportions for TPP and TDP do not sum to one."
    assert set(gdf["type"]).issubset(
        set(prop_df["type"])
    ), f"'gdf' contains unknown treatment types: {set(gdf['type']) - set(prop_df['type'])}."

    gdf = gdf.merge(prop_df, how="left", on=["sector", "type"])

    fracs = ["DIN", "TON", "TPP", "TDP"]
    for frac in fracs:
        if frac[-1] == "N":
            tot_col = totn_col
        else:
            tot_col = totp_col
        unit = tot_col.split("_")[-1]
        if unit == "tonnes":
            unit_factor = 1000
        elif unit == "kg":
            unit_factor = 1
        else:
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
        gdf:    (Geo)dataframe of discharges. Must include columns for site/kommune, treatment
                type and BOF/KOF
        sector: Str. Type of sites being processed. Must be one of
                ["Large wastewater", "Small wastewater", "Industry"]

    Returns
        Geodataframe in same format as 'gdf', but with TOC added.
    """
    sectors = ["Large wastewater", "Small wastewater", "Industry"]
    assert sector in sectors, f"{sector} must be one of {sectors}."

    url = r"https://raw.githubusercontent.com/NIVANorge/teotil3/main/data/point_source_treatment_types.csv"
    prop_df = pd.read_csv(url)
    prop_df = prop_df.query("sector == @sector")

    assert set(gdf["type"]).issubset(
        set(prop_df["type"])
    ), f"'gdf' contains unknown treatment types: {set(gdf['type']) - set(prop_df['type'])}."

    if sector == "Industry":
        bof_col = "BOF5"
        kof_col = "KOF"
        unit = "tonnes"
    else:
        bof_col = "BOF5"
        kof_col = "KOF"
        unit = "kg"

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
    for col in prop_df.columns:
        if col not in ["sector", "type"]:
            del gdf[col]

    return gdf
