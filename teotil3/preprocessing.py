import geopandas as gpd
import numpy as np
import pandas as pd
from rasterstats import zonal_stats
from sqlalchemy import text


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
    reg_gdf["vassom"] = reg_gdf["regine"].str.split(".", 1).str[0].astype(int)
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
    lc_df = int_gdf.groupby(["regine", "teotil"]).sum()["area_km2"]
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
    int_gdf = int_gdf.groupby("regine")["a_lake_nve_km2"].sum().reset_index()
    reg_gdf = pd.merge(reg_gdf, int_gdf, on="regine", how="left")
    reg_gdf["a_lake_nve_km2"].fillna(0, inplace=True)

    # Move 'geometry' to end
    cols = reg_gdf.columns.tolist()
    cols.remove("geometry")
    cols.append("geometry")
    reg_gdf = reg_gdf[cols]

    return reg_gdf


def calculate_lake_retention_vollenweider(df, res_time_col, par_name, sigma, n):
    """Calculate retention and transmission factors for individual lakes
    according to Vollenweider (1975).

        R = sigma / (sigma + rho^n)

    where rho is the reciprocal of the residence time (in years), and
    sigma and n are parameter-specific constants. See

    https://niva.brage.unit.no/niva-xmlui/bitstream/handle/11250/2985726/7726-2022%2bhigh.pdf?sequence=1&isAllowed=y#page=23

    for details.

    Args
        df:           Obj. Dataframe of lake data
        res_time_col: Str. Column name in 'df' containing water residence
                      times (in years)
        par_name:     Str. Name for parameter
        sigma:        Int or Float. Parameter-specific constant. Must be
                      positive
        n:            Int or Float. Parameter-specific exponent. Must be
                      positive

    Returns
        Dataframe. 'df' is returned with two new columns added: 'trans_par'
        and 'ret_par', where 'par' is the 'par_name' provided.
    """
    assert res_time_col in df.columns, "'res_time_col' not found in 'df'."
    assert sigma > 0, "'sigma' must be positive."
    assert n > 0, "'n' must be positive."

    df[f"ret_{par_name}"] = sigma / (sigma + ((1 / df[res_time_col]) ** n))
    df[f"trans_{par_name}"] = 1 - df[f"ret_{par_name}"]

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

    reg_df = df.groupby(regine_col).prod()[trans_cols].reset_index()
    for par in pars:
        reg_df[f"ret_{par}"] = 1 - reg_df[f"trans_{par}"]

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

    # Vollenweider parameters for individual lakes.
    # The parameters sigma and n are those matching the following (equivalent) equations
    #     R = sigma / (sigma + rho^n) where rho = 1/tau
    #     T = 1 / (1 + sigma * tau^n)
    # par_name: (sigma, n)
    voll_dict = {
        "totp": (1, 0.5),
        "tdp": (0.5, 0.5),
        "tpp": (2, 0.5),
        "totn": (0.75, 0.4),
        "din": (1, 1),
        "ton": (0.2, 1),
        "ss": (90, 1),
        "toc": (0.6, 0.4),
    }
    for par in voll_dict.keys():
        sigma, n = voll_dict[par]
        df = calculate_lake_retention_vollenweider(
            df, "res_time_yr", par, sigma=sigma, n=n
        )

    # Non-Vollenweider params for individual lakes
    # Original ret_n assumed to be 0.2*ret_totp
    df["ret_orig-totn"] = 0.2 * df["ret_totp"]
    df["trans_orig-totn"] = 1 - df["ret_orig-totn"]

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


def read_raw_aquaculture_data(xl_path, sheet_name, year, eng):
    """Read the raw aquaculture data from Fiskeridirektoratet. Identifies sites missing
    co-ordinates and those not already in the database. Returns a dataframe of new site
    locations to add to the database, plus a dataframe of raw monthly data for further
    processing.

    Args
        xl_path:    Str. Path to Excel file from Fiskeridirektoratet
        sheet_name: Str. Worksheet to read
        year:       Int. Year being processed
        eng:        Obj. Active database connection object connected to PostGIS

    Returns
        Tuple of Dataframes (new_locs_df, data_df).
    """
    assert isinstance(xl_path, str), "'xl_path' must be a valid file path."
    assert isinstance(sheet_name, str), "'sheet_name' must be a string."
    assert isinstance(year, int), "'year' must be an integer."

    # Relevant cols in raw data
    cols_dict = {
        "AAR": "year",
        "LOKNR": "site_id",
        "LOKNAVN": "name",
        "N_DESIMALGRADER_Y": "lat",
        "O_DESIMALGRADER_X": "lon",
    }

    # Read raw file
    df = pd.read_excel(xl_path, sheet_name=sheet_name)
    df.dropna(how="all", inplace=True)
    df.rename(
        cols_dict,
        axis="columns",
        inplace=True,
    )
    df = df.query("year == @year")
    df["site_id"] = df["site_id"].astype(str)
    # df = df[cols_dict.values()]

    # Check for missing co-ords
    no_coords_df = df.query("(lat != lat) or (lon != lon)")[
        ["site_id", "name"]
    ].sort_values("site_id")
    print(
        f"{len(no_coords_df)} locations do not have co-ordinates in this year's data."
    )

    # Check for sites are not already in db
    sql = text(
        """SELECT DISTINCT(site_id) FROM teotil3.point_source_locations
           WHERE type = 'Aquaculture'
        """
    )
    in_db_df = pd.read_sql_query(sql, eng)
    not_in_db = set(df["site_id"].values) - set(in_db_df["site_id"].values)
    not_in_db_df = df[df["site_id"].isin(list(not_in_db))][
        ["site_id", "name", "lat", "lon"]
    ].drop_duplicates(subset=["site_id"])
    print(f"{len(not_in_db_df)} locations are not in the database.")
    if len(not_in_db_df) > 0:
        # Format df to match db
        not_in_db_df.dropna(subset=["lat", "lon"], how="any", inplace=True)
        not_in_db_gdf = gpd.GeoDataFrame(
            not_in_db_df,
            geometry=gpd.points_from_xy(
                not_in_db_df["lon"], not_in_db_df["lat"], crs="epsg:4326"
            ),
        )
        not_in_db_gdf = not_in_db_gdf.to_crs("epsg:25833")
        not_in_db_gdf["type"] = "Aquaculture"
        not_in_db_gdf = not_in_db_gdf[["site_id", "name", "type", "geometry"]]
        not_in_db_gdf.rename({"geometry": "geom"}, axis="columns", inplace=True)
        not_in_db_gdf.reset_index(drop=True, inplace=True)

        return not_in_db_gdf, df

    return None, df


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
        assert isinstance(cu_tonnes, (int, float)), "'cu_tonnes' must be a number."

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
    agg_df = df.groupby(by=["site_id", "MAANED", "FISKEARTID"])
    sum_df = agg_df.sum()[["FORFORBRUK_KILO", "biomass_kg"]]

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
    agg_df = sum_df.groupby(by=["site_id"])
    sum_df = agg_df.sum()[["TOTN_kg", "TOTP_kg", "TOC_kg"]]

    # Distribute Cu according to P production
    if cu_tonnes:
        cu_loss_tonnes = 0.85 * cu_tonnes
        print(
            f"The total annual copper lost to water from aquaculture is {cu_loss_tonnes:.1f} tonnes."
        )
        sum_df["Cu_kg"] = (
            1000 * cu_loss_tonnes * sum_df["TOTP_kg"] / sum_df["TOTP_kg"].sum()
        )

    # Convert to par_ids and melt to format required by db
    sql = text(
        """SELECT in_par_id,
             CONCAT_WS('_', name, unit) AS par_unit
           FROM teotil3.input_param_definitions
        """
    )
    input_par_df = pd.read_sql(sql, eng)
    par_map = input_par_df.set_index("par_unit").to_dict()["in_par_id"]

    sum_df.rename(par_map, axis="columns", inplace=True)
    sum_df.reset_index(inplace=True)
    sum_df = pd.melt(
        sum_df, id_vars="site_id", var_name="in_par_id", value_name="value"
    )
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
    loc = row.name[0]
    mon = row.name[1]
    spec = row.name[2]

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
