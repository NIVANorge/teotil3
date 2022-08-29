import re

import geopandas as gpd
import numpy as np
import pandas as pd


def summarise_regine_hydrology(reg_gdf):
    """Summarise basic, regine-level hydrology information. Note that this
    function hard-codes column names as currently used by NVE in their FGDB
    version of the regine dataset. These names are defined in NVE's metadata
    and should not change in the future, but if they do this function will
    need updating/generalising. Note also that names are different in e.g.
    the shapefile version of the dataset (due to the 10 character limit on
    field names).

    Args
        reg_gdf: Geodataframe of regines, with column names as originally
                 defined by NVE in their FGDB version of the dataset (see
                 'ObjekttyperEgenskaper_Nedborfelt.pdf' in the download
                 metadata for details)

    Returns
        Geodataframe. A copy of the original geodataframe, but with new
        columns added containing hydrological information and unnecessary
        columns removed. Catchments on Svalbard are also removed from the
        dataset and all regines are sorted alphanumerically based on their
        ID code.
    """
    reg_gdf = reg_gdf.copy()
    reg_gdf.rename(
        {
            "vassdragsnummer": "regine",
            "regineAreal_km2": "a_cat_land_km2",  # NOTE: This is the land area in each regine, not necessarily the polygon area
            "nedborfeltOppstromAreal_km2": "upstr_a_km2",
            "QNedborfeltOppstrom_Mm3Aar": "upstr_runoff_Mm3/yr",
        },
        axis="columns",
        inplace=True,
    )

    # Calculate derived columns
    reg_gdf["a_cat_poly_km2"] = reg_gdf.to_crs({"proj": "cea"})["geometry"].area / 1e6
    reg_gdf["q_sp_m3/s/km2"] = reg_gdf["QNormalr6190_lskm2"] / 1000
    reg_gdf["q_cat_m3/s"] = reg_gdf["q_sp_m3/s/km2"] * reg_gdf["a_cat_land_km2"]
    reg_gdf["runoff_mm/yr"] = reg_gdf["q_sp_m3/s/km2"] * 60 * 60 * 24 * 365.25 / 1000

    # Remove Svalbard
    reg_gdf["vassom"] = reg_gdf["regine"].str.split(".", 1).str[0].astype(int)
    reg_gdf = reg_gdf.query("vassom < 400").copy()
    reg_gdf["vassom"] = reg_gdf["vassom"].apply(lambda x: f"{x:03}")

    # Get cols of interest
    reg_cols = [
        "regine",
        "a_cat_land_km2",
        "a_cat_poly_km2",
        "upstr_a_km2",
        "upstr_runoff_Mm3/yr",
        "q_sp_m3/s/km2",
        "runoff_mm/yr",
        "q_cat_m3/s",
        "vassom",
        "geometry",
    ]
    reg_gdf = reg_gdf[reg_cols]
    reg_gdf.sort_values("regine", inplace=True)
    reg_gdf.reset_index(inplace=True, drop=True)

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


def geodataframe_to_geopackage(
    gdf, gpkg_path, layer, cols=None, attrib_tab_csv=None, index=False
):
    """Save a geodataframe as a layer within a geopackage. If the geopackage
    doesn't already exist, it is created. If it does exist, the layer is
    appended. Options are included to subset/reorder columns before saving,
    and to save the attribute table as a separate CSV if desired.

    Args
        gdf:            Geodataframe to save
        gpkg_path:      Str. Path to geopackage where 'gdf' will be saved. If
                        the geopackage does not exist, it will be created
        layer:          Str. Name of layer to create within the geopackage. If
                        the layer already exists, it will be overwritten.
        cols:           List of str or None. Default None. Optionally subset
                        or reorder columns in 'gdf' before saving. If None,
                        all columns are saved in the original order
        attrib_tab_csv: Str or None. Default None. Optional path to CSV where
                        the "attribute table" should be saved. This is a table
                        containing all the non-spatial information from the
                        geodataframe
        index:          Bool. Default False. Whether to save the (geo)dataframe
                        index

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
        Bool. True if 's' can be converted to an integer, otherwise
        False.
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
    """Determine hydrological ordering of regine catchments based on
    catchment IDs. See

    https://www.nve.no/media/2297/regines-inndelingssystem.pdf

    for an overview of the regine coding scheme.

    The default parameters aim to reproduce the behaviour of the original
    model, but are not necessarily the best choices in most cases.

    Args
        df:              (Geo)Dataframe of regine data
        regine_col:      Str. Default 'regine'. Name of column in 'df'
                         containing the regine IDs
        regine_down_col: Str. Default 'regine_down'. Name of new column to be
                         created to store the next down IDs
        order_coastal:   Bool. Default True. If True, coastal catchments will
                         be ordered hierarchically, as in the original TEOTIL
                         model (e.g. '001.6' flows into '001.5', which flows
                         into '001.4' etc.). In most cases this does not make
                         sense, as the numbering of kystfelt does not reflect
                         dominant circulation patterns. If False all kystfelt
                         are assigned directly to the main vassdragsområde
                         (e.g. '001.6', '001.5' and '001.4' would all flow
                         into '001.')
        nan_to_vass:     Bool. Default False. Whether catchments for which no
                         downstream ID can be identified should be assigned
                         directly to the parent vassdragsområde. In some areas -
                         notably catchments in the east draining to Sweden - the
                         regines are truncated mid-catchment. There is therefore
                         no valid downstream catchment. For completeness, it is
                         often useful to assign these catchments to the parent
                         vassdragsområde, rather than leaving them as NaN. This
                         behaviour can be disabled by setting 'nan_to_vass' to
                         False. If True, a warning will be printed if NaNs are
                         filled for any catchments other than those draining to
                         Sweden (vassdragsområder 301 to 315)
        land_to_vass:    Bool. Default False. Whether downstream non-coastal
                         catchments should be linked directly to the parent
                         vassdragsområde, or to the highest ranking kystfelt. In
                         the original TEOTIL model, terrestrial catchments are
                         linked to the highest-ranking coastal fields. However,
                         these are not necessarily anywhere near the catchment
                         outflows. An alternative is to link terrestrial outflows
                         directly to the vassdragområder, which is simpler but
                         may give misleading results for some kystfelt with large
                         river mouths. If 'order_coastal' is True, 'land_to_vass'
                         must be False. Catchments will then be arranged as for
                         the original model. If 'order_coastal' is False, use
                         this kwarg to control whether the algorithm links
                         terrestrial fluxes to kystfelt, or directly to the parent
                         vassdragområde
        add_offshore:    Bool. Default False. Whether to add additional rows
                         extending the hierarchy offshore to OSPAR areas (based
                         on data hosted on GitHub)

    Returns
        (Geo)Dataframe. Copy of 'df' with two new columns added. The first is
        simply 'regine_col' cast to upper case; the second is named 'regine_down'
        and contains the regine ID of the next catchment downstream.
    """
    assert regine_col in df.columns, f"Column '{regine_col}' not found in 'df'."
    for kwarg in [order_coastal, nan_to_vass, add_offshore]:
        assert isinstance(
            kwarg, bool
        ), "Boolean keyword argument must be either True or False."
    if order_coastal:
        assert (
            land_to_vass is False
        ), "'land_to_vass' must be False when 'order_coastal' is True."
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
    res_csv = f"https://raw.githubusercontent.com/NIVANorge/teotil3/main/data/lake_residence_times_{dtm_res}m_dem.csv"
    df = pd.read_csv(res_csv)

    # Vollenweider parameters for individual lakes
    # par_name: (sigma, n)
    voll_dict = {
        "orig-totp": (1, 0.5),
    }
    for par in voll_dict.keys():
        sigma, n = voll_dict[par]
        df = calculate_lake_retention_vollenweider(
            df, "res_time_yr", par, sigma=1, n=0.5
        )

    # Non-Vollenweider params for individual lakes
    # Original ret_n assumed to be 0.2*ret_p
    df["ret_orig-totn"] = 0.2 * df["ret_orig-totp"]
    df["trans_orig-totn"] = 1 - df["ret_orig-totn"]

    # Aggregate to regine level
    pars = [col[6:] for col in df.columns if col.startswith("trans_")]
    reg_df = calculate_regine_retention(df, regine_col=regine_col, pars=pars)

    reg_gdf = reg_gdf.merge(reg_df, on=regine_col, how="left")

    for col in reg_gdf.columns:
        if col.startswith("ret_"):
            reg_gdf[col].fillna(0, inplace=True)
        elif col.startswith("trans_"):
            reg_gdf[col].fillna(1, inplace=True)
        else:
            pass

    return reg_gdf
