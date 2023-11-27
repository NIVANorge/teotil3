import contextily as cx
import folium
import geopandas as gpd
import graphviz
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from folium import IFrame
from sqlalchemy import text


def plot_network(
    g, catch_id, direct="down", stat="local", quant="a_cat_land_km2", image_path=None
):
    """
    Create schematic diagram upstream or downstream of a specified node.

    Args
        g: NetworkX graph object returned by other TEOTIL functions
        catch_id: Str. Catchment ID of interest
        direct: Str. 'up' or 'down'. Direction to trace network. Defaults to 'down'
        stat: Str. 'local' or 'accum'. Type of results to display. Defaults to 'local'
        quant: Str. Any of the returned result types. Defaults to 'a_cat_land_km2'
        image_path: Str. Save the network to disk. The file extension must be '.pdf' or '.png'.
            Defaults to None.

    Returns
        Graphviz image object.

    Raises
        ValueError: If 'direct' is not 'up' or 'down'.
        ValueError: If 'stat' is not 'local' or 'accum'.
        ValueError: If 'image_path' does not end in '.png' or '.pdf'.
    """
    if direct not in ("up", "down"):
        raise ValueError("'direct' must be either 'up' or 'down'")
    if stat not in ("local", "accum"):
        raise ValueError("'stat' must be either 'local' or 'accum'")
    if image_path and image_path[-4:] not in (".png", ".pdf"):
        raise ValueError("'image_path' must end in '.png' or '.pdf'.")

    # Parse direction
    if direct == "down":
        g2 = nx.dfs_tree(g, catch_id)
    else:
        g2 = nx.dfs_tree(g.reverse(), catch_id).reverse()

    # Update labels with 'quant'
    for nd in list(nx.topological_sort(g2)):
        g2.nodes[nd]["label"] = "%s\n(%.2f)" % (nd, g.nodes[nd][stat][quant])

    # Draw
    res = nx.nx_agraph.to_agraph(g2)
    res.layout("dot")
    res = graphviz.Source(res.to_string())

    if image_path:
        res.render(outfile=image_path)

    return res


def plot_catchment(
    g,
    catch_id,
    cat_gdf,
    id_col="regine",
    include_connected=False,
    direct="up",
    basemap="kartverket",
):
    """
    Create a map showing the catchment boundary overlaid on a simple basemap.

    Args
        g: NetworkX graph object returned by other TEOTIL functions
        catch_id: Str. Catchment ID of interest
        cat_gdf: Geodataframe. Must contain catchment polygons with the same ID as used by 'g'
        id_col: Str. Name of column in 'cat_gdf' containing catchment IDs. Defaults to 'regine'
        include_connected: Bool. Whether to identify all upstream polygons and dissolve them to
            identify the whole catchment. Defaults to False
        direct: Str. 'up' or 'down'. Direction in which to trace network and merge catchments.
            Ignored unless 'include_connected' is True. Defaults to 'up'.
        basemap: Str or Obj. If str, must be either 'kartverket' or a valid WMS tile URL. If
            obj, must be a valid contextily provider object. Defaults to 'kartverket'.

    Returns
        A tuple containing a geodataframe with the catchment polygon of interest and a matplotlib
        axis showing the polygon plotted on a simple basemap.

    Raises
        ValueError: If 'direct' is not 'up' or 'down'.
        ValueError: If 'include_connected' is not a boolean.
    """
    if direct not in ("up", "down"):
        raise ValueError("'direct' must be either 'up' or 'down'.")
    if not isinstance(include_connected, bool):
        raise ValueError("'include_connected' must be either True or False.")

    if basemap == "kartverket":
        basemap = "https://cache.kartverket.no/topo4/v1/gmaps/{z}/{x}/{y}.png"

    if include_connected:
        if direct == "down":
            g2 = nx.dfs_tree(g, catch_id)
        else:
            g2 = nx.dfs_tree(g.reverse(), catch_id).reverse()

        node_list = list(g2.nodes)
    else:
        node_list = [catch_id]

    # Dissolve
    gdf = cat_gdf.query(f"{id_col} in @node_list").copy()
    gdf["dissolve_field"] = 1
    gdf = gdf.dissolve(by="dissolve_field").reset_index(drop=True)

    # Plot
    ax = gdf.plot(figsize=(10, 10), facecolor="none", edgecolor="red", linewidth=2)
    cx.add_basemap(
        ax,
        crs=gdf.crs.to_string(),
        attribution=False,
        source=basemap,
    )
    ax.set_axis_off()

    return (gdf, ax)


def choropleth_map(
    g,
    cat_gdf,
    id_col="regine",
    stat="accum",
    quant="q_m3/s",
    trans="none",
    cmap="viridis",
    scheme="quantiles",
    n_classes=10,
    figsize=(8, 12),
    plot_path=None,
    ax=None,
    legend_loc="upper left",
):
    """
    Display a map of the regine catchments, coloured according to the quantity specified.

    Args
        g: NetworkX graph object returned by teo.run_model()
        cat_gdf: Geodataframe. Polygons representing catchments
        id_col: Str. Name of column identifying unique catchments in 'cat_gdf' and nodes in 'g'.
            Defaults to 'regine'
        stat: Str. 'local' or 'accum'. Type of results to display. Defaults to 'accum'
        quant: Str. Any of the returned result types. Defaults to 'q_m3/s'
        trans: Str. One of ['none', 'log', 'sqrt']. Whether to transform 'quant' before plotting.
            Defaults to 'none'
        cmap: Str. Valid matplotlib colourmap. Defaults to 'viridis'
        scheme: Str. Valid map classify scheme name. Defaults to 'quantiles'
        n_classes: Int. Number of classes in 'scheme'. Defaults to 10
        figsize: Tuple. Figure (width, height) in inches. Ignored if 'ax' is specified. Defaults
            to (8, 12)
        plot_path: Str. Path to which plot will be saved. Defaults to None
        ax: Obj. Existing axis on which to draw the plot, if desired. Defaults to None
        legend_loc: Str. Position for legend. Defaults to 'upper left'

    Returns
        matplotlib.axes._subplots.AxesSubplot: A matplotlib axis showing the choropleth map.

    Raises
        ValueError: If 'id_col' not found in 'cat_gdf'.
        ValueError: If 'stat' is not 'local' or 'accum'.
        ValueError: If 'trans' is not one of ['none', 'log', 'sqrt'].
    """
    if id_col not in cat_gdf.columns:
        raise ValueError("'id_col' not found in 'cat_gdf'.")
    if stat not in ["local", "accum"]:
        raise ValueError("'stat' must be either 'local' or 'accum'.")
    if trans not in ["none", "log", "sqrt"]:
        raise ValueError("'trans' must be one of ['none', 'log', 'sqrt'].")

    # Extract data of interest from graph
    cat_list = []
    val_list = []
    for nd in list(nx.topological_sort(g))[:-1]:
        cat_list.append(g.nodes[nd]["local"][id_col])
        val_list.append(g.nodes[nd][stat][quant])

    df = pd.DataFrame(data={quant: val_list, id_col: cat_list})

    # Map title
    stat = stat.capitalize()
    tit = quant.split("_")
    name = " ".join(tit[:-1]).capitalize()
    unit = tit[-1]

    # Transform if necessary
    if trans == "none":
        tit = f"{stat} {name} ({unit})"
    elif trans == "log":
        tit = f"log[{stat} {name} ({unit})]"
        df[quant] = np.log10(df[quant])
    elif trans == "sqrt":
        tit = f"sqrt[{stat} {name} ({unit})]"
        df[quant] = df[quant] ** 0.5

    # Join catchments
    cat_gdf = cat_gdf[[id_col, cat_gdf.geometry.name]].copy()
    gdf = pd.merge(cat_gdf, df, how="left", on=id_col)
    gdf.dropna(subset=cat_gdf.geometry.name, inplace=True)

    # Plot
    ax = gdf.plot(
        column=quant,
        legend=True,
        scheme=scheme,
        edgecolor="face",
        figsize=figsize,
        cmap=cmap,
        legend_kwds={"loc": legend_loc},
        classification_kwds={"k": n_classes},
        ax=ax,
    )
    ax.set_title(tit, fontsize=20)
    ax.axis("off")

    # Save
    if plot_path:
        plt.savefig(plot_path, dpi=300)

    return ax


def point_sources_map(year, eng, loc_type="outlet"):
    """Create an interactive Leaflet map using Folium showing point sources for the specified
    year.

    Args
        year: Int. Year of interest
        eng: Obj. Active database connection object connected to PostGIS
        loc_type: Str. Default 'outlet'. Either 'outlet' or 'site'. Type of locations to return.

    Returns
        Folium map.

    Raises
        ValueError: If 'loc_type' is not 'outlet' or 'site'.
    """
    if loc_type not in ["outlet", "site"]:
        raise ValueError("'loc_type' must be either 'outlet' or 'site'.")

    # Get point data from database
    sql = text(
        "SELECT * FROM teotil3.point_source_locations "
        "WHERE year = :year "
        f"AND {loc_type}_geom IS NOT NULL "
        f"AND NOT ST_IsEmpty({loc_type}_geom)"
    )
    gdf = gpd.read_postgis(
        sql,
        eng,
        geom_col=f"{loc_type}_geom",
        params={"year": year},
    )
    gdf = gdf.to_crs("epsg:4326")
    gdf["latitude"] = gdf.geometry.y
    gdf["longitude"] = gdf.geometry.x

    # Interactive map
    m = folium.Map(location=[64, 12], zoom_start=3)

    # Define a colors for categories
    colors = {
        "Large wastewater": "red",
        "Small wastewater": "yellow",
        "Industry": "blue",
        "Aquaculture": "green",
    }

    # Create a feature group for each category
    feature_groups = {
        category: folium.FeatureGroup(name=category) for category in colors.keys()
    }

    # Add points to the map
    for idx, row in gdf.iterrows():
        lon, lat = row["longitude"], row["latitude"]
        category = row["sector"]
        site_id = row["site_id"]
        name = row["name"]
        year = row["year"]

        color = colors.get(category, "gray")  # Grey is the default color

        # Popup
        html = f"""
            <table style="width:100%">
                <tr>
                    <th>Site ID:</th>
                    <td>{site_id}</td>
                </tr>
                <tr>
                    <th>Name:</th>
                    <td>{name}</td>
                </tr>
                <tr>
                    <th>Year:</th>
                    <td>{year}</td>
                </tr>
            </table>
        """
        iframe = IFrame(html, width=300, height=100)
        popup = folium.Popup(iframe, max_width=300)

        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            popup=popup,
        ).add_to(feature_groups[category])

    for feature_group in feature_groups.values():
        feature_group.add_to(m)

    folium.LayerControl().add_to(m)

    return m
