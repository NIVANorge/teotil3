import contextily as cx
import geopandas as gpd
import graphviz
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def plot_network(
    g, catch_id, direct="down", stat="local", quant="a_cat_land_km2", image_path=None
):
    """Create schematic diagram upstream or downstream of a specified node.

    Args
        g:          NetworkX graph object returned by other TEOTIL functions (e.g.
                    'teo.model.build_graph' or 'teo.model.run_model')
        catch_id:   Str. Catchment ID of interest
        direct:     Str. Default 'down'. 'up' or 'down'. Direction to trace network
        stat:       Str. Default 'local'. 'local' or 'accum'. Type of results to
                    display
        quant:      Str. Default 'a_cat_land_km2'. Any of the returned result types
        image_path: Str. Optional. Default None. Save the network to disk. The file
                    extension must be '.pdf' or '.png'

    Returns
        Graphviz image object.
    """
    assert direct in ("up", "down"), "'direct' must be either 'up' or 'down'"
    assert stat in ("local", "accum"), "'stat' must be either 'local' or 'accum'"
    if image_path:
        assert image_path[-4:] in (
            ".png",
            ".pdf",
        ), "'image_path' must end in '.png' or '.pdf'."

    # Parse direction
    if direct == "down":
        # Get sub-tree
        g2 = nx.dfs_tree(g, catch_id)

        # Update labels with 'quant'
        for nd in list(nx.topological_sort(g2))[:-1]:
            g2.nodes[nd]["label"] = "%s\n(%.2f)" % (nd, g.nodes[nd][stat][quant])

    else:
        # 'direct' = 'up'
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
    """Create a map showing the catchment boundary overlaid on a simple basemap.

    Args
        g:                 NetworkX graph object returned by other TEOTIL
                           functions (e.g. 'teo.model.build_graph' or
                           'teo.model.run_model')
        catch_id:          Str. Catchment ID of interest
        cat_gdf:           Geodataframe. Must contain catchment polygons with the
                           same ID as used by 'g'
        id_col:            Str. Default 'regine'. Name of column in 'cat_gdf'
                           containing catchment IDs
        include_connected: Bool. Default False. Whether to identify all upstream
                           polygons and dissolve them to identify the whole
                           catchment
        direct:            Str. Default 'up'. 'up' or 'down'. Direction in which
                           to trace network and merge catchments. Ignored unless
                           'include_connected' is True
        basemap:           Str or Obj. Default 'kartverket'. If str, must be either
                           'kartverket' or a valid WMS tile URL (e.g.
                           https://cache.kartverket.no/topo4/v1/gmaps/{z}/{x}/{y}.png)
                           that will be passed as the 'source' argument to
                           contextily's 'add_basemap' function. If obj, must be a
                           valid contextily provider object (see
                           https://contextily.readthedocs.io/en/latest/intro_guide.html#Providers
                           for details).

    Returns
        Tuple (gdf, matplotlib_axis), where 'gdf' is a geodataframe containing
        the catchment polygon of interest and 'matplotlib_axis' shows the polygon
        plotted on a simple basemap.
    """
    assert direct in ("up", "down"), "'direct' must be either 'up' or 'down'"
    assert isinstance(
        include_connected, bool
    ), "'include_connected' must be either True or False."

    if basemap == "kartverket":
        basemap = "https://cache.kartverket.no/topo4/v1/gmaps/{z}/{x}/{y}.png"

    if include_connected:
        if direct == "down":
            # Get sub-tree
            g2 = nx.dfs_tree(g, catch_id)
        else:
            # 'direct' = 'up'
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
    """Display a map of the regine catchments, coloured according
    to the quantity specified.
    Args:
        g           NetworkX graph object returned by teo.run_model()
        cat_gdf:    Geodataframe. Polygons representing catchments
        id_col:     Str. Name of column identifying unique catchments in 'cat_gdf' and
                    nodes in 'g'
        stat:       Str. 'local' or 'accum'. Type of results to display
        quant:      Str. Any of the returned result types
        trans:      Str. One of ['none', 'log', 'sqrt']. Whether to transform 'quant'
                    before plotting
        cmap:       Str. Valid matplotlib colourmap
        scheme:     Str. Valid map classify scheme name. See here for details:
                        https://github.com/pysal/mapclassify
        n_classes:  Int. Number of classes in 'scheme'. Corresponds to parameter 'k' here:
                        https://github.com/pysal/mapclassify
        figsize:    Tuple. Figure (width, height) in inches. Ignored if 'ax' is specified
        plot_path:  Raw Str. Optional. Path to which plot will be saved
        ax:         Matplotlib axis or None. Default None. Existing axis on which to
                    draw the plot, if desired
        legend_loc: Str. Deafault 'upper left'. Postition for legend.
    Returns:
        None
    """
    assert id_col in cat_gdf.columns, "'id_col' not found in 'cat_gdf'."
    assert stat in ["local", "accum"], "'stat' must be either 'local' or 'accum'."

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
    else:
        raise ValueError("'trans' must be one of ['none', 'log', 'sqrt'].")

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
