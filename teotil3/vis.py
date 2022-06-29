import contextily as cx
import geopandas as gpd
import graphviz
import networkx as nx
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
    g, catch_id, cat_gdf, id_col="regine", include_connected=False, direct="up"
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

    Returns
        Tuple (gdf, matplotlib_axis), where 'gdf' is a geodataframe containing
        the catchment polygon of interest and 'matplotlib_axis' shows the polygon
        plotted on a simple basemap.
    """
    assert direct in ("up", "down"), "'direct' must be either 'up' or 'down'"
    assert isinstance(
        include_connected, bool
    ), "'include_connected' must be either True or False."

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
        source=cx.providers.OpenStreetMap.Mapnik,
    )
    ax.set_axis_off()

    return (gdf, ax)
