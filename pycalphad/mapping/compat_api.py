from pycalphad import Database, variables as v
from pycalphad.mapping.mapper import Mapper
from pycalphad.mapping.starting_points import automatic_starting_points_from_axis_limits, _generate_fixed_variable_conditions
import matplotlib.pyplot as plt
from pycalphad.mapping.plotting import plot_map
import numpy as np
from pycalphad.core.utils import unpack_components, filter_phases
from pycalphad.plot.utils import phase_legend
from pycalphad.plot import triangular  # register triangular projection
from .primitives import STATEVARS
import copy


def binplot(database, components, phases, conditions, plot_kwargs=None, **map_kwargs):
    """
    Calculate the binary isobaric phase diagram.

    This function is a convenience wrapper around map_binary() and plot_boundaries()

    Parameters
    ----------
    database : Database
        Thermodynamic database containing the relevant parameters.
    components : Sequence[str]
        Names of components to consider in the calculation.
    phases : Sequence[str]
        Names of phases to consider in the calculation.
    conditions : Mapping[v.StateVariable, Union[float, Tuple[float, float, float]]]
        Maps StateVariables to values and/or iterables of values.
        For binplot only one changing composition and one potential coordinate each is supported.
    eq_kwargs : dict, optional
        Keyword arguments to use in equilibrium() within map_binary(). If
        eq_kwargs is defined in map_kwargs, this argument takes precedence.
    map_kwargs : dict, optional
        Additional keyword arguments to map_binary().
    plot_kwargs : dict, optional
        Keyword arguments to plot_boundaries().

    Returns
    -------
    Axes
        Matplotlib Axes of the phase diagram

    Examples
    --------
    None yet.
    """
    indep_comps = [key for key, value in conditions.items() if isinstance(key, v.MoleFraction) and len(np.atleast_1d(value)) > 1]
    indep_pots = [key for key, value in conditions.items() if (type(key) is v.StateVariable) and len(np.atleast_1d(value)) > 1]
    if (len(indep_comps) != 1) or (len(indep_pots) != 1):
        raise ValueError('binplot() requires exactly one composition coordinate and one potential coordinate')
    # TODO: try to give full backwards compatible support for plot_kwargs and map_kwargs
    # remaining plot_kwargs from pycalphad.plot.binary.plot.plot_boundaries:
    # tieline_color=(0, 1, 0, 1)
    # remaining map_kwargs from pycalphad.plot.binary.map.map_binary:
    # calc_kwargs=None
    plot_kwargs = plot_kwargs if plot_kwargs is not None else dict()

    # TODO: filtering phases should be done by the mapper
    phases = filter_phases(database, unpack_components(database, components), phases)

    eq_kwargs = map_kwargs.get("eq_kwargs", {})  # only used for start points, not in the mapping currently
    mapper = Mapper(database, components, phases, conditions)
    mapper.strategy.verbose = map_kwargs.get('verbose', False)
    start_points, start_dir = automatic_starting_points_from_axis_limits(database, components, phases, conditions, **eq_kwargs)
    mapper.strategy.add_starting_points_with_axes(start_points, start_dir)
    mapper.do_map()

    ax = plot_kwargs.get("ax")
    if ax is None:
        ax = plt.figure().gca()
    legend_generator = plot_kwargs.get('legend_generator', phase_legend)
    plot_map(mapper, tielines=plot_kwargs.get("tielines", True), ax=ax, legend_generator=legend_generator)
    ax.grid(plot_kwargs.get("gridlines", False))

    return ax


def ternplot(dbf, comps, phases, conds, x=None, y=None, eq_kwargs=None, **plot_kwargs):
    """
    Calculate the ternary isothermal, isobaric phase diagram.
    This function is a convenience wrapper around equilibrium() and eqplot().

    Parameters
    ----------
    dbf : Database
        Thermodynamic database containing the relevant parameters.
    comps : Sequence[str]
        Names of components to consider in the calculation.
    phases : Sequence[str]
        Names of phases to consider in the calculation.
    conds : Mapping[v.StateVariable, Union[float, Tuple[float, float, float]]]
        Maps StateVariables to values and/or iterables of values.
        For ternplot only one changing composition and one potential coordinate each is supported.
    x : v.MoleFraction
        instance of a pycalphad.variables.composition to plot on the x-axis.
        Must correspond to an independent condition.
    y : v.MoleFraction
        instance of a pycalphad.variables.composition to plot on the y-axis.
        Must correspond to an independent condition.
    eq_kwargs : optional
        Keyword arguments to equilibrium().
    plot_kwargs : optional
        Keyword arguments to eqplot().

    Returns
    -------
    A phase diagram as a figure.

    Examples
    --------
    None yet.
    """
    # remaining plot_kwargs from pycalphad.plot.eqplot
    # x=None, y=None, z=None, tieline_color=(0, 1, 0, 1), tie_triangle_color=(1, 0, 0, 1), **kwargs
    # kwargs passed ot ax.scatter
    indep_comps = [key for key, value in conds.items() if isinstance(key, v.MoleFraction) and len(np.atleast_1d(value)) > 1]
    indep_pots = [key for key, value in conds.items() if (type(key) is v.StateVariable) and len(np.atleast_1d(value)) > 1]
    if (len(indep_comps) != 2) or (len(indep_pots) != 0):
        raise ValueError('ternplot() requires exactly two composition coordinates')

    phases = filter_phases(dbf, unpack_components(dbf, comps), phases)
    mapper = Mapper(dbf, comps, phases, conds)
    mapper.strategy.verbose = eq_kwargs.get('verbose', False)
    start_points, start_dir = automatic_starting_points_from_axis_limits(dbf, comps, phases, conds)
    mapper.strategy.add_starting_points_with_axes(start_points, start_dir)
    mapper.do_map()

    ax = plot_kwargs.get("ax")
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': "triangular"})

    legend_generator = plot_kwargs.get('legend_generator', phase_legend)
    plot_map(mapper, tielines=plot_kwargs.get("tielines", True), ax=ax, legend_generator=legend_generator)

    return ax

def isoplethplot(database, components, phases, conditions, plot_kwargs=None, **map_kwargs):
    """
    Calculates an isopleth phase diagram.
    For now, we'll define isopleths as having 1 potential condition and 1 non-potential condition
    TODO: 

    This function is a convenience wrapper around map_binary() and plot_boundaries()

    Parameters
    ----------
    database : Database
        Thermodynamic database containing the relevant parameters.
    components : Sequence[str]
        Names of components to consider in the calculation.
    phases : Sequence[str]
        Names of phases to consider in the calculation.
    conditions : Mapping[v.StateVariable, Union[float, Tuple[float, float, float]]]
        Maps StateVariables to values and/or iterables of values.
        For binplot only one changing composition and one potential coordinate each is supported.
    eq_kwargs : dict, optional
        Keyword arguments to use in equilibrium() within map_binary(). If
        eq_kwargs is defined in map_kwargs, this argument takes precedence.
    map_kwargs : dict, optional
        Additional keyword arguments to map_binary().
    plot_kwargs : dict, optional
        Keyword arguments to plot_boundaries().

    Returns
    -------
    Axes
        Matplotlib Axes of the phase diagram

    Examples
    --------
    None yet.
    """
    indep_comps = [key for key, value in conditions.items() if isinstance(key, v.MoleFraction) and len(np.atleast_1d(value)) > 1]
    indep_pots = [key for key, value in conditions.items() if (type(key) is v.StateVariable) and len(np.atleast_1d(value)) > 1]
    if (len(indep_comps) != 1) or (len(indep_pots) != 1):
        raise ValueError('isoplethplot() requires exactly one composition coordinate and one potential coordinate')
    # TODO: try to give full backwards compatible support for plot_kwargs and map_kwargs
    # remaining plot_kwargs from pycalphad.plot.binary.plot.plot_boundaries:
    # tieline_color=(0, 1, 0, 1)
    # remaining map_kwargs from pycalphad.plot.binary.map.map_binary:
    # calc_kwargs=None
    plot_kwargs = plot_kwargs if plot_kwargs is not None else dict()

    # TODO: filtering phases should be done by the mapper
    phases = filter_phases(database, unpack_components(database, components), phases)

    eq_kwargs = map_kwargs.get("eq_kwargs", {})  # only used for start points, not in the mapping currently

    comp_offset = sum([conditions[v] for v in conditions if (not isinstance(conditions[v], tuple) and v not in STATEVARS)])
    conditions[indep_comps[0]] = (conditions[indep_comps[0]][0], np.amin([conditions[indep_comps[0]][1], 1-comp_offset]), conditions[indep_comps[0]][2])
    axis_cond_keys = [key for key, val in conditions.items() if isinstance(val, tuple)]
    
    mapper = Mapper(database, components, phases, conditions)
    
    #Do step mapping along edges of variables (starting point for step mapping will be halfway point)
    half_conds = {axis_var: (conditions[axis_var][0]+conditions[axis_var][1])/2 for axis_var in axis_cond_keys}
    step_conditions = []
    for axis_var in axis_cond_keys:
        edge_conditions = {key:val for key,val in conditions.items()}
        edge_conditions[axis_var] = half_conds[axis_var]

        other_var = axis_cond_keys[1-axis_cond_keys.index(axis_var)]
        if axis_var in STATEVARS:
            edge_conditions[other_var] = np.amax([1e-2, conditions[other_var][0]])
            step_conditions.append((copy.copy(edge_conditions), axis_var))
            edge_conditions[other_var] = np.amin([conditions[other_var][1], 1-comp_offset-1e-2])
            step_conditions.append((copy.copy(edge_conditions), axis_var))
        # else:
        #     edge_conditions[other_var] = conditions[other_var][0]
        #     step_conditions.append((copy.copy(edge_conditions), axis_var))
        #     edge_conditions[other_var] = conditions[other_var][1]
        #     step_conditions.append((copy.copy(edge_conditions), axis_var)) 
    
    mapper.strategy.verbose = map_kwargs.get('verbose', False)
    for sc in step_conditions:
        mapper.strategy.add_nodes_from_conditions(*sc)

    mapper.do_map()

    ax = plot_kwargs.get("ax")
    if ax is None:
        ax = plt.figure().gca()
    legend_generator = plot_kwargs.get('legend_generator', phase_legend)
    plot_map(mapper, ax=ax, legend_generator=legend_generator)
    ax.grid(plot_kwargs.get("gridlines", False))

    return ax

def stepplot(database, components, phases, conditions, plot_kwargs=None, **map_kwargs):
    """
    Calculate the binary isobaric phase diagram.

    This function is a convenience wrapper around map_binary() and plot_boundaries()

    Parameters
    ----------
    database : Database
        Thermodynamic database containing the relevant parameters.
    components : Sequence[str]
        Names of components to consider in the calculation.
    phases : Sequence[str]
        Names of phases to consider in the calculation.
    conditions : Mapping[v.StateVariable, Union[float, Tuple[float, float, float]]]
        Maps StateVariables to values and/or iterables of values.
        For binplot only one changing composition and one potential coordinate each is supported.
    eq_kwargs : dict, optional
        Keyword arguments to use in equilibrium() within map_binary(). If
        eq_kwargs is defined in map_kwargs, this argument takes precedence.
    map_kwargs : dict, optional
        Additional keyword arguments to map_binary().
    plot_kwargs : dict, optional
        Keyword arguments to plot_boundaries().

    Returns
    -------
    Axes
        Matplotlib Axes of the phase diagram

    Examples
    --------
    None yet.
    """
    indep_vars = [key for key, value in conditions.items() if len(np.atleast_1d(value)) > 1]
    if len(indep_vars) != 1:
        raise ValueError('stepplot() requires exactly one coordinate')
    # TODO: try to give full backwards compatible support for plot_kwargs and map_kwargs
    # remaining plot_kwargs from pycalphad.plot.binary.plot.plot_boundaries:
    # tieline_color=(0, 1, 0, 1)
    # remaining map_kwargs from pycalphad.plot.binary.map.map_binary:
    # calc_kwargs=None
    plot_kwargs = plot_kwargs if plot_kwargs is not None else dict()

    # TODO: filtering phases should be done by the mapper
    phases = filter_phases(database, unpack_components(database, components), phases)

    eq_kwargs = map_kwargs.get("eq_kwargs", {})  # only used for start points, not in the mapping currently
    mapper = Mapper(database, components, phases, conditions)
    mapper.strategy.verbose = map_kwargs.get('verbose', False)
    start_conditions = {key:value for key,value in conditions.items()}
    start_conditions[indep_vars[0]] = (conditions[indep_vars[0]][0] + conditions[indep_vars[0]][1]) / 2
    mapper.strategy.add_nodes_from_conditions(start_conditions)
    mapper.do_map()

    ax = plot_kwargs.get("ax")
    if ax is None:
        ax = plt.figure().gca()
    legend_generator = plot_kwargs.get('legend_generator', phase_legend)
    plot_map(mapper, tielines=plot_kwargs.get("tielines", True), ax=ax, legend_generator=legend_generator)
    ax.grid(plot_kwargs.get("gridlines", False))

    ax.set_ylim(bottom=0)

    return ax
