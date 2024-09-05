# ------------------------------------------------------------------------------
# Copyright (C) 2012-2017 Guillaume Sagnol
# Copyright (C) 2018-2019 Maximilian Stahlberg
#
# This file is part of PICOS.
#
# PICOS is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# PICOS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.
# ------------------------------------------------------------------------------

"""Implementation of :class:`FlowConstraint`."""

from collections import namedtuple

from ..apidoc import api_end, api_start
from ..caching import cached_property
from .constraint import Constraint, ConstraintConversion

_API_START = api_start(globals())
# -------------------------------


class FlowConstraint(Constraint):
    """Network flow constraint.

    .. note ::
        Unlike other :class:`~.constraint.Constraint` implementations, this one
        is instanciated by the user (via a wrapper function), so it is raising
        exceptions instead of making assertions.
    """

    class Conversion(ConstraintConversion):
        """Network flow constraint conversion."""

        @classmethod
        def predict(cls, subtype, options):
            """Implement :meth:`~.constraint.ConstraintConversion.predict`."""
            from ..expressions import RealVariable
            from . import AffineConstraint

            numNodes, numEdges, flowCons, hasCapacities = subtype
            V, E = numNodes, numEdges

            # Capacity and non-negativity constraints.
            yield ("con", AffineConstraint.make_type(dim=1, eq=False),
                2*E if hasCapacities else E)

            if len(flowCons) == 1:
                yield ("con", AffineConstraint.make_type(dim=1, eq=True), V - 1)
            else:
                for k in flowCons:
                    yield ("var", RealVariable.make_var_type(dim=1, bnd=0), E)

                    for addition in cls.predict((V, E, (k,), False), options):
                        yield addition

                yield ("con", AffineConstraint.make_type(dim=1, eq=True), E)

        @classmethod
        def convert(cls, con, options):
            """Implement :meth:`~.constraint.ConstraintConversion.convert`."""
            from ..modeling import Problem
            P = Problem()
            return cls._convert(con, P)  # Allow for recursion.

        @classmethod
        def _convert(cls, con, P):
            from ..expressions import new_param, sum

            G          = con.graph
            f          = con.f
            source     = con.source
            sink       = con.sink
            flow_value = con.flow_value
            capacity   = con.capacity
            graphName  = con.graphName

            # Add capacity constraints.
            if capacity is not None:
                c = {}
                for v, w, data in G.edges(data=True):
                    c[(v, w)] = data[capacity]
                c = new_param('c', c)

                P.add_list_of_constraints([f[e] <= c[e] for e in G.edges()])

            # Add non-negativity constraints.
            P.add_list_of_constraints([f[e] >= 0 for e in G.edges()])

            # One source, one sink.
            if not isinstance(source, list) and not isinstance(sink, list):
                # Add flow conversation constrains.
                P.add_list_of_constraints([
                    sum([f[p, i] for p in G.predecessors(i)])
                    == sum([f[i, j] for j in G.successors(i)])
                    for i in G.nodes() if i != sink and i != source])

                # Source flow at S
                P.add_constraint(
                    sum([f[p, source] for p in G.predecessors(source)])
                    + flow_value ==
                    sum([f[source, j] for j in G.successors(source)]))

            # One source, multiple sinks.
            elif not isinstance(source, list):
                # Add flow conversation constrains.
                P.add_list_of_constraints([
                    sum([f[p, i] for p in G.predecessors(i)])
                    == sum([f[i, j] for j in G.successors(i)])
                    for i in G.nodes() if i not in sink and i != source])

                for k in range(0, len(sink)):
                    # Sink flow at T
                    P.add_constraint(
                        sum([f[p, sink[k]] for p in G.predecessors(sink[k])])
                        == sum([f[sink[k], j] for j in G.successors(sink[k])])
                        + flow_value[k])

            # Multiple sources, one sink.
            elif not isinstance(sink, list):
                # Add flow conversation constrains.
                P.add_list_of_constraints([
                    sum([f[p, i] for p in G.predecessors(i)])
                    == sum([f[i, j] for j in G.successors(i)])
                    for i in G.nodes() if i not in source and i != sink])

                for k in range(0, len(source)):
                    # Source flow at T
                    P.add_constraint(sum(
                        [f[p, source[k]] for p in G.predecessors(source[k])])
                        + flow_value[k] ==
                        sum([f[source[k], j] for j in G.successors(source[k])]))

            # Multiple sources, multiple sinks.
            # TODO: Recursion adds redundant non-negativity constraints.
            elif isinstance(sink, list) and isinstance(source, list):
                SS = list(set(source))
                TT = list(set(sink))

                if len(SS) <= len(TT):
                    ftmp = {}
                    for s in SS:
                        ftmp[s] = {}
                        sinks_from_s = [
                            t for (i, t) in enumerate(sink) if source[i] == s]
                        values_from_s = [v for (i, v)
                            in enumerate(flow_value) if source[i] == s]

                        for e in G.edges():
                            ftmp[s][e] = P.add_variable(
                                '__f[{0}][{1}]'.format(s, e), 1)

                        # Immediately convert another FlowConstraint so that the
                        # reformulation created from this conversion doesn't
                        # need to be run twice in a row.
                        cls._convert(cls(
                            G, ftmp[s], source=s, sink=sinks_from_s,
                            flow_value=values_from_s, graphName=graphName), P)

                    P.add_list_of_constraints([
                        f[e] == sum([ftmp[s][e] for s in SS])
                        for e in G.edges()])
                else:
                    ftmp = {}
                    for t in TT:
                        ftmp[t] = {}
                        sources_to_t = [
                            s for (i, s) in enumerate(source) if sink[i] == t]
                        values_to_t = [v for (i, v) in enumerate(flow_value)
                            if sink[i] == t]

                        for e in G.edges():
                            ftmp[t][e] = P.add_variable(
                                '__f[{0}][{1}]'.format(t, e), 1)

                        # Immediately convert another FlowConstraint so that the
                        # reformulation created from this conversion doesn't
                        # need to be run twice in a row.
                        cls._convert(cls(
                            G, ftmp[t], source=sources_to_t, sink=t,
                            flow_value=values_to_t, graphName=graphName), P)

                    P.add_list_of_constraints([
                        f[e] == sum([ftmp[t][e] for t in TT])
                        for e in G.edges()])

            else:
                assert False, "Dijkstra-IF fallthrough."

            return P

    def __init__(
            self, G, f, source, sink, flow_value, capacity=None, graphName=""):
        """Construct a network flow constraint.

        :param G: A directed graph.
        :type G: `networkx DiGraph <http://networkx.lanl.gov/index.html>`_.

        :param dict f: A dictionary of variables indexed by the edges of ``G``.

        :param source: Either a node of ``G`` or a list of nodes in case of a
            multi-source flow.

        :param sink: Either a node of ``G`` or a list of nodes in case of a
            multi-sink flow.

        :param flow_value: The value of the flow, or a list of values in case of
            a single-source/multi-sink flow. In the latter case, the values
            represent the demands of each sink (resp. of each source for a
            multi-source/single-sink flow). The values can be either constants
            or :class:`~picos.expressions.AffineExpression`.

        :param capacity: Either ``None`` or a string. If this is a string, it
            indicates the key of the edge dictionaries of ``G`` that is used for
            the capacity of the links. Otherwise, edges have an unbounded
            capacity.

        :param str graphName: Name of the graph as used in the string
            representation of the constraint.
        """
        if len(f) != len(G.edges()):
            raise ValueError(
                "The number of variables does not match the number of edges.")

        if isinstance(sink, list) and len(sink) == 1:
            source = source[0]

        if isinstance(sink, list) and len(sink) == 1:
            sink = sink[0]

        if isinstance(source, list) and len(source) != len(flow_value):
            raise ValueError("The number of sources does not match the number "
                "of flow values.")

        if isinstance(sink, list) and len(sink) != len(flow_value):
            raise ValueError("The number of sinks does not match the number "
                "of flow values.")

        if isinstance(source, list) and isinstance(sink, list) \
        and len(sink) != len(source):
            raise ValueError("The number of sinks does not match the number "
                "of sources.")

        self.graph      = G
        self.f          = f
        self.source     = source
        self.sink       = sink
        self.flow_value = flow_value
        self.capacity   = capacity
        self.graphName  = graphName

        # Build the string description.
        if isinstance(source, list):
            sourceStr = "(" + ",".join(source) + ")"
        else:
            sourceStr = str(source)

        if isinstance(sink, list):
            sinkStr = "(" + ",".join(sink) + ")"
        else:
            sinkStr = str(sink)

        if isinstance(flow_value, list):
            valueStr = "values " + ", ".join([v.string if hasattr(v, "string")
                else str(v) for v in flow_value])
        else:
            valueStr = "value " + flow_value.string \
                if hasattr(flow_value, "string") else str(flow_value)

        self.comment = "{}{}-{}-flow{} has {}.".format(
            "Feasible " if capacity is not None else "", sourceStr, sinkStr,
            " in {}".format(graphName) if graphName else "", valueStr)

        super(FlowConstraint, self).__init__("Flow")

    Subtype = namedtuple("Subtype",
        ("lenV", "lenE", "flowCons", "hasCapacities"))

    @classmethod
    def _cost(cls, subtype):
        return subtype.lenE  # Somewhat arbitrary.

    def _subtype(self):
        s = len(self.source) if isinstance(self.source, list) else 1
        t = len(self.sink)   if isinstance(self.sink, list)   else 1
        V = len(self.graph.nodes())
        E = len(self.graph.edges())

        if s == 1 or t == 1:
            flowCons = (max(s, t),)
        else:
            flowCons = []
            S, T = self.source, self.sink
            A, B = set(S), set(T)
            if len(A) > len(B):
                S, T = T, S
                A, B = B, A
            for s in A:
                flowCons.append(S.count(s))
            flowCons = tuple(flowCons)

        assert sum(flowCons) == max(s, t)

        return self.Subtype(V, E, flowCons, self.capacity is not None)

    def _expression_names(self):
        return
        yield

    # HACK: The variables are not stored in named expressions but in a
    #       dictionary. To make prediction work, they must be considered part
    #       of the problem that the flow constraint was added to.
    @cached_property
    def mutables(self):  # noqa
        return frozenset(self.f.values())

    # HACK: See above.
    def replace_mutables(self, mapping):  # noqa
        f = {key: mapping[var] for key, var in self.f.items()}
        return FlowConstraint(self.graph, f, self.source, self.sink,
            self.flow_value, self.capacity, self.graphName)

    def _str(self):
        return self.comment

    def _get_slack(self):
        raise NotImplementedError

    def draw(self):
        """Draw the graph."""
        G = self.graph

        import networkx as nx
        import matplotlib.pyplot as plt

        pos = nx.spring_layout(G)
        edge_labels = dict([((u, v,), d["capacity"])
                            for u, v, d in G.edges(data=True)])
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        nx.draw(G, pos)
        plt.show()


# --------------------------------------
__all__ = api_end(_API_START, globals())
