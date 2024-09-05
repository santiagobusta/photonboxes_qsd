# ------------------------------------------------------------------------------
# Copyright (C) 2012-2017 Guillaume Sagnol
# Copyright (C)      2019 Maximilian Stahlberg
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

"""Functions for writing optimization problems to a file."""

# ------------------------------------------------------------------------------
# TODO: This is a temporary solution using outdated code. Ideally, writing to
#       a file should be integrated with the reformulation pipeline e.g. by
#       distinguishing a SolutionStrategy and an ExportStrategy.
# ------------------------------------------------------------------------------

from itertools import chain

import cvxopt
import numpy

from ..apidoc import api_end, api_start
from ..constraints import (
    AffineConstraint,
    LMIConstraint,
    RSOCConstraint,
    SOCConstraint,
)

from ..solvers import CVXOPTSolver, CPLEXSolver, MOSEKSolver, GurobiSolver
from ..expressions import IntegerVariable, BinaryVariable, CONTINUOUS_VARTYPES
from ..expressions.vectorizations import FullVectorization

_API_START = api_start(globals())
# -------------------------------


INFINITY = 1e16  #: A number deemed too large to appear in practice.


def write(picos_problem, filename, writer="picos"):
    r"""Write an optimization problem to a file.

    :param ~picos.Problem P: The problem to write.

    :param str filename: Path and name of the output file. The export format
        is inferred from the file extension. Supported extensions and their
        associated format are:

        * ``'.cbf'`` -- Conic Benchmark Format.

          This format is suitable for optimization problems involving second
          order and/or semidefinite cone constraints. This is a standard
          choice for conic optimization problems. Visit the website of
          `The Conic Benchmark Library <http://cblib.zib.de/>`_ or read
          `A benchmark library for conic mixed-integer and continuous
          optimization
          <http://www.optimization-online.org/DB_HTML/2014/03/4301.html>`_
          by Henrik A. Friberg for more information.

        * ``'.lp'`` -- `LP format
          <http://docs.mosek.com/6.0/pyapi/node022.html>`_.

          This format handles only linear constraints, unless the writer
          ``'cplex'`` is used. In the latter case the extended `CPLEX LP format
          <https://www.ibm.com/support/knowledgecenter/SSSA5P_12.10.0/
          ilog.odms.cplex.help/CPLEX/FileFormats/topics/LP.html>`_
          is used instead.

        * ``'.mps'`` -- `MPS format
          <http://docs.mosek.com/6.0/pyapi/node021.html>`_.

          As the writer, you need to choose one of ``'cplex'``, ``'gurobi'``
          or ``'mosek'``.

        * ``'.opf'`` -- `OPF format
          <http://docs.mosek.com/6.0/pyapi/node023.html>`_.

          As the writer, you need to choose ``'mosek'``.

        * ``'.dat-s'`` -- `Sparse SDPA format
          <http://sdpa.indsys.chuo-u.ac.jp/sdpa/download.html#sdpa>`_.

          This format is suitable for semidefinite programs. Second order
          cone constraints are stored as semidefinite constraints on an
          *arrow shaped* matrix.

    :param str writer: The default ``'picos'`` denotes PICOS' internal
        writer, which can export to *LP*, *CBF*, and *Sparse SDPA* formats.
        If CPLEX, Gurobi or MOSEK is installed, you can choose ``'cplex'``,
        ``'gurobi'``, or ``'mosek'``, respectively, to make use of that
        solver's export function and get access to more formats.

    .. warning::

        For problems involving a symmetric matrix variable :math:`X`
        (typically, semidefinite programs), the expressions involving
        :math:`X` are stored in PICOS as a function of :math:`svec(X)`, the
        symmetric vectorized form of :math:`X` (see `Dattorro, ch.2.2.2.1
        <http://meboo.convexoptimization.com/Meboo.html>`_),
        and are also exported in that form. As a result, using an external
        solver on a problem description file exported by PICOS will also
        yield a solution in this symmetric vectorized form.

        The CBF writer tries to write symmetric variables :math:`X` in the
        section ``PSDVAR`` of the .cbf file. However, this is possible only
        if the constraint :math:`X \succeq 0` appears in the problem, and no
        other LMI involves :math:`X`. If these two conditions are not
        satisfied, then the symmetric vectorization of :math:`X` is used as
        a (free) variable of the section ``VAR`` in the .cbf file, as
        explained in the previous paragraph.

    .. warning::

        This function is severly outdated and may fail or not function as
        advertised.
    """
    # HACK: This method abuses the internal problem representation of CVXOPT.
    # TODO: Add a proper method that transforms the problem into the canonical
    #       form required here, and, if applicable, make also CVXOPT use it.

    from .strategy import NoStrategyFound

    # We first prepare the problem so it becomes exportable
    COMMERCIAL_WRITERS = ['cplex', 'mosek', 'gurobi']
    if writer in COMMERCIAL_WRITERS:
        P = picos_problem.prepared(solver=writer)
    else:
        if picos_problem.continuous:
            # This should ensure that all quadratics have been cast as SOC
            # constraints.
            P = picos_problem.prepared(solver='cvxopt', assume_conic=True)
        else:
            try:
                P = picos_problem.prepared(solver=None)
            except NoStrategyFound as error:
                # FIXME: This will typically happen for integer linear programs
                #        if no IP solver is available.
                raise RuntimeError("You try to export a problem for which no "
                    "solver is available") from error

            assert 'Linear' in P.type, "only LINEAR integer problems can be " \
                "exported with the default writer"

    # TODO: Need proper way to detect exponential cone constraints. This catches
    #       only the Geometric Programs.
    if P.numberLSEConstraints:
        raise NotImplementedError("It is not possible (yet) to export GPs or "
            "problems with Exponential Cone Constraints")

    # automatic extension recognition
    if not (any(filename.endswith(ext) for ext in
                (".mps", ".opf", ".cbf", ".ptf", ".lp", ".dat-s"))):
        if writer == "gurobi":
            if (P.numberConeConstraints + P.numberQuadConstraints) == 0:
                filename += ".lp"
            else:
                filename += ".mps"
        elif writer == "cplex":
            filename += ".lp"
        elif writer == "mosek":
            if (P.numberConeConstraints + P.numberQuadConstraints
                    + P.numberSDPConstraints) == 0:
                filename += ".lp"
            elif P.numberQuadConstraints == 0:
                filename += ".cbf"
            else:
                filename += ".mps"
        elif writer == "picos":
            assert not P.numberQuadConstraints, \
                "Expected no quadratic constraints after call to 'prepared'."
            if (P.numberConeConstraints + P.numberSDPConstraints) == 0:
                filename += ".lp"
            elif P.numberConeConstraints == 0:
                filename += ".dat-s"
            else:
                filename += ".cbf"
        else:
            raise Exception("unexpected writer")

    if writer == "cplex":
        cpl = CPLEXSolver(P)
        cpl._load_problem()
        cpl.int.write(filename)

    elif writer == "mosek":
        if filename.endswith(".cbf"):
            # This ensures that all quadratics are converted to SOC
            P = picos_problem.prepared(solver='cvxopt')
        msk = MOSEKSolver(P)
        msk._load_problem()
        msk.int.writedata(filename)

    elif writer == "gurobi":
        grb = GurobiSolver(P)
        grb._load_problem()
        grb.int.write(filename)

    elif writer == "picos":
        if filename[-3:] == ".lp":
            _write_lp(P, filename)
        elif filename[-6:] == ".dat-s":
            _write_sdpa(P, filename)
        elif filename[-4:] == ".cbf":
            _write_cbf(P, filename)
        else:
            raise Exception("unexpected file extension")
    else:
        raise Exception("unknown writer")


def _write_lp(P, filename):
    """Write the problem to a file in LP format."""
    # add extension
    if filename[-3:] != ".lp":
        filename += ".lp"
    # check lp compatibility
    if (
            P.numberConeConstraints
            + P.numberQuadConstraints
            + P.numberLSEConstraints
            + P.numberSDPConstraints
    ) > 0:
        raise Exception("the picos LP writer only accepts (MI)LP")
    # open file
    f = open(filename, "w")
    f.write("\\* file " + filename + " generated by picos*\\\n")

    # HACK: This abuses the internal problem representation of CVXOPT.
    if P.continuous:
        localCvxoptInstance = CVXOPTSolver(P)
    else:
        localCvxoptInstance = CVXOPTSolver(P.continuous_relaxation())
    localCvxoptInstance.import_variable_bounds = False
    localCvxoptInstance._load_problem()
    cvxoptVars = localCvxoptInstance.internal_problem()

    # variable names
    varnames = {}
    i = 0
    for name, v in P.variables.items():
        # full vectorization is used, name variables as x, x[j] or x[j,k]
        if isinstance(v._vec, FullVectorization):
            for ind in range(v.dim):
                if v.size == (1, 1):
                    varnames[i] = name
                elif v.size[1] == 1:
                    varnames[i] = name + "(" + str(ind) + ")"
                else:
                    k, j = divmod(ind, v.size[0])
                    varnames[i] = name + "(" + str(j) + "," + str(k) + ")"
                varnames[i] = varnames[i].replace("[", "(")
                varnames[i] = varnames[i].replace("]", ")")
                i += 1
        else:
            for ind in range(v.dim):
                varnames[i] = 'vec_' + name + "(" + str(ind) + ")"
                i += 1

    # affexpr writer
    def affexp_writer(constraint_name, indices, coefs):
        s = ""
        s += constraint_name
        s += " : "
        start = True
        for (i, v) in zip(indices, coefs):
            if v > 0 and not (start):
                s += "+ "
            s += "%.12g" % v
            s += " "
            s += varnames[i]
            # not the first term anymore
            start = False
        if not (coefs):
            s += "0.0 "
            s += varnames[0]
        return s

    print("writing problem in " + filename + "...")

    # objective
    if P.objective.direction == "max":
        f.write("Maximize\n")
        # max handled directly
        cvxoptVars["c"] = -cvxoptVars["c"]
    else:
        f.write("Minimize\n")
    I = cvxopt.sparse(cvxoptVars["c"]).I
    V = cvxopt.sparse(cvxoptVars["c"]).V

    f.write(affexp_writer("obj", I, V))
    f.write("\n")

    f.write("Subject To\n")
    bounds = {}
    # equality constraints:
    Ai, Aj, Av = (cvxoptVars["A"].I, cvxoptVars["A"].J, cvxoptVars["A"].V)
    ijvs = sorted(zip(Ai, Aj, Av))
    del Ai, Aj, Av
    itojv = {}
    lasti = -1
    for (i, j, v) in ijvs:
        if i == lasti:
            itojv[i].append((j, v))
        else:
            lasti = i
            itojv[i] = [(j, v)]
    ieq = 0
    for i, jv in itojv.items():
        J = [jvk[0] for jvk in jv]
        V = [jvk[1] for jvk in jv]
        if len(J) == 1:
            # fixed variable
            b = cvxoptVars["b"][i] / V[0]
            bounds[J[0]] = (b, b)
        else:
            # affine equality
            b = cvxoptVars["b"][i]
            f.write(affexp_writer("eq" + str(ieq), J, V))
            f.write(" = ")
            f.write("%.12g" % b)
            f.write("\n")
            ieq += 1

    # inequality constraints:
    Gli, Glj, Glv = (cvxoptVars["Gl"].I, cvxoptVars["Gl"].J, cvxoptVars["Gl"].V)
    ijvs = sorted(zip(Gli, Glj, Glv))
    del Gli, Glj, Glv
    itojv = {}
    lasti = -1
    for (i, j, v) in ijvs:
        if i == lasti:
            itojv[i].append((j, v))
        else:
            lasti = i
            itojv[i] = [(j, v)]
    iaff = 0
    for i, jv in itojv.items():
        J = [jvk[0] for jvk in jv]
        V = [jvk[1] for jvk in jv]
        b = cvxoptVars["hl"][i]
        f.write(affexp_writer("in" + str(iaff), J, V))
        f.write(" <= ")
        f.write("%.12g" % b)
        f.write("\n")
        iaff += 1

    # variable bounds
    # retrieve as a dictionary {index -> (lo,up)}
    i_var = 0
    for var in P.variables.values():
        for ind, lo in var.bound_dicts[0].items():
            (current_lo, current_up) = bounds.get(
                i_var + ind, (-INFINITY, INFINITY))
            new_lo = max(lo, current_lo)
            bounds[i_var + ind] = (new_lo, current_up)
        for ind, up in var.bound_dicts[1].items():
            (current_lo, current_up) = bounds.get(
                i_var + ind, (-INFINITY, INFINITY))
            new_up = min(up, current_up)
            bounds[i_var + ind] = (current_lo, new_up)
        i_var += var.dim

    f.write("Bounds\n")
    for i in range(P.numberOfVars):
        if i in bounds:
            bl, bu = bounds[i]
        else:
            bl, bu = -INFINITY, INFINITY
        if bl == -INFINITY and bu == INFINITY:
            f.write(varnames[i] + " free")
        elif bl == bu:
            f.write(varnames[i] + (" = %.12g" % bl))
        elif bl < bu:
            if bl == -INFINITY:
                f.write("-inf <= ")
            else:
                f.write("%.12g" % bl)
                f.write(" <= ")
            f.write(varnames[i])
            if bu == INFINITY:
                f.write("<= +inf")
            else:
                f.write(" <= ")
                f.write("%.12g" % bu)
        f.write("\n")

    # general integers
    f.write("Generals\n")
    i_var = 0
    for name, v in P.variables.items():
        if isinstance(v, IntegerVariable):
            for ind in range(v.dim):
                f.write(varnames[i_var + ind] + "\n")
        i_var += v.dim

    # binary variables
    f.write("Binaries\n")
    i_var = 0
    for name, v in P.variables.items():
        if isinstance(v, BinaryVariable):
            for ind in range(v.dim):
                f.write(varnames[i_var + ind] + "\n")
        i_var += v.dim

    f.write("End\n")
    print("done.")
    f.close()


def _write_sdpa(P, filename):
    """Write the problem to a file in Sparse SDPA format."""
    # HACK: This abuses the internal problem representation of CVXOPT.
    localCvxoptInstance = CVXOPTSolver(P)
    localCvxoptInstance._load_problem()
    cvxoptVars = localCvxoptInstance.internal_problem()

    dims = {}
    dims["s"] = [int(numpy.sqrt(Gsi.size[0])) for Gsi in cvxoptVars["Gs"]]
    dims["l"] = cvxoptVars["Gl"].size[0]
    dims["q"] = [Gqi.size[0] for Gqi in cvxoptVars["Gq"]]
    G = cvxoptVars["Gl"]
    h = cvxoptVars["hl"]

    # handle the equalities as 2 ineq
    if cvxoptVars["A"].size[0] > 0:
        G = cvxopt.sparse([G, cvxoptVars["A"]])
        G = cvxopt.sparse([G, -cvxoptVars["A"]])
        h = cvxopt.matrix([h, cvxoptVars["b"]])
        h = cvxopt.matrix([h, -cvxoptVars["b"]])
        dims["l"] += 2 * cvxoptVars["A"].size[0]

    for i in range(len(dims["q"])):
        G = cvxopt.sparse([G, cvxoptVars["Gq"][i]])
        h = cvxopt.matrix([h, cvxoptVars["hq"][i]])

    for i in range(len(dims["s"])):
        G = cvxopt.sparse([G, cvxoptVars["Gs"][i]])
        h = cvxopt.matrix([h, cvxoptVars["hs"][i]])

    # Remove the lines in A and b corresponding to 0==0
    JP = list(set(cvxoptVars["A"].I))
    IP = range(len(JP))
    VP = [1] * len(JP)

    # is there a constraint of the form 0==a(a not 0) ?
    if any([b for (i, b) in enumerate(cvxoptVars["b"]) if i not in JP]):
        raise Exception("infeasible constraint of the form 0=a")

    from cvxopt import sparse, spmatrix

    PP = spmatrix(VP, IP, JP, (len(IP), cvxoptVars["A"].size[0]))
    cvxoptVars["A"] = PP * cvxoptVars["A"]
    cvxoptVars["b"] = PP * cvxoptVars["b"]
    c = cvxoptVars["c"]
    # ------------------------------------------------------------#
    # make A,B,and blockstruct.                                  #
    # This code is a modification of the conelp function in SMCP #
    # ------------------------------------------------------------#
    Nl = dims["l"]
    Nq = dims["q"]
    Ns = dims["s"]
    if not Nl:
        Nl = 0

    P_m = G.size[1]

    P_b = -c
    P_blockstruct = []
    if Nl:
        P_blockstruct.append(-Nl)
    for i in Nq:
        P_blockstruct.append(i)
    for i in Ns:
        P_blockstruct.append(i)

    # write data
    # add extension
    if filename[-6:] != ".dat-s":
        filename += ".dat-s"

    # open file
    f = open(filename, "w")
    f.write('"file ' + filename + ' generated by picos"\n')
    if P.options.verbosity >= 1:
        print("writing problem in " + filename + "...")
    f.write(str(P.numberOfVars) + " = number of vars\n")
    f.write(str(len(P_blockstruct)) + " = number of blocs\n")
    # bloc structure
    f.write(str(P_blockstruct).replace("[", "(").replace("]", ")"))
    f.write(" = BlocStructure\n")
    # c vector (objective)
    f.write(str(list(-P_b)).replace("[", "{").replace("]", "}"))
    f.write("\n")
    # coefs
    for k in range(P_m + 1):
        if k != 0:
            v = sparse(G[:, k - 1])
        else:
            v = +sparse(h)

        ptr = 0
        block = 0
        # lin. constraints
        if Nl:
            u = v[:Nl]
            for i, j, value in zip(u.I, u.I, u.V):
                f.write(
                    "{0}\t{1}\t{2}\t{3}\t{4}\n".format(
                        k, block + 1, j + 1, i + 1, -value
                    )
                )
            ptr += Nl
            block += 1

        # SOC constraints
        for nq in Nq:
            u0 = v[ptr]
            u1 = v[ptr + 1:ptr + nq]
            tmp = spmatrix(
                u1.V, [nq - 1 for j in range(len(u1))], u1.I, (nq, nq)
            )
            if not u0 == 0.0:
                tmp += spmatrix(u0, range(nq), range(nq), (nq, nq))
            for i, j, value in zip(tmp.I, tmp.J, tmp.V):
                f.write(
                    "{0}\t{1}\t{2}\t{3}\t{4}\n".format(
                        k, block + 1, j + 1, i + 1, -value
                    )
                )
            ptr += nq
            block += 1

        # SDP constraints
        for ns in Ns:
            u = v[ptr:ptr + ns ** 2]
            for index_k, index in enumerate(u.I):
                j, i = divmod(index, ns)
                if j <= i:
                    f.write(
                        "{0}\t{1}\t{2}\t{3}\t{4}\n".format(
                            k, block + 1, j + 1, i + 1, -u.V[index_k]
                        )
                    )
            ptr += ns ** 2
            block += 1

    f.close()


# This is an ugly function from a former version of picos
# It separates a linear constraint between 'plain' vars and matrix 'bar'
# variables J and V denote the sparse indices/values of the constraints for the
# whole (s-)vectorized vector (without offset)
def _separate_linear_cons_plain_bar_vars(J, V, idx_sdp_vars):
    # sparse values of the constraint for 'plain' variables
    jj = []
    vv = []
    # sparse values of the constraint for the next svec bar variable
    js = []
    vs = []
    mats = []
    offset = 0
    if idx_sdp_vars:
        idxsdpvars = [ti for ti in idx_sdp_vars]
        nextsdp = idxsdpvars.pop()
    else:
        return J, V, []
    for (j, v) in zip(J, V):
        if j < nextsdp[0]:
            jj.append(j - offset)
            vv.append(v)
        elif j < nextsdp[1]:
            js.append(j - nextsdp[0])
            vs.append(v)
        else:
            while j >= nextsdp[1]:
                mats.append(
                    devectorize(
                        cvxopt.spmatrix(
                            vs,
                            js,
                            [0] * len(js),
                            (nextsdp[1] - nextsdp[0], 1)
                        )).T)
                js = []
                vs = []
                offset += (nextsdp[1] - nextsdp[0])
                try:
                    nextsdp = idxsdpvars.pop()
                except IndexError:
                    nextsdp = (float('inf'), float('inf'))
            if j < nextsdp[0]:
                jj.append(j - offset)
                vv.append(v)
            elif j < nextsdp[1]:
                js.append(j - nextsdp[0])
                vs.append(v)
    while len(mats) < len(idx_sdp_vars):
        mats.append(
            devectorize(
                cvxopt.spmatrix(
                    vs,
                    js,
                    [0] * len(js),
                    (nextsdp[1] - nextsdp[0], 1)
                )).T)
        js = []
        vs = []
        nextsdp = (0, 1)  # doesnt matter, it will be an empt matrix anyway
    return jj, vv, mats


def devectorize(vec):
    """Create a matrix from a symmetric vectorization."""
    from ..expressions.vectorizations import SymmetricVectorization
    v = vec.size[0]
    n = int((1 + 8 * v) ** 0.5 - 1) // 2
    return SymmetricVectorization((n, n)).devectorize(vec)


def _write_cbf(P, filename, uptri=False):
    """Write the problem to a file in Sparse SDPA format.

    :param bool uptri: Whether upper triangular elements of symmetric
        matrices are specified.
    """
    # write data
    # add extension
    if filename[-4:] != ".cbf":
        filename += ".cbf"

    # parse variables
    semidef_vars = set()
    for cons in P.constraints:
        cs = P.constraints[cons]
        if isinstance(cs, LMIConstraint):
            if cs.semidefVar:
                semidef_vars.add(cs.semidefVar)

    NUMVAR_SCALAR = int(
        sum(
            [var.dim
             for var in P.variables.values()
             if var not in semidef_vars]
        )
    )

    ind = 0
    indices = []
    start_indices = {}
    for v in P.variables.values():
        indices.append((ind, ind + v.dim, v))
        start_indices[v] = ind
        ind += v.dim

    indices = sorted(indices)
    idxsdpvars = [
        (si, ei) for (si, ei, v) in indices[::-1] if v in semidef_vars]
    # search if some semidef vars are implied in other semidef constraints
    PSD_not_handled = []
    for c in P.constraints.values():
        if isinstance(c, LMIConstraint) and c not in semidef_vars:
            for v in (c.lhs - c.rhs)._coefs:
                if v in semidef_vars:
                    start_index = start_indices[v]
                    idx = (start_index, start_index + v.dim)
                    if idx in idxsdpvars:
                        PSD_not_handled.append(v)
                        NUMVAR_SCALAR += idx[1] - idx[0]
                        idxsdpvars.remove(idx)

    barvars = bool(idxsdpvars)

    # find integer variables, retrieve bounds and put 0-1 bounds on binaries
    ints = []
    ind = 0
    varbounds_lo = {}
    varbounds_up = {}

    for k, var in P.variables.items():
        if isinstance(var, BinaryVariable):
            for relind, absind in enumerate(range(ind, ind + var.dim)):
                ints.append(absind)
                clb = var.bound_dicts[0].get(relind, -INFINITY)
                cub = var.bound_dicts[1].get(relind, INFINITY)
                varbounds_lo[absind] = max(0.0, clb)
                varbounds_up[absind] = min(1.0, cub)

        elif (isinstance(var, IntegerVariable) or
              isinstance(var, CONTINUOUS_VARTYPES)):
            for relind, absind in enumerate(range(ind, ind + var.dim)):
                if isinstance(var, IntegerVariable):
                    ints.append(absind)
                clb = var.bound_dicts[0].get(relind, -INFINITY)
                cub = var.bound_dicts[1].get(relind, INFINITY)
                varbounds_lo[absind] = clb
                varbounds_up[absind] = cub

        else:
            raise Exception("variable type not handled by _write_cbf()")
        ind += var.dim

    if barvars:
        ints, _, mats = _separate_linear_cons_plain_bar_vars(
            ints, [0.0] * len(ints), idxsdpvars
        )
        if any([bool(mat) for mat in mats]):
            raise Exception(
                "semidef vars with integer elements are not supported"
            )

    # open file
    f = open(filename, "w")
    f.write("#file " + filename + " generated by picos\n")
    print("writing problem in " + filename + "...")

    f.write("VER\n")
    f.write("1\n\n")

    f.write("OBJSENSE\n")
    if P.objective.direction == "max":
        f.write("MAX\n\n")
    else:
        f.write("MIN\n\n")

    # VARIABLEs

    if barvars:
        f.write("PSDVAR\n")
        f.write(str(len(idxsdpvars)) + "\n")
        for si, ei in idxsdpvars:
            ni = int(((8 * (ei - si) + 1) ** 0.5 - 1) / 2.0)
            f.write(str(ni) + "\n")
        f.write("\n")

    # bounds
    cones = []
    conecons = []
    Acoord = []
    Bcoord = []
    iaff = 0
    offset = 0
    for si, ei, v in indices:
        if v in semidef_vars and not (v in PSD_not_handled):
            offset += ei - si
        else:
            if all(varbounds_lo[ind] == 0 for ind in range(si, ei)):
                cones.append(("L+", ei - si))
            elif all(varbounds_up[ind] == 0 for ind in range(si, ei)):
                cones.append(("L-", ei - si))
            else:
                cones.append(("F", ei - si))
            if any(varbounds_lo[ind] != 0 for ind in range(si, ei)):
                for ind in range(si, ei):
                    l = varbounds_lo[ind]
                    if l - 1 > -INFINITY:
                        Acoord.append((iaff, ind - offset, 1.0))
                        Bcoord.append((iaff, -l))
                        iaff += 1
            if any(varbounds_up[ind] != 0 for ind in range(si, ei)):
                for ind in range(si, ei):
                    u = varbounds_up[ind]
                    if u + 1 < INFINITY:
                        Acoord.append((iaff, ind - offset, -1.0))
                        Bcoord.append((iaff, u))
                        iaff += 1
    if iaff:
        conecons.append(("L+", iaff))

    f.write("VAR\n")
    f.write(str(NUMVAR_SCALAR) + " " + str(len(cones)) + "\n")
    for tp, n in cones:
        f.write(tp + " " + str(n) + "\n")

    f.write("\n")

    # integers
    if ints:
        f.write("INT\n")
        f.write(str(len(ints)) + "\n")
        for i in ints:
            f.write(str(i) + "\n")
        f.write("\n")

    # constraints
    psdcons = []
    isdp = 0
    Fcoord = []
    Hcoord = []
    Dcoord = []
    ObjAcoord = []
    ObjBcoord = []
    ObjFcoord = []

    # dummy constraint for the objective
    dummy_cons = P.objective.function >= 0
    setattr(dummy_cons, "dummycon", None)

    for cons in chain((dummy_cons,), P.constraints.values()):
        if isinstance(cons, LMIConstraint):
            v = cons.semidefVar
            if v is not None and v not in PSD_not_handled:
                continue

        # get sparse indices
        if isinstance(cons, AffineConstraint):
            expcone = cons.lhs - cons.rhs
            if hasattr(cons, "dummycon"):
                conetype = "0"  # Dummy type for the objective function.
            elif cons.is_equality():
                conetype = "L="
            elif cons.is_increasing():
                conetype = "L-"
            elif cons.is_decreasing():
                conetype = "L+"
            else:
                assert False, "Unexpected constraint relation."
        elif isinstance(cons, SOCConstraint):
            expcone = (cons.ub) // (cons.ne[:])
            conetype = "Q"
        elif isinstance(cons, RSOCConstraint):
            expcone = (cons.ub1) // (0.5 * cons.ub2) // (cons.ne[:])
            conetype = "QR"
        elif isinstance(cons, LMIConstraint):
            if cons.is_increasing():
                expcone = cons.rhs - cons.lhs
                conetype = None
            elif cons.is_decreasing():
                expcone = cons.lhs - cons.rhs
                conetype = None
            else:
                assert False, "Unexpected constraint relation."
        else:
            assert False, "Unexpected constraint type."

        ijv = []
        for var, fact in expcone._coefs.items():
            if not isinstance(fact, cvxopt.base.spmatrix):
                fact = cvxopt.sparse(fact)
            sj = start_indices[var]
            ijv.extend(zip(fact.I, fact.J + sj, fact.V))
        ijvs = sorted(ijv)

        itojv = {}
        lasti = -1
        for (i, j, v) in ijvs:
            if i == lasti:
                itojv[i].append((j, v))
            else:
                lasti = i
                itojv[i] = [(j, v)]

        if conetype:
            if conetype != "0":
                dim = expcone.size[0] * expcone.size[1]
                conecons.append((conetype, dim))
        else:
            dim = expcone.size[0]
            psdcons.append(dim)

        if conetype:
            for i, jv in itojv.items():
                J = [jvk[0] for jvk in jv]
                V = [jvk[1] for jvk in jv]
                J, V, mats = _separate_linear_cons_plain_bar_vars(
                    J, V, idxsdpvars)
                for j, v in zip(J, V):
                    if conetype != "0":
                        Acoord.append((iaff + i, j, v))
                    else:
                        ObjAcoord.append((j, v))
                for k, mat in enumerate(mats):
                    for row, col, v in zip(mat.I, mat.J, mat.V):
                        if conetype != "0":
                            Fcoord.append((iaff + i, k, row, col, v))
                        else:
                            ObjFcoord.append((k, row, col, v))
                        if uptri and row != col:
                            if conetype != "0":
                                Fcoord.append((iaff + i, k, col, row, v))
                            else:
                                ObjFcoord.append((k, col, row, v))
            constant = expcone._const
            if not (constant is None):
                constant = cvxopt.sparse(constant)
                for i, v in zip(constant.I, constant.V):
                    if conetype != "0":
                        Bcoord.append((iaff + i, v))
                    else:
                        ObjBcoord.append(v)
        else:
            for i, jv in itojv.items():
                col, row = divmod(i, dim)
                if not (uptri) and row < col:
                    continue
                J = [jvk[0] for jvk in jv]
                V = [jvk[1] for jvk in jv]
                J, V, mats = _separate_linear_cons_plain_bar_vars(
                    J, V, idxsdpvars)
                if any([bool(m) for m in mats]):
                    raise Exception("SDP cons should not depend on PSD var")
                for j, v in zip(J, V):
                    Hcoord.append((isdp, j, row, col, v))

            constant = expcone._const
            if not (constant is None):
                constant = cvxopt.sparse(constant)
                for i, v in zip(constant.I, constant.V):
                    col, row = divmod(i, dim)
                    if row < col:
                        continue
                    Dcoord.append((isdp, row, col, v))

        if conetype:
            if conetype != "0":
                iaff += dim
        else:
            isdp += 1

    if iaff > 0:
        f.write("CON\n")
        f.write(str(iaff) + " " + str(len(conecons)) + "\n")
        for tp, n in conecons:
            f.write(tp + " " + str(n))
            f.write("\n")

        f.write("\n")

    if isdp > 0:
        f.write("PSDCON\n")
        f.write(str(isdp) + "\n")
        for n in psdcons:
            f.write(str(n) + "\n")
        f.write("\n")

    if ObjFcoord:
        f.write("OBJFCOORD\n")
        f.write(str(len(ObjFcoord)) + "\n")
        for (k, row, col, v) in ObjFcoord:
            f.write("{0} {1} {2} {3}\n".format(k, row, col, v))
        f.write("\n")

    if ObjAcoord:
        f.write("OBJACOORD\n")
        f.write(str(len(ObjAcoord)) + "\n")
        for (j, v) in ObjAcoord:
            f.write("{0} {1}\n".format(j, v))
        f.write("\n")

    if ObjBcoord:
        f.write("OBJBCOORD\n")
        v = ObjBcoord[0]
        f.write("{0}\n".format(v))
        f.write("\n")

    if Fcoord:
        f.write("FCOORD\n")
        f.write(str(len(Fcoord)) + "\n")
        for (i, k, row, col, v) in Fcoord:
            f.write("{0} {1} {2} {3} {4}\n".format(i, k, row, col, v))
        f.write("\n")

    if Acoord:
        f.write("ACOORD\n")
        f.write(str(len(Acoord)) + "\n")
        for (i, j, v) in Acoord:
            f.write("{0} {1} {2}\n".format(i, j, v))
        f.write("\n")

    if Bcoord:
        f.write("BCOORD\n")
        f.write(str(len(Bcoord)) + "\n")
        for (i, v) in Bcoord:
            f.write("{0} {1}\n".format(i, v))
        f.write("\n")

    if Hcoord:
        f.write("HCOORD\n")
        f.write(str(len(Hcoord)) + "\n")
        for (i, j, row, col, v) in Hcoord:
            f.write("{0} {1} {2} {3} {4}\n".format(i, j, row, col, v))
        f.write("\n")

    if Dcoord:
        f.write("DCOORD\n")
        f.write(str(len(Dcoord)) + "\n")
        for (i, row, col, v) in Dcoord:
            f.write("{0} {1} {2} {3}\n".format(i, row, col, v))
        f.write("\n")

    print("done.")
    f.close()


# --------------------------------------
__all__ = api_end(_API_START, globals())
