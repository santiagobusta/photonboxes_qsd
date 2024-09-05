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

"""Functions for reading optimization problems from a file."""

# ------------------------------------------------------------------------------
# TODO: This file is using ad-hoc parser implementations. Use a proper parser
#       generation library and a unified 'read' function for different formats.
# ------------------------------------------------------------------------------

import warnings

import cvxopt
import numpy

from ..apidoc import api_end, api_start
from ..expressions import AffineExpression, Constant
from .problem import Problem

_API_START = api_start(globals())
# -------------------------------


def _spmatrix(*args, **kwargs):
    """Create a CVXOPT sparse matrix.

    A wrapper around :func:`cvxopt.spmatrix` that converts indices to
    :class:`int`, if necessary.

    Works around PICOS sometimes passing indices as :class:`numpy:numpy.int64`.
    """
    try:
        return cvxopt.spmatrix(*args, **kwargs)
    except TypeError as error:
        # CVXOPT does not like NumPy's int64 scalar type for indices, so attempt
        # to convert all indices to Python's int.
        if str(error) == "non-numeric type in list":
            newargs = list(args)

            for argNum, arg in enumerate(args):
                if argNum in (1, 2):  # Positional I, J.
                    newargs[argNum] = [int(x) for x in args[argNum]]

            for kw in "IJ":
                if kw in kwargs:
                    kwargs[kw] = [int(x) for x in kwargs[kw]]

            return cvxopt.spmatrix(*newargs, **kwargs)
        else:
            raise


def _break_cols(mat, sizes):
    """Help with the reading of CBF files."""
    n = len(sizes)
    I, J, V = [], [], []
    for i in range(n):
        I.append([])
        J.append([])
        V.append([])
    cumsz = numpy.cumsum(sizes)
    import bisect

    for i, j, v in zip(mat.I, mat.J, mat.V):
        block = bisect.bisect(cumsz, j)
        I[block].append(i)
        V[block].append(v)
        if block == 0:
            J[block].append(j)
        else:
            J[block].append(j - cumsz[block - 1])
    return [
        _spmatrix(V[k], I[k], J[k], (mat.size[0], sz))
        for k, sz in enumerate(sizes)
    ]


def _break_rows(mat, sizes):
    """Help with the reading of CBF files."""
    n = len(sizes)
    I, J, V = [], [], []
    for i in range(n):
        I.append([])
        J.append([])
        V.append([])
    cumsz = numpy.cumsum(sizes)
    import bisect

    for i, j, v in zip(mat.I, mat.J, mat.V):
        block = bisect.bisect(cumsz, i)
        J[block].append(j)
        V[block].append(v)
        if block == 0:
            I[block].append(i)
        else:
            I[block].append(i - cumsz[block - 1])
    return [
        _spmatrix(V[k], I[k], J[k], (sz, mat.size[1]))
        for k, sz in enumerate(sizes)
    ]


def _block_idx(i, sizes):
    """Help with the reading of CBF files."""
    # if there are blocks of sizes n1,...,nk and i is
    # the index of an element of the big vectorized variable,
    # returns the block of i and its index inside the sub-block.
    cumsz = numpy.cumsum(sizes)
    import bisect

    block = bisect.bisect(cumsz, i)
    return block, (i if block == 0 else i - cumsz[block - 1])


def _read_cbf_block(P, blocname, f, parsed_blocks):
    """Help with the reading of CBF files."""
    if blocname == "OBJSENSE":
        objsense = f.readline().split()[0].lower()
        P.set_objective(objsense, P.objective.normalized.function)
        return None
    elif blocname == "PSDVAR":
        n = int(f.readline())
        vardims = []
        XX = []
        for i in range(n):
            ni = int(f.readline())
            vardims.append(ni)
            Xi = P.add_variable("X[" + str(i) + "]", (ni, ni), "symmetric")
            XX.append(Xi)
            P.add_constraint(Xi >> 0)
        return vardims, XX
    elif blocname == "VAR":
        Nscalar, ncones = [int(fi) for fi in f.readline().split()]
        tot_dim = 0
        var_structure = []
        xx = []
        for i in range(ncones):
            lsplit = f.readline().split()
            tp, dim = lsplit[0], int(lsplit[1])
            tot_dim += dim
            var_structure.append(dim)
            if tp == "F":
                xi = P.add_variable("x[" + str(i) + "]", dim)
            elif tp == "L+":
                xi = P.add_variable("x[" + str(i) + "]", dim, lower=0)
            elif tp == "L-":
                xi = P.add_variable("x[" + str(i) + "]", dim, upper=0)
            elif tp == "L=":
                xi = P.add_variable("x[" + str(i) + "]", dim, lower=0, upper=0)
            elif tp == "Q":
                xi = P.add_variable("x[" + str(i) + "]", dim)
                P.add_constraint(abs(xi[1:]) < xi[0])
            elif tp == "QR":
                xi = P.add_variable("x[" + str(i) + "]", dim)
                P.add_constraint(abs(xi[2:]) ** 2 < 2 * xi[0] * xi[1])
            xx.append(xi)
        if tot_dim != Nscalar:
            raise Exception("VAR dimensions do not match the header")
        return Nscalar, var_structure, xx
    elif blocname == "INT":
        n = int(f.readline())
        ints = {}
        for k in range(n):
            j = int(f.readline())
            i, col = _block_idx(j, parsed_blocks["VAR"][1])
            ints.setdefault(i, [])
            ints[i].append(col)
        x = parsed_blocks["VAR"][2]
        for i in ints:
            if len(ints[i]) == x[i].size[0]:
                x[i].vtype = "integer"
            else:
                x.append(
                    P.add_variable(
                        "x_int[" + str(i) + "]", len(ints[i]), "integer"
                    )
                )
                for k, j in enumerate(ints[i]):
                    P.add_constraint(x[i][j] == x[-1][k])
        return x
    elif blocname == "CON":
        Ncons, ncones = [int(fi) for fi in f.readline().split()]
        cons_structure = []
        tot_dim = 0
        for i in range(ncones):
            lsplit = f.readline().split()
            tp, dim = lsplit[0], int(lsplit[1])
            tot_dim += dim
            cons_structure.append((tp, dim))
        if tot_dim != Ncons:
            raise Exception("CON dimensions do not match the header")
        return Ncons, cons_structure
    elif blocname == "PSDCON":
        n = int(f.readline())
        psdcons_structure = []
        for i in range(n):
            ni = int(f.readline())
            psdcons_structure.append(ni)
        return psdcons_structure
    elif blocname == "OBJACOORD":
        n = int(f.readline())
        J = []
        V = []
        for i in range(n):
            lsplit = f.readline().split()
            j, v = int(lsplit[0]), float(lsplit[1])
            J.append(j)
            V.append(v)
        return _spmatrix(V, [0] * len(J), J, (1, parsed_blocks["VAR"][0]))
    elif blocname == "OBJBCOORD":
        return float(f.readline())
    elif blocname == "OBJFCOORD":
        n = int(f.readline())
        Fobj = [
            _spmatrix([], [], [], (ni, ni)) for ni in parsed_blocks["PSDVAR"][0]
        ]
        for k in range(n):
            lsplit = f.readline().split()
            j, row, col, v = (
                int(lsplit[0]),
                int(lsplit[1]),
                int(lsplit[2]),
                float(lsplit[3]),
            )
            Fobj[j][row, col] = v
            if row != col:
                Fobj[j][col, row] = v
        return Fobj
    elif blocname == "FCOORD":
        n = int(f.readline())
        Fblocks = {}
        for k in range(n):
            lsplit = f.readline().split()
            i, j, row, col, v = (
                int(lsplit[0]),
                int(lsplit[1]),
                int(lsplit[2]),
                int(lsplit[3]),
                float(lsplit[4]),
            )
            if i not in Fblocks:
                Fblocks[i] = [
                    _spmatrix([], [], [], (ni, ni))
                    for ni in parsed_blocks["PSDVAR"][0]
                ]
            Fblocks[i][j][row, col] = v
            if row != col:
                Fblocks[i][j][col, row] = v
        return Fblocks
    elif blocname == "ACOORD":
        n = int(f.readline())
        J = []
        V = []
        I = []
        for k in range(n):
            lsplit = f.readline().split()
            i, j, v = int(lsplit[0]), int(lsplit[1]), float(lsplit[2])
            I.append(i)
            J.append(j)
            V.append(v)
        return _spmatrix(
            V, I, J, (parsed_blocks["CON"][0], parsed_blocks["VAR"][0])
        )
    elif blocname == "BCOORD":
        n = int(f.readline())
        V = []
        I = []
        for k in range(n):
            lsplit = f.readline().split()
            i, v = int(lsplit[0]), float(lsplit[1])
            I.append(i)
            V.append(v)
        return _spmatrix(V, I, [0] * len(I), (parsed_blocks["CON"][0], 1))
    elif blocname == "HCOORD":
        n = int(f.readline())
        Hblocks = {}
        for k in range(n):
            lsplit = f.readline().split()
            i, j, row, col, v = (
                int(lsplit[0]),
                int(lsplit[1]),
                int(lsplit[2]),
                int(lsplit[3]),
                float(lsplit[4]),
            )
            if j not in Hblocks:
                Hblocks[j] = [
                    _spmatrix([], [], [], (ni, ni))
                    for ni in parsed_blocks["PSDCON"]
                ]
            Hblocks[j][i][row, col] = v
            if row != col:
                Hblocks[j][i][col, row] = v
        return Hblocks
    elif blocname == "DCOORD":
        n = int(f.readline())
        Dblocks = [
            _spmatrix([], [], [], (ni, ni)) for ni in parsed_blocks["PSDCON"]
        ]
        for k in range(n):
            lsplit = f.readline().split()
            i, row, col, v = (
                int(lsplit[0]),
                int(lsplit[1]),
                int(lsplit[2]),
                float(lsplit[3]),
            )
            Dblocks[i][row, col] = v
            if row != col:
                Dblocks[i][col, row] = v
        return Dblocks
    else:
        raise Exception("unexpected block name")


def import_cbf(filename):
    """Create a :class:`~picos.Problem` from a CBF file.

    The created problem contains one (multidimensional) variable for each cone
    specified in the section ``VAR`` of the .cbf file, and one
    (multidimmensional) constraint for each cone specified in the sections
    ``CON`` and ``PSDCON``.

    :returns: A tuple ``(P, x, X, params)`` where

        - ``P`` is the imported picos :class:`~picos.Problem` object,
        - ``x`` is a list of multidimensional variables representing the scalar
          variables found in the file,
        - ``X`` is a list of symmetric variables representing the positive
          semidefinite variables found in the file, and
        - ``params`` is a dictionary containing PICOS cosntants used to define
          the problem. Indexing is with respect to the blocks of variables as
          defined in the sections ``VAR`` and  ``CON`` of the file.
    """
    P = Problem()

    try:
        f = open(filename, "r")
    except IOError:
        filename += ".cbf"
        f = open(filename, "r")

    line = f.readline()
    while not line.startswith("VER"):
        line = f.readline()

    ver = int(f.readline())
    if ver != 1:
        warnings.warn("CBF file has a version other than 1.")

    structure_keywords = ["OBJSENSE", "PSDVAR", "VAR", "INT", "PSDCON", "CON"]
    data_keywords = [
        "OBJFCOORD",
        "OBJACOORD",
        "OBJBCOORD",
        "FCOORD",
        "ACOORD",
        "BCOORD",
        "HCOORD",
        "DCOORD",
    ]

    structure_mode = True  # still parsing structure blocks
    seen_blocks = []
    parsed_blocks = {}
    while True:
        line = f.readline()
        if not line:
            break
        lsplit = line.split()
        if lsplit and lsplit[0] in structure_keywords:
            if lsplit[0] == "INT" and ("VAR" not in seen_blocks):
                raise Exception("INT BLOCK before VAR BLOCK")
            if lsplit[0] == "CON" and not (
                "VAR" in seen_blocks or "PSDVAR" in seen_blocks
            ):
                raise Exception("CON BLOCK before VAR/PSDVAR BLOCK")
            if lsplit[0] == "PSDCON" and not (
                "VAR" in seen_blocks or "PSDVAR" in seen_blocks
            ):
                raise Exception("PSDCON BLOCK before VAR/PSDVAR BLOCK")
            if lsplit[0] == "VAR" and (
                "CON" in seen_blocks or "PSDCON" in seen_blocks
            ):
                raise Exception("VAR BLOCK after CON/PSDCON BLOCK")
            if lsplit[0] == "PSDVAR" and (
                "CON" in seen_blocks or "PSDCON" in seen_blocks
            ):
                raise Exception("PSDVAR BLOCK after CON/PSDCON BLOCK")
            if structure_mode:
                parsed_blocks[lsplit[0]] = _read_cbf_block(
                    P, lsplit[0], f, parsed_blocks
                )
                seen_blocks.append(lsplit[0])
            else:
                raise Exception("Structure keyword after first data item")
        if lsplit and lsplit[0] in data_keywords:
            if "OBJSENSE" not in seen_blocks:
                raise Exception("missing OBJSENSE block")
            if not ("VAR" in seen_blocks or "PSDVAR" in seen_blocks):
                raise Exception("missing VAR/PSDVAR block")
            if lsplit[0] in ("OBJFCOORD", "FCOORD") and not (
                "PSDVAR" in seen_blocks
            ):
                raise Exception("missing PSDVAR block")
            if lsplit[0] in ("OBJACOORD", "ACOORD", "HCOORD") and not (
                "VAR" in seen_blocks
            ):
                raise Exception("missing VAR block")
            if lsplit[0] in ("DCOORD", "HCOORD") and not (
                "PSDCON" in seen_blocks
            ):
                raise Exception("missing PSDCON block")
            structure_mode = False
            parsed_blocks[lsplit[0]] = _read_cbf_block(
                P, lsplit[0], f, parsed_blocks
            )
            seen_blocks.append(lsplit[0])

    f.close()
    # variables
    if "VAR" in parsed_blocks:
        Nvars, varsz, x = parsed_blocks["VAR"]
    else:
        x = None

    if "INT" in parsed_blocks:
        x = parsed_blocks["INT"]

    if "PSDVAR" in parsed_blocks:
        psdsz, X = parsed_blocks["PSDVAR"]
    else:
        X = None

    # objective
    obj_constant = parsed_blocks.get("OBJBCOORD", 0)
    bobj = Constant("bobj", obj_constant)
    obj = Constant("bobj", obj_constant)

    aobj = {}
    if "OBJACOORD" in parsed_blocks:
        obj_vecs = _break_cols(parsed_blocks["OBJACOORD"], varsz)
        aobj = {}
        for k, v in enumerate(obj_vecs):
            if v:
                aobj[k] = Constant("c[" + str(k) + "]", v)
                obj += aobj[k] * x[k]

    Fobj = {}
    if "OBJFCOORD" in parsed_blocks:
        Fbl = parsed_blocks["OBJFCOORD"]
        for i, Fi in enumerate(Fbl):
            if Fi:
                Fobj[i] = Constant("F[" + str(i) + "]", Fi)
                obj += Fobj[i] | X[i]

    P.set_objective(P.objective.normalized.direction, obj)

    # cone constraints
    bb = {}
    AA = {}
    FF = {}
    if "CON" in parsed_blocks:
        Ncons, structcons = parsed_blocks["CON"]
        szcons = [s for tp, s in structcons]

        b = parsed_blocks.get("BCOORD", _spmatrix([], [], [], (Ncons, 1)))
        bvecs = _break_rows(b, szcons)
        consexp = []
        for i, bi in enumerate(bvecs):
            bb[i] = Constant("b[" + str(i) + "]", bi)
            consexp.append(Constant("b[" + str(i) + "]", bi))

        A = parsed_blocks.get("ACOORD", _spmatrix([], [], [], (Ncons, Nvars)))
        Ablc = _break_rows(A, szcons)
        for i, Ai in enumerate(Ablc):
            Aiblocs = _break_cols(Ai, varsz)
            for j, Aij in enumerate(Aiblocs):
                if Aij:
                    AA[i, j] = Constant("A[" + str((i, j)) + "]", Aij)
                    consexp[i] += AA[i, j] * x[j]

        Fcoords = parsed_blocks.get("FCOORD", {})
        for k, mats in Fcoords.items():
            i, row = _block_idx(k, szcons)
            row_exp = AffineExpression.zero()
            for j, mat in enumerate(mats):
                if mat:
                    FF[i, j, row] = Constant("F[" + str((i, j, row)) + "]", mat)
                    row_exp += FF[i, j, row] | X[j]

            consexp[i] += (
                "e_" + str(row) + "(" + str(szcons[i]) + ",1)"
            ) * row_exp

        for i, (tp, sz) in enumerate(structcons):
            if tp == "F":
                continue
            elif tp == "L-":
                P.add_constraint(consexp[i] < 0)
            elif tp == "L+":
                P.add_constraint(consexp[i] > 0)
            elif tp == "L=":
                P.add_constraint(consexp[i] == 0)
            elif tp == "Q":
                P.add_constraint(abs(consexp[i][1:]) < consexp[i][0])
            elif tp == "QR":
                P.add_constraint(
                    abs(consexp[i][2:]) ** 2 < 2 * consexp[i][0] * consexp[i][1]
                )
            else:
                raise Exception("unexpected cone type")

    DD = {}
    HH = {}
    if "PSDCON" in parsed_blocks:
        Dblocks = parsed_blocks.get(
            "DCOORD",
            [_spmatrix([], [], [], (ni, ni)) for ni in parsed_blocks["PSDCON"]],
        )
        Hblocks = parsed_blocks.get("HCOORD", {})

        consexp = []
        for i, Di in enumerate(Dblocks):
            DD[i] = Constant("D[" + str(i) + "]", Di)
            consexp.append(Constant("D[" + str(i) + "]", Di))

        for j, Hj in Hblocks.items():
            i, col = _block_idx(j, varsz)
            for k, Hij in enumerate(Hj):
                if Hij:
                    HH[k, i, col] = Constant("H[" + str((k, i, col)) + "]", Hij)
                    consexp[k] += HH[k, i, col] * x[i][col]

        for exp in consexp:
            P.add_constraint(exp >> 0)

    params = {
        "aobj": aobj,
        "bobj": bobj,
        "Fobj": Fobj,
        "A": AA,
        "b": bb,
        "F": FF,
        "D": DD,
        "H": HH,
    }

    return P, x, X, params


# --------------------------------------
__all__ = api_end(_API_START, globals())
