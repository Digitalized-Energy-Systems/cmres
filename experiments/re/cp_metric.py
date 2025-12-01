# pip install jax jaxlib

import monee
import monee.model as mm
import monee.network.mes as mes

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# ----------------------------
# Build Ybus from AC branches
# ----------------------------
def build_ybus_internal(nb, branches):
    """
    branches: (f, t, r, x, g_fr, b_fr, g_to, b_to, tap, shift_rad)
    """
    Ybus = jnp.zeros((nb, nb), dtype=jnp.complex64)
    for (f, t, r, x, gfr, bfr, gto, bto, tap, shift_rad) in branches:
        y_series = 1.0 / complex(r, x)
        y_sh_fr = complex(gfr, bfr)
        y_sh_to = complex(gto, bto)

        a = tap if tap != 0.0 else 1.0
        th = shift_rad
        tap_c = a * jnp.exp(1j * th)

        # Off-diagonals
        Ybus = Ybus.at[f, t].add(-y_series / jnp.conj(tap_c))
        Ybus = Ybus.at[t, f].add(-y_series / tap_c)

        # Diagonals
        Ybus = Ybus.at[f, f].add((y_series / (tap_c * jnp.conj(tap_c))) + y_sh_fr)
        Ybus = Ybus.at[t, t].add(y_series + y_sh_to)

    return Ybus

# ----------------------------
# AC nodal injections
# ----------------------------
def power_injections(theta, V, Ybus):
    """
    theta [nb], V [nb], Ybus complex -> P[nb], Q[nb]
    """
    E = V * jnp.exp(1j * theta)   # bus voltages
    I = Ybus @ E
    S = E * jnp.conj(I)
    return S.real, S.imag

# ----------------------------
# Line active power P_ij for each branch (from i to j)
# ----------------------------
def line_active_powers(theta, V, branches):
    P = []
    for (f, t, r, x, gfr, bfr, gto, bto, tap, shift_rad) in branches:
        a = tap if tap != 0.0 else 1.0
        ths = shift_rad
        y = 1.0 / complex(r, x)
        g = jnp.float32(y.real)
        b = jnp.float32(y.imag)

        Vi, Vj = V[f], V[t]
        dth = theta[f] - theta[t] - ths

        # From-end active power (standard π-model, tap on from side)
        # P_ij = (Vi^2 / a^2) * g - (Vi*Vj / a) * ( g*cos(dth) + b*sin(dth) )
        Pij = (Vi*Vi / (a*a)) * g - (Vi*Vj / a) * ( g*jnp.cos(dth) + b*jnp.sin(dth) )
        P.append(Pij)
    return jnp.array(P)

# ----------------------------
# Reduced AC Jacobian (PV/PQ/slack)
# ----------------------------
def build_reduced_jacobian(theta, V, Ybus, bus_types):
    """
    bus_types: 0=PQ, 1=PV, 2=Slack (exactly one slack).
    Returns:
      J_red  : [[dP/dθ_non-slack, dP/dV_PQ],
                [dQ/dθ_non-slack, dQ/dV_PQ]]
      idx_theta (non-slack), idx_Vpq (PQ),
      idx_P (P rows for non-slack buses),
      idx_Q (Q rows for PQ buses)
    """
    nb = theta.shape[0]
    is_slack = (bus_types == 2)
    is_pv    = (bus_types == 1)
    is_pq    = (bus_types == 0)

    idx_theta = jnp.where(~is_slack)[0]   # state angles (exclude slack)
    idx_Vpq   = jnp.where(is_pq)[0]       # state voltages for PQ buses
    idx_P     = jnp.where(~is_slack)[0]   # P equations for non-slack
    idx_Q     = jnp.where(is_pq)[0]       # Q equations for PQ buses

    def P_of(th, Vm): return power_injections(th, Vm, Ybus)[0]
    def Q_of(th, Vm): return power_injections(th, Vm, Ybus)[1]

    H = jax.jacobian(P_of, argnums=0)(theta, V)  # dP/dθ
    N = jax.jacobian(P_of, argnums=1)(theta, V)  # dP/dV
    M = jax.jacobian(Q_of, argnums=0)(theta, V)  # dQ/dθ
    L = jax.jacobian(Q_of, argnums=1)(theta, V)  # dQ/dV

    H_red = H[jnp.ix_(idx_P, idx_theta)]
    N_red = N[jnp.ix_(idx_P, idx_Vpq)]
    M_red = M[jnp.ix_(idx_Q, idx_theta)]
    L_red = L[jnp.ix_(idx_Q, idx_Vpq)]

    top = jnp.concatenate([H_red, N_red], axis=1)
    bot = jnp.concatenate([M_red, L_red], axis=1)
    J_red = jnp.concatenate([top, bot], axis=0)

    return J_red, idx_theta, idx_Vpq, idx_P, idx_Q

# ----------------------------
# AC-PTDF for one transfer s->t
# ----------------------------
def ac_ptdf_single(theta, V, Ybus, branches, bus_types, s, t):
    """
    Returns ΔP_line per 1 MW transfer s->t (vector len = n_lines).
    """
    J_red, idx_theta, idx_Vpq, idx_P, idx_Q = build_reduced_jacobian(theta, V, Ybus, bus_types)

    nb = theta.shape[0]
    dP = jnp.zeros(nb).at[s].add(1.0).at[t].add(-1.0)
    dQ = jnp.zeros(nb)

    # Reduce RHS to match J_red rows: [ΔP(non-slack); ΔQ(PQ)]
    rhs = jnp.concatenate([dP[idx_P], dQ[idx_Q]], axis=0)

    # singular values
    S = jnp.linalg.svd(J_red, compute_uv=False)

    # Solve for state increments: [Δθ_non-slack; ΔV_PQ]
    dx = jnp.linalg.solve(J_red, rhs)

    # Expand to full Δθ, ΔV
    dtheta = jnp.zeros(nb).at[idx_theta].set(dx[:idx_theta.shape[0]])
    dV     = jnp.zeros(nb).at[idx_Vpq].set(dx[idx_theta.shape[0]:])

    # Line-flow sensitivity via JVP (exact first-order)
    def P_lines(th, Vm): return line_active_powers(th, Vm, branches)
    _, dP_lines = jax.jvp(lambda th, vm: P_lines(th, vm),
                          (theta, V), (dtheta, dV))
    return dP_lines, J_red

# ----------------------------
# Convenience: PTDF matrix for many pairs
# ----------------------------
def ac_ptdf_matrix_internal(theta, V, Ybus, branches, bus_types, sources, sinks):
    """
    sources/sinks: lists of equal length -> returns (n_lines x len(pairs)) PTDF matrix.
    """
    cols = [ac_ptdf_single(theta, V, Ybus, branches, bus_types, s, t)
            for s, t in zip(sources, sinks)]
    return jnp.stack(cols, axis=1)

def singular_values_of_jacobian(theta, V, Ybus, bus_types):
    J_red, _, _, _, _ = build_reduced_jacobian(theta, V, Ybus, bus_types)
    return jnp.linalg.svd(J_red, compute_uv=False)

def build_branches(monee_net):
    return [(b.from_node_id,
                        b.to_node_id,
                        b.model.br_r,
                        b.model.br_x,
                        b.model.g_fr,
                        b.model.b_fr,
                        b.model.g_to,
                        b.model.b_to,
                        b.model.tap,
                        b.model.shift) for b in monee_net.branches_by_type(mm.GenericPowerBranch) if b.active and b.model.on_off == 1]

#  branches: (f, t, r, x, g_fr, b_fr, g_to, b_to, tap, shift_deg)
def build_ybus(monee_net):
    monee_branches = build_branches(monee_net)
    return monee_branches, build_ybus_internal(len(monee_net.nodes_by_type(mm.Bus)), monee_branches)

def ac_ptdf_matrix(monee_net, from_nodes=[0], to_nodes=[0]):
     # 0=PQ,1=PV,2=Slack
    bus_types = [2 if monee_net.has_any_child_of_type(n, mm.ExtPowerGrid) else 0 
                 for n in monee_net.nodes_by_type(mm.Bus)]
    bus_types = jnp.array(bus_types)

    branches, Ybus = build_ybus(monee_net)
    print(branches)
    # Operating point (from your solved AC PF/OPF)
    theta = jnp.array([mm.value(n.model.va_radians) for n in monee_net.nodes_by_type(mm.Bus)])
    V = jnp.array([mm.value(n.model.vm_pu) for n in monee_net.nodes_by_type(mm.Bus)])
    print(V)
    # PTDFs for transfers 0→3 and 1→4
    PTDF = ac_ptdf_matrix_internal(theta, V, Ybus, branches, bus_types, sources=from_nodes, sinks=to_nodes)
    print(PTDF)  # shape (n_lines, 2)

if __name__ == "__main__":
    net = mes.create_monee_benchmark_net()
    result = monee.run_energy_flow(net)
    print(result)
    ac_ptdf_matrix(result.network, from_nodes=[4], to_nodes=[1])


