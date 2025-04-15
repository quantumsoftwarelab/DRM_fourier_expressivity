import itertools

import dask
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch
from dask import compute, delayed
from tqdm import tqdm


def plot_element_comparison(
    i, i1, i1p, j, j1, j1p, integral_haar, exp_term_k, show_matrices=True
):
    plt.figure(figsize=(8, 2))

    # Updated title for better readability and formatting
    plt.suptitle(
        f"Comparison of Components for i={i} ({i1}, {i1p}) and j={j} ({j1}, {j1p})",
        fontsize=14,
        y=1.1,
    )

    plt.subplot(1, 3, 1)
    plt.imshow(integral_haar, cmap="viridis")
    plt.title("Haar Integral", fontsize=12)
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(torch.real(exp_term_k), cmap="viridis")
    plt.colorbar()
    plt.title("Real Part", fontsize=12)

    plt.subplot(1, 3, 3)
    plt.imshow(torch.imag(exp_term_k), cmap="cividis")
    plt.title("Imaginary Part", fontsize=12)
    plt.colorbar()

    # Adjust subplot spacing and layout for better presentation
    plt.subplots_adjust(
        left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.3
    )

    plt.show()


def haar_matrix(j1, i1p, j1p, i1, N, device="cpu"):
    """computes the haar matrix corresponding to E[W|j1><j1'|W* otimes W|i1'><i1|W*]"""

    def d(i, j):
        return i == j

    haar_matrix = torch.zeros((N, N, N, N), device=device)
    for k1 in range(N):
        for l1 in range(N):
            for k2 in range(N):
                for l2 in range(N):
                    haar_matrix[l1, l2, k1, k2] = (
                        d(l1, k1) * d(l2, k2) * d(j1, j1p) * d(i1, i1p)
                        + d(l1, k2) * d(k1, l2) * d(j1, i1) * d(i1p, j1p)
                    ) / (N**2 - 1) - (
                        d(l1, k1) * d(l2, k2) * d(j1, i1) * d(i1p, j1p)
                        + d(l1, k2) * d(k1, l2) * d(j1, j1p) * d(i1, i1p)
                    ) / (
                        N * (N**2 - 1)
                    )

    haar_matrix = haar_matrix.reshape((N**2, N**2))

    return haar_matrix


def epsilon_monomial_batch_dask(Ws, device="cpu", show_matrices=True):
    n_exp = Ws.shape[0]
    N = Ws.shape[1]
    outer1_einsum = torch.einsum("bij,bkl->bikjl", Ws, Ws).reshape(n_exp, N**2, N**2)
    outer2_einsum = torch.transpose(outer1_einsum, 1, 2).conj()

    BATCH_SIZE = 64
    num_batches = (N**2 + BATCH_SIZE - 1) // BATCH_SIZE
    all_distances = []

    @delayed
    def compute_batch_distance(start_idx, end_idx, outer1_einsum, outer2_einsum, n_exp, N):
        batch_distance = []
        for j in range(start_idx, end_idx):
            j1, j1p = j // N, j % N
            jp = j1p * N + j1
            if jp <= j:
                matrix_ket = outer1_einsum[:, j]
                bra_matrix = outer2_einsum[:, :, jp]
                integral_theta = torch.einsum("bl,bm->lm", matrix_ket, bra_matrix) / n_exp
                integral_haar = haar_matrix(j1, j1p, j1p, j1, N, device=device)
                max_distance = torch.max(torch.abs(integral_haar - integral_theta)).item()
                batch_distance.append(max_distance)
        return max(batch_distance)

    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, N**2)
        batch_distance = compute_batch_distance(start_idx, end_idx, outer1_einsum, outer2_einsum, n_exp, N)
        all_distances.append(batch_distance)

    all_distances_results = compute(*all_distances)
    return np.max(all_distances_results) * N**2


def epsilon_monomial(Ws, device="cpu", show_matrices=True):
    n_exp = Ws.shape[0]
    N = Ws.shape[1]
    outer1_einsum = torch.einsum("bij,bkl->bikjl", Ws, Ws).reshape(n_exp, N**2, N**2)
    outer2_einsum = torch.transpose(outer1_einsum, 1, 2).conj()

    BATCH_SIZE = 64
    num_batches = (N**2 + BATCH_SIZE - 1) // BATCH_SIZE
    distance = []

    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, N**2)

        for j in range(start_idx, end_idx):
            j1, j1p = j // N, j % N
            jp = j1p * N + j1
            if jp <= j:
                matrix_ket = outer1_einsum[:, j]
                bra_matrix = outer2_einsum[:, :, jp]
                integral_theta = torch.einsum("bl,bm->lm", matrix_ket, bra_matrix) / n_exp
                integral_haar = haar_matrix(j1, j1p, j1p, j1, N, device=device)

                max_distance = torch.max(torch.abs(integral_haar - integral_theta))
                distance.append(max_distance.detach().cpu().numpy())
                if show_matrices:
                    plot_element_comparison(j, j1, j1p, jp, j1p, j1, integral_haar, integral_theta)

    distance = np.array(distance)
    return np.max(distance) * N**2

def epsilon_monomial_orig(Ws, device="cpu", show_matrices=True):
    n_exp = Ws.shape[0]
    N = Ws.shape[1]
    outer1_einsum = torch.einsum("bij,bkl->bikjl", Ws, Ws).reshape(
        n_exp, N**2, N**2
    )
    outer2_einsum = torch.transpose(outer1_einsum, 1, 2).conj()

    distance = []
    for j in tqdm(range(N**2)):
        # Pass i and j as i1, j1, i1p, j1p to the haar_matrix function
        j1, j1p = j // N, j % N
        jp = j1p * N + j1
        if jp <= j:
            matrix_ket = outer1_einsum[:, j]
            bra_matrix = outer2_einsum[:, :, jp]
            integral_theta = torch.einsum("bl,bm->lm", matrix_ket, bra_matrix) / n_exp
            # Pass i and j as i1, j1, i1p, j1p to the haar_matrix function
            integral_haar = haar_matrix(j1, j1p, j1p, j1, N, device=device)

            distance.append(
                torch.max(torch.abs(integral_haar - integral_theta))
                .detach()
                .cpu()
                .numpy()
            )
            if show_matrices:
                plot_element_comparison(
                    j, j1, j1p, jp, j1p, j1, integral_haar, integral_theta
                )

    distance = np.array(distance)
    return np.max(distance) * N**2


def A_00_matrices(n_qubits, Model, n_exp, lpsa, paulis=False, device="cpu"):
    """
    Computes the first term of the expectation value
    exp[k]= E[<j1|W|0><0|W*|j1'> <i1'|W|0><0|W*|i1>] for all k in R
    """

    # Generate matrices W|0><0|
    seeds = torch.randint(0, 10000, (n_exp,), dtype=torch.int64)
    states = torch.zeros((n_exp, 2**n_qubits), dtype=torch.complex128, device=device)
    for i in range(n_exp):
        seed = seeds[i]
        torch.manual_seed(seed)

        if not paulis:
            circuit = Model(n_qubits, lpsa).to(device)
        else:
            circuit = Model(n_qubits, lpsa, paulis).to(device)
        with torch.no_grad():
            state = circuit([0]).reshape(2**n_qubits)
        states[i] = state
    m = torch.einsum("bi,bj->bij", states, torch.conj(states))
    return m


def first_term_exp(R_freqs, n_qubits, Model, n_exp, lpsa, device="cpu"):
    """
    Computes the first term of the expectation value
    exp[k]= E[<j1|W|0><0|W*|j1'> <i1'|W|0><0|W*|i1>] for all k in R
    """
    exp = torch.zeros(len(R_freqs) ** 2, dtype=torch.complex128, device=device)
    m = A_00_matrices(n_qubits, Model, n_exp, lpsa, device=device)

    k = 0
    for j1, j1p in R_freqs:
        for i1, i1p in R_freqs:
            exp[k] = torch.einsum("b,b->", m[:, j1, j1p], m[:, i1p, i1]) / n_exp
            k += 1
    return exp


def last_term_exp_optimized(
    R_freqs, n_qubits, Model, n_exp, lpsa, paulis=False, device="cpu"
):
    l = torch.zeros(len(R_freqs) ** 2, dtype=torch.complex128, device=device)
    N = 2**n_qubits
    distinct_indices = list(set(itertools.chain.from_iterable(R_freqs)))
    index_mapping = dict(
        zip(distinct_indices, [k for k in range(len(distinct_indices))])
    )

    seeds = torch.randint(0, 10000, (n_exp,), dtype=torch.int64)
    states1 = []
    states2 = []
    for i in range(n_exp):
        seed = seeds[i]
        torch.manual_seed(seed)

        if not paulis:
            circuit = Model(n_qubits, lpsa).to(device)
        else:
            circuit = Model(n_qubits, lpsa, paulis).to(device)
        with torch.no_grad():
            state = circuit(distinct_indices)

        state2 = torch.zeros_like(state)
        for j in range(n_qubits):
            state2 += Z(state, [j], n_qubits)  # APPLY QML Z HERE
        states1.append(state.reshape((N, -1)))
        states2.append(state2.reshape((N, -1)))

    k = 0
    states1 = torch.stack(states1)
    states2 = torch.stack(states2)

    for j1, j1p in R_freqs:
        for i1, i1p in R_freqs:
            l[k] = (
                torch.einsum(
                    "b,b->",
                    torch.einsum(
                        "bi,bi->b",
                        states1[:, :, index_mapping[j1]],
                        torch.conj(states2[:, :, index_mapping[j1p]]),
                    ),
                    torch.einsum(
                        "bi,bi->b",
                        states1[:, :, index_mapping[i1p]],
                        torch.conj(states2[:, :, index_mapping[i1]]),
                    ),
                )
                / n_exp
            )
            k += 1

    return l

def haar_var_deltas(i_1,ip_1,j_1,jp_1,C_1,C_2,freq):
    element = 0
    if i_1 == j_1 and ip_1 == jp_1:
        element += C_2
    if i_1 == ip_1 and j_1 == jp_1:
        element += C_1
        if i_1 == j_1:
            element += C_1 + C_2
def exp_haar(indices1, indices2, n_qubits):
    """
    exp_haar of the first layer
    """
    j1, j1p, j2, j2p = indices1
    i1p, i1, i2p, i2 = indices2
    N = 2**n_qubits

    def d(i, j):
        return i == j

    exp = (
        d(j2, j2p) * d(i2, i2p) * d(j1, j1p) * d(i1, i1p)
        + d(i2, j2) * d(j2p, i2p) * d(i1, j1) * d(j1p, i1p)
    ) / (N**2 - 1) - (
        d(j2, j2p) * d(i2, i2p) * d(i1, j1) * d(j1p, i1p)
        + d(i2, j2) * d(j2p, i2p) * d(j1, j1p) * d(i1, i1p)
    ) / (
        N * (N**2 - 1)
    )
    return torch.tensor(exp, dtype=torch.complex128)


def exp_haar_final(indices1, indices2, n_qubits):
    """
    exp haar of the second layer
    """
    _, _, j1, j1p = indices1
    _, _, i1p, i1 = indices2

    def d(i, j):
        return i == j

    N = 2**n_qubits
    C1 = -1 / (N**2 - 1)
    C2 = N / (N**2 - 1)
    exp = C1 * d(j1, j1p) * d(i1, i1p) + C2 * d(j1, i1) * d(j1p, i1p)
    return torch.tensor(exp, dtype=torch.complex128)


def calculate_results(R, n_qubits, Model, n_exp, lpsa, N, device="cpu"):
    C2 = N / (N**2 - 1)
    results = {"real_var": [], "res_haar": [], "res1": [], "res2": [], "res3": []}

    for freq in tqdm(R.keys(), desc="Calculating for each frequency"):
        res, res_exp, res_haar, res2, res3 = 0, 0, 0, 0, 0
        first_exp = first_term_exp(
            R[freq], n_qubits, Model, n_exp=n_exp, lpsa=lpsa, device=device
        )
        last_exp = last_term_exp_optimized(
            R[freq], n_qubits, Model, n_exp=n_exp, lpsa=lpsa, device=device
        )
        res = torch.einsum("k,k->", first_exp, last_exp)
        k = 0
        for j1, j1p in R[freq]:
            for i1, i1p in R[freq]:
                if i1 == j1 and j1p == i1p:
                    res2 += torch.abs((C2 - last_exp[k]) / (N * (N + 1)))
                    res3 += torch.abs(C2 * (1 / (N * (N + 1)) - first_exp[k]))

                indices = [(0, 0, j1, j1p), (0, 0, i1p, i1)]
                haar_exp = exp_haar(*indices, n_qubits)
                haar_final_exp = exp_haar_final(*indices, n_qubits)

                res_haar += haar_exp * haar_final_exp
                res_exp += torch.abs(
                    (haar_exp - first_exp[k]) * (haar_final_exp - last_exp[k])
                )
                k += 1

        results["real_var"].append(torch.real(res).cpu().detach().numpy())
        results["res_haar"].append(torch.real(res_haar).cpu().detach().numpy())
        results["res1"].append(torch.real(res_exp).cpu().detach().numpy())
        results["res2"].append(res2.cpu().detach().numpy())
        results["res3"].append(res3.cpu().detach().numpy())

    for key in results.keys():
        results[key] = np.array(results[key])

    return results


def calculate_bound_list(R, epsilon, N, res_haar_list):
    C2 = N / (N**2 - 1)
    bound_list_negl = []
    bound_list_all = []

    for i, freq in enumerate(R.keys()):
        bound1 = epsilon**2 * len(R[freq]) ** 2 / N**2
        bound2 = epsilon * len(R[freq]) / (N * (N + 1))
        bound3 = epsilon * C2 * len(R[freq]) / N**2
        bound_negl = bound2 + bound3 + abs(res_haar_list[i])
        bound_all = bound1 + bound_negl
        bound_list_negl.append(bound_negl)
        bound_list_all.append(bound_all)

    return np.array(bound_list_negl), np.array(bound_list_all)


def get_red_1layer_Pauli(n_qubits):
    """
    Computes eigenvalues of the global hamiltonian (one Pauli rotation on each qubit)
    """
    eigenvalues = [0.5, -0.5]
    init_eigs = [1, 0]  # counts of the first eigenvalue
    for i in range(n_qubits - 1):
        init_eigs = [x + 1 for x in init_eigs] + init_eigs
    Lambdas = [i - n_qubits / 2 for i in init_eigs]

    return Lambdas


def get_spectrum_Pauli1(n_qubits):
    """
    Computes R(w) from the eigenvalues of the global hamiltonian
    """
    all_sums = get_red_1layer_Pauli(n_qubits)
    m = [i for i in range(len(all_sums))]
    spectrum = list([(i[0] - i[1]) for i in itertools.combinations(all_sums, 2)])
    R = {k: [] for k in range(-n_qubits, n_qubits + 1)}
    for i in itertools.combinations(m, 2):
        R[(all_sums[i[0]] - all_sums[i[1]])].append(i)
        R[(all_sums[i[1]] - all_sums[i[0]])].append((i[1], i[0]))
    for element in m:
        R[0].append((element, element))
    R = {k: v for k, v in R.items() if k >= 0}
    return R

def create_filtered_circuit(original_qnode):
    non_encoding_ops = [op for op in original_qnode.qtape.operations if op.id is None]
    measurements = original_qnode.qtape.measurements

    def new_quantum_function(*qnode_args, **qnode_kwargs):
        # Reapply the non-encoding operations
        for op in non_encoding_ops:
            qml.apply(op)
        
        # Reapply the measurements
        return [qml.apply(m) for m in measurements]

    # Create a new QNode with the same device as the original one
    new_circuit = qml.QNode(new_quantum_function, original_qnode.device)
    return new_circuit

def fourier_coefficients(f, max_freq, cost,parallel=True):
    t = np.linspace(0, 2 * np.pi, max_freq, endpoint=False)
    if cost == "global":
        measure = f(t)[:,:,0]
    else:
        measure = f(t) 
    
    if parallel:
        y = np.fft.rfft(measure.T) / t.size
        return y 
    else:
        y = np.fft.rfft(measure) / t.size
        return np.fft.fftshift(y) 