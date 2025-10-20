from scipy.stats import uniform, expon, poisson
import numpy as np
from TemporalNetwork import ContTempNetwork

def EDEDE(inter_tau = 2, activ_tau = 2, t_start = 0, t_end = 300):

    number_of_events = poisson.rvs(size = 1, mu = (t_end - t_start) / activ_tau)[0]
    starting_times = np.sort(uniform.rvs(size=number_of_events, loc=t_start, scale=t_end - t_start))
    ending_times = expon.rvs(size=number_of_events, scale= inter_tau)
    ending_times = ending_times + starting_times
    return number_of_events, starting_times, ending_times


def make_step_block_probs(deltat1,
                          n_groups=27, n_per_group = 3,
                          basis_num_communities = 3, powers_num_communities = [3, 2, 1], list_p_within_community = [49/50] * len([27, 9, 3])):
    """
    Returns a function that generates the block probability matrix as a function of time.

    Parameters:
    - deltat1 (int): Length of each temporal step.
    - n_groups (int): Total number of cores in the network.
    - basis_num_communities (int): The base number of communities for symmetry.
    - powers_num_communities (list of int): Powers of the base defining the number of communities in each stage.
    - list_p_within_community (list of float): List of within-community interaction probabilities for each stage.

    """
    
    list_num_communities=list(np.power(basis_num_communities, powers_num_communities))
    
    def calculate_pout(p_within, community_size, total_size):
        k = (1 / p_within - community_size) / (total_size - community_size)
        return k * p_within

    def generate_block_matrix(num_communities, p_within_base):
        # Determine community sizes
        community_size = n_groups // num_communities
        
        p_within = p_within_base / community_size
        p_within_outgroup = p_within_base / (community_size * n_per_group -1)  # Adjust to avoid self_events
        
        pout = calculate_pout(p_within, community_size, n_groups) / n_per_group

        # Create block matrix
        block_matrix = np.zeros((n_groups*n_per_group, n_groups*n_per_group))
        for i in range(num_communities):
            start = i * community_size*n_per_group
            end = start + community_size*n_per_group
            block_matrix[start:end, start:end] = p_within_outgroup 
 
        block_matrix[block_matrix == 0] = pout
        np.fill_diagonal(block_matrix, 0 , wrap=False) # Correction for self_events
        
        return block_matrix

    # Precompute the matrices for different stages
    stage_matrices = []
    for num_communities, p_within_community in zip(list_num_communities, list_p_within_community):
        stage_matrices.append(generate_block_matrix(num_communities, p_within_community))

    def block_mod_func(t):
        total_stages = len(stage_matrices)
        stage_duration = deltat1

        for stage_index in range(total_stages):
            if t >= stage_index * stage_duration and t < (stage_index + 1) * stage_duration:
                return stage_matrices[stage_index]

        print("Warning: t is out of bounds. Returning identity matrix.")
        return np.eye(n_groups)

    return block_mod_func


def generate_evolving_SBM(inter_tau = 2, activ_tau = 2,
                          n_per_group = 10, n_groups = 27,
                          t_start = 0, t_end = 300,
                          basis_num_communities = 3, powers_num_communities = [3, 2, 1], list_p_within_community = [49/50] * len([27, 9, 3])):

    number_of_events, starting_times, ending_times = EDEDE(inter_tau = inter_tau, activ_tau = activ_tau, t_start = t_start, t_end = t_end)

    source_nodes = np.random.choice(n_groups * n_per_group, number_of_events, replace=True)
    block_mod_func = make_step_block_probs(deltat1 = (t_end - t_start) / len(powers_num_communities),
                                           n_groups=n_groups, n_per_group=n_per_group,
                                           basis_num_communities=basis_num_communities, powers_num_communities=powers_num_communities, list_p_within_community=list_p_within_community)


    target_nodes = []
    for i, source in enumerate(source_nodes):
        
        target_nodes.append(np.random.choice(n_groups * n_per_group, 1, p=block_mod_func(starting_times[i])[source])[0])
    
    temporal_net = ContTempNetwork(source_nodes=source_nodes,
                            target_nodes=target_nodes,
                            starting_times=starting_times,
                            ending_times=ending_times,
                            merge_overlapping_events=True)
    
    return temporal_net