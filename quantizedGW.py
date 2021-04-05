import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import ot
import random

from networkx.algorithms.community.asyn_fluid import asyn_fluidc
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
from networkx.algorithms.shortest_paths.generic import shortest_path_length

import pickle

from sklearn.metrics.pairwise import pairwise_distances

from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix

import time

from random import sample


""""
---quantized Gromov Wasserstein---
The main algorithm is here.
Variants are below (quantized Fused GW and a version specifically for point clouds).
"""

def renormalize_prob(pv):
    # Robust method to turn an arbitrary vector into a probability vector
    q = pv.copy()
    if pv.sum() > 1:
        diff = pv.sum()-1
        q[q.argmax()] -= diff # take off mass from the heaviest
    elif pv.sum() < 1:
        diff = 1-pv.sum()
        q[q.argmin()] += diff # add mass to the lightest

    return q


def deterministic_coupling(Dist,p,node_subset):

    n = Dist.shape[0]

    # Get distance matrix from all nodes to the subset nodes
    D_subset = Dist[:,node_subset]

    # Find shortest distances to the subset
    dists_to_subset = np.sort(D_subset)[:,0]

    # Construct the coupling
    coup = np.zeros([n,n])
    for j in range(n):
        # Find nodes in `node_subset` which are nearest neighbors to the query node
        closest_in_subset = [node_subset[k] for k in list(np.argwhere(D_subset[j,:] == dists_to_subset[j]).T[0])]
        # Divide mass evenly from query node to its nearest subset neighbors
        mass = p[j]/len(closest_in_subset)
        # Place mass in appropriate positions of the coupling
        for k in closest_in_subset:
            coup[j,k] = mass

    return coup

def compress_graph_from_subset(Dist,p,node_subset):

    coup = deterministic_coupling(Dist,p,node_subset)
    p_compressed = renormalize_prob(np.squeeze(np.array(np.sum(coup, axis = 0))))

    return coup, p_compressed


def compress_graph(Dist,p_compressed):

    good_inds = [j for j in range(len(p_compressed)) if p_compressed[j] > 0]

    Dist_new = Dist[np.ix_(good_inds,good_inds)]

    p_new = renormalize_prob(np.array([p_compressed[j] for j in range(len(p_compressed)) if p_compressed[j] > 0]))

    return Dist_new, p_new

def find_submatching(pm,pn):

    Distm = np.eye(len(pm))
    Distn = np.eye(len(pn))

    coup_sub, log = gwa.gromov_wasserstein(Distm, Distn, pm, pn)

    return coup_sub


def find_support(p_compressed):

    supp = list(np.argwhere(p_compressed > 0).ravel())

    return supp

def compress_graph_partition(Dist,node_subset,p):

    size = Dist.shape[0]

    sorted_dists = np.sort(Dist[:,node_subset])
    p_compressed = np.zeros(len(node_subset))
    p_compressed_full = np.zeros(size)
    coup = np.zeros([size,size])

    for j in range(size):
        target_node_idx = random.choice(np.argwhere(Dist[j,node_subset] == sorted_dists[j,0]).ravel())
        p_compressed[target_node_idx] += p[j]
        p_compressed_full[node_subset[target_node_idx]] += p[j]
        coup[j,node_subset[target_node_idx]] = p[j]

    p_compressed = p_compressed/np.sum(p_compressed)
    p_compressed_full = p_compressed_full/np.sum(p_compressed_full)

    Dist_compressed = Dist[node_subset,:][:,node_subset]

    return Dist_compressed, p_compressed, p_compressed_full, coup

def find_submatching_locally_linear(Dist1,Dist2,coup1,coup2,i,j):

    subgraph_i = find_support(coup1[:,i])
    p_i = coup1[:,i][subgraph_i]/np.sum(coup1[:,i][subgraph_i])

    subgraph_j = find_support(coup2[:,j])
    p_j = coup2[:,j][subgraph_j]/np.sum(coup2[:,j][subgraph_j])

    x_i = list(Dist1[i,:][subgraph_i].reshape(len(subgraph_i),))
    x_j = list(Dist2[j,:][subgraph_j].reshape(len(subgraph_j),))

    coup_sub_ij = ot.emd_1d(x_i,x_j,p_i,p_j,p=2)

    return coup_sub_ij

"""
Main Algorithm
"""

def compressed_gw(Dist1,Dist2,p1,p2,node_subset1,node_subset2, verbose = False, return_dense = True):

    """
    In:
    Dist1, Dist2 --- distance matrices of size nxn and mxm
    p1,p2 --- probability vectors of length n and m
    node_subset1, node_subset2 ---  subsets of point indices. This version of the qGW code
                                    specifically uses Voronoi partitions from fixed subsets
                                    (usually these are chosen randomly). Other partitioning schems
                                    are possible, but not currently implemented here.
    verbose --- print status and compute times
    return_dense --- some parts of the algorithm use sparse matrices. If 'False' a sparse matrix is returned.

    Out:
    full_coup --- coupling matrix of size nxm giving a probabilistic correspondence between metric spaces.
    """
    # Compress Graphs
    start = time.time()
    if verbose:
        print('Compressing Graphs...')

    coup1, p_compressed1 = compress_graph_from_subset(Dist1,p1,node_subset1)
    coup2, p_compressed2 = compress_graph_from_subset(Dist2,p2,node_subset2)

    Dist_new1, p_new1 = compress_graph(Dist1,p_compressed1)
    Dist_new2, p_new2 = compress_graph(Dist2,p_compressed2)

    if verbose:
        print('Time for Compressing:', time.time() - start)

    # Match compressed graphs
    start = time.time()
    if verbose:
        print('Matching Compressed Graphs...')
    coup_compressed, log = gwa.gromov_wasserstein(Dist_new1, Dist_new2, p_new1, p_new2)
    if verbose:
        print('Time for Matching Compressed:', time.time() - start)

    # Find submatchings and create full coupling
    if verbose:
        print('Matching Subgraphs and Constructing Coupling...')
    supp1 = find_support(p_compressed1)
    supp2 = find_support(p_compressed2)

    full_coup = coo_matrix((Dist1.shape[0], Dist2.shape[0]))

    matching_time = 0
    matching_and_expanding_time = 0
    num_local_matches = 0

    for (i_enum, i) in enumerate(supp1):
        subgraph_i = find_support(coup1[:,i])
        for (j_enum, j) in enumerate(supp2):
            start = time.time()
            w_ij = coup_compressed[i_enum,j_enum]
            if w_ij > 1e-10:
                num_local_matches += 1
                subgraph_j = find_support(coup2[:,j])
                # Compute submatching
                coup_sub_ij = find_submatching_locally_linear(Dist1,Dist2,coup1,coup2,i,j)
                matching_time += time.time()-start
                # Expand to correct size
                idx = np.argwhere(coup_sub_ij > 1e-10)
                idx_i = idx.T[0]
                idx_j = idx.T[1]
                row = np.array(subgraph_i)[idx_i]
                col = np.array(subgraph_j)[idx_j]
                data = w_ij*np.array([coup_sub_ij[p[0],p[1]] for p in list(idx)])
                expanded_coup_sub_ij = coo_matrix((data, (row,col)), shape=(full_coup.shape[0], full_coup.shape[1]))
                # Update full coupling
                full_coup += expanded_coup_sub_ij
                matching_and_expanding_time += time.time()-start

    if verbose:
        print('Total Time for',num_local_matches,'local matches:')
        print('Local matching:', matching_time)
        print('Local Matching Plus Expansion:', matching_and_expanding_time)

    if return_dense:
        return full_coup.toarray()
    else:
        return full_coup


"""-----Tall Skinny-----

Graph version of compressed_gw that operates on tall, skinny distance matrices
- This avoids the need to compute a full distance matrix, favoring Dijkstra over Floyd-Warshall
"""
def compress_graph_from_subset_ts(dists,p,node_subset):
    """
    Takes in a tall, skinny distance matrix and returns distances
    between anchors as well as compression mapping for measures.
    """
    # Mimics `deterministic coupling` function
    coup = []
    dists_to_subset = np.array([np.min(dists[v,:]) for v in range(dists.shape[0])])
    for j in range(dists.shape[0]):
        idx_of_closest_in_subset = np.where(dists[j,:]==dists_to_subset[j])[0] #sending all mass to first loc
                                                                    #could speed this via tolist().index()
        # Divide mass evenly from query node to its nearest subset neighbors
        mass = p[j]/len(idx_of_closest_in_subset)
        # Place mass in appropriate positions of the coupling
        for idx in idx_of_closest_in_subset:
            coup.append((j,idx,mass))

    coup = np.array(coup)
    coup = coo_matrix((coup[:,2],(coup[:,0],coup[:,1])),shape=dists.shape).tocsr()

    p_new = coup.sum(axis=0)
    dists_new = dists[node_subset,:]
    return dists_new, p_new, coup

def find_submatching_locally_linear_ts(dists1,dists2,coup1,coup2,i_enum,j_enum):
    """
    Compute locally linear matching assuming tall, skinny distance matrices

    Parameters:
    dists1, dists2 : tall skinny ndarrays
    coup1, coup2 : tall skinny csr matrices
    i_enum, j_enum : anchor node indices in tall skinny representation

    """
    subgraph_i = coup1[:,i_enum].nonzero()[0]
    subgraph_j = coup2[:,j_enum].nonzero()[0]

    p_i = coup1[subgraph_i,i_enum].toarray()
    p_i /= np.sum(p_i)
    p_j = coup2[subgraph_j,j_enum].toarray()
    p_j /= np.sum(p_j)

    x_i = dists1[subgraph_i,i_enum]
    x_j = dists2[subgraph_j,j_enum]

    coup_sub_ij = ot.emd_1d(x_i,x_j,p_i.reshape(-1),p_j.reshape(-1),p=2)

    return coup_sub_ij

def compressed_gw_ts(dists1,dists2,p1,p2,node_subset1,node_subset2,
                     verbose = False, return_dense = True, tol=1e-10):
    """
    Compressed GW for tall skinny distance matrices

    """
    # Compress Graphs
    start = time.time()
    if verbose:
        print('Compressing Graphs...')

    Dist_new1, p_new1,coup1 = compress_graph_from_subset_ts(dists1,p1,node_subset1)
    Dist_new2, p_new2,coup2 = compress_graph_from_subset_ts(dists2,p2,node_subset2)

    if verbose:
        print('Time for Compressing:', time.time() - start)

    # Match compressed graphs
    start = time.time()
    if verbose:
        print('Matching Compressed Graphs...')
    coup_compressed, log = gwa.gromov_wasserstein(Dist_new1, Dist_new2, np.array(p_new1).reshape(-1), np.array(p_new2).reshape(-1))
    if verbose:
        print('Time for Matching Compressed:', time.time() - start)

    # Find submatchings and create full coupling
    if verbose:
        print('Matching Subgraphs and Constructing Coupling...')
    supp1 = node_subset1 #find_support(p_compressed1)
    supp2 = node_subset2 #find_support(p_compressed2)

    full_coup = coo_matrix((dists1.shape[0], dists2.shape[0]))

    matching_time = 0
    matching_and_expanding_time = 0
    num_local_matches = 0

    for (i_enum, i) in enumerate(supp1):
        subgraph_i = coup1[:,i_enum].nonzero()[0]
        for (j_enum, j) in enumerate(supp2):
            start = time.time()
            w_ij = coup_compressed[i_enum,j_enum]
            if w_ij > tol:
                num_local_matches += 1
                subgraph_j = coup2[:,j_enum].nonzero()[0]
                # Compute submatching
                coup_sub_ij = find_submatching_locally_linear_ts(dists1,dists2,coup1,coup2,i_enum,j_enum)
                matching_time += time.time()-start
                # Expand to correct size
                idx = np.argwhere(coup_sub_ij > tol)
                idx_i = idx.T[0]
                idx_j = idx.T[1]
                row = np.array(subgraph_i)[idx_i]
                col = np.array(subgraph_j)[idx_j]
                data = w_ij*np.array([coup_sub_ij[p[0],p[1]] for p in list(idx)])
                expanded_coup_sub_ij = coo_matrix((data, (row,col)), shape=(full_coup.shape[0], full_coup.shape[1]))
                # Update full coupling
                full_coup += expanded_coup_sub_ij
                matching_and_expanding_time += time.time()-start

    if verbose:
        print('Total Time for',num_local_matches,'local matches:')
        print('Local matching:', matching_time)
        print('Local Matching Plus Expansion:', matching_and_expanding_time)

    if return_dense:
        return full_coup.toarray()
    else:
        return full_coup


def get_compressed_coupling_ts(dists1,dists2,p1,p2,node_subset1,node_subset2):
    """
    Compute compressed coupling from tall skinny distance matrices

    Returns:
    coup1, coup2 : tall skinny matrices giving mass transport due to compression
    coup_compressed : compressed coupling


    """
    Dist_new1, p_new1,coup1 = compress_graph_from_subset_ts(dists1,p1,node_subset1)
    Dist_new2, p_new2,coup2 = compress_graph_from_subset_ts(dists2,p2,node_subset2)
    coup_compressed, log = gwa.gromov_wasserstein(Dist_new1, Dist_new2,
                                                  np.array(p_new1).reshape(-1),
                                                  np.array(p_new2).reshape(-1))
    return coup1, coup2, coup_compressed

def query_point_coupling_ts(x_idx,dists1,dists2,coup1,coup2,coup_compressed):
    """
    One-point coupling: Given a query point, return its row in the coupling

    Parameters:
    x_idx : query index
    dists1, dists2 : tall skinny distance matrices
    coup1, coup2 : tall skinny compression-mass transport matrices
    coup_compressed : compressed coupling between anchor points

    """
    cx = np.zeros((1,coup2.shape[0])) # initialize probability vector that x maps to
    pa_idx = coup1[x_idx,:].nonzero()[1] # Get parent indices
    for A in pa_idx:
        chA = coup1[:,A].nonzero()[0]
        idx_x_in_chA = chA.tolist().index(x_idx)
        locs = np.where(coup_compressed[A,:] > 1e-10)[0]
        for B in locs:
            wAB = coup_compressed[A,B]
            chB = coup2[:,B].nonzero()[0]
            coup_sub_AB = find_submatching_locally_linear_ts(dists1,dists2,coup1,coup2,A,B)
            for idx, val in enumerate(coup_sub_AB[idx_x_in_chA,:]):
                if val > 1e-10:
                    cx[0,chB[idx]] += wAB*val
    return cx


"""
-----Graph FGW from partitions-----

Graph version of compressed_fgw that operates on "sparse" tall-skinny distance matrices
derived from a partition
- distances are only computed from a point to its anchor
- slight changes to the code that take advantage of partition structure
"""
def wl_label(G,degrees):
    """
    Weisfeiler-Lehman update
    G : NetworkX graph
    degrees : dict of node names and one-hot encodings
    """
    ndegs = {}
    for key in degrees.keys(): #iterate over each node
        ndegs[key] = degrees[key].copy()
        for v in G.neighbors(key): #iterate over neighbors
            ndegs[key] += degrees[v]
    return ndegs

def partition_featurize_graph_fpdwl(G,k=100,dims=64,wl_steps=1,
                                    distribution_offset=0,distribution_exponent=0):
    """
    Partition+Anchor a graph using Fluid communities+Pagerank and produce node features using Degree+WL
    (Hence fpdwl)
    -----------
    Parameters:
    G : NetworkX graph
    k : number of blocks in partition
    dims : dimension of feature space
    wl_steps : number of Weisfeiler-Lehman aggregations to carry out
    -------
    Returns:
    p : dict with keys=node labels and values=probabilities on nodes
    partition : list of sets containing node labels
    node_subset : list of anchor node labels
    dists : distances between anchors
    features : degree+WL based node features
    """
    pr = pagerank(G)
    # Partition graph via Fluid
    partition_iter = asyn_fluidc(G,k)
    partition = []
    for i in partition_iter:
        partition.append(i)

    # Create anchors via PageRank
    anchors = []
    for p in partition:
        part_pr = {}
        for s in p:
            part_pr[s] = pr[s]
        anchors.append(max(part_pr, key=part_pr.get))
    anchors = sorted(anchors) # Fix an ordering on anchors

    # Featurize using degrees and Weisfeiler-Lehman
    degrees = dict(nx.degree(G))
    # One-hot encoding of degrees
    for key in degrees.keys():
        deg = degrees[key]
        feat = np.zeros(dims)
        if deg < dims:
            feat[deg]+=1 #Create one-hot encoding
        degrees[key] = feat #Replace scalar degree with one-hot vector
    for i in range(wl_steps):
        degrees = wl_label(G,degrees)
    # Rename, obtain sorted node names and features
    features = degrees
    a,b = list(zip(*sorted(features.items())))
    nodes = list(a)
    features = np.array(b)

    # Obtain probability vector
    p = np.array([(G.degree(n)+distribution_offset)**distribution_exponent for n in nodes])
    p = p/np.sum(p)

    # Rename anything else
    node_subset = anchors
    node_subset_idx = [nodes.index(v) for v in node_subset] #indices of anchor nodes in node list

    return nodes, features, p, partition, node_subset, node_subset_idx


def compress_graph_from_hard_partition_ts(G,nodes,features,p,partition,node_subset):
    """
    Obtain a sparse tall-skinny matrix and new probabilities from a hard partition of a graph.
    For each point, we only find the distance to its anchor, not to all other anchors.
    -----------
    Parameters:
    G : NetworkX graph
    nodes : sorted list of graph nodes
    p : probability vector of sorted nodes
    partition : list of sets containing node labels
    node_subset : sorted list of anchor node labels
    -------
    Returns:
    dists : |nodes|x|node_subset| matrix of distances from each
                block of partition to anchor in that block
    membership : |nodes|x|node_subset| membership matrix
    p_compressed : vector of aggregated probabilities on anchors
    """

    # Distances between anchors
    dists_subset = np.zeros((len(node_subset),len(node_subset)))
    for i in range(len(node_subset)):
        for j in range(i+1,len(node_subset)):
            dists_subset[i,j] = shortest_path_length(G,node_subset[i],node_subset[j])
    dists_subset = dists_subset + dists_subset.T

    # Sparse tall-skinny matrix of distances and feature-vector distances from points to their own anchors
    # Also, tall-skinny membership matrix and mass-compression matrix
    row_idx, col_idx, dist_data, mass_data, fdist_data = [], [], [], [], []
    for (aidx,anchor) in enumerate(node_subset):
        bidx = [anchor in v for v in partition].index(True) #block containing current anchor point
        block = partition[bidx]
        for b in block:
            idx = nodes.index(b)
            d = shortest_path_length(G,nodes[idx],anchor)
            fd = pairwise_distances(features[nodes.index(anchor),:].reshape(1,-1),
                                    features[idx,:].reshape(1,-1))[0][0]
            row_idx.append(idx)
            col_idx.append(aidx)
            dist_data.append(d)
            mass_data.append(p[idx])
            fdist_data.append(fd)

    dists = coo_matrix((dist_data, (row_idx, col_idx)),shape=(len(nodes), len(node_subset)))
    fdists = coo_matrix((fdist_data, (row_idx, col_idx)),shape=(len(nodes), len(node_subset)))
    membership = coo_matrix(([1 for v in row_idx], (row_idx, col_idx)),shape=(len(nodes), len(node_subset)))
#     coup = coo_matrix((mass_data, (row_idx, col_idx)),shape=(len(nodes), len(node_subset)))

    p_subset = csr_matrix.dot(p, membership)
    return dists.tocsr(),fdists.tocsr(),membership.tocsr(),p_subset, dists_subset


def compressed_fgw(dists1,dists2,fdists1,fdists2,
                   membership1,membership2,
                   features1,features2,p1,p2,
                   node_subset_idx1,node_subset_idx2,
                  dists_subset1,dists_subset2,
                  p_subset1,p_subset2, alpha=0.5,beta=0.5,verbose = False, return_dense = True):

    """
    Compressed FGW on partitioned data structures
    -----------
    Parameters:
    dists1,dists2,fdists1,fdists2,membership1,membership2 : |nodes| x |node_subset| csr matrices
    features1,features2 : |nodes| x |features| ndarrays
    p1,p2 : |nodes| x 1 ndarrays
    node_subset_idx1, node_subset_idx2 : |node_subset| lists
    dists_subset1, dists_subset2 : |node_subset| x |node_subset| ndarrays
    p_subset1, p_subset2 : |node_subset| x 1 ndarrays
    -----------
    Returns:
    full_coup : |nodes| x |nodes| csr matrix
    """


    M_compressed = pairwise_distances(features1[node_subset_idx1,:],features2[node_subset_idx2,:])
    # Match compressed graphs
    start = time.time()
    if verbose:
        print('Matching Compressed Graphs...')
    coup_compressed = ot.gromov.fused_gromov_wasserstein(M_compressed,
                                                         dists_subset1, dists_subset2,
                                                         p_subset1, p_subset2, alpha = alpha)

    if verbose:
        print('Time for Matching Compressed:', time.time() - start)

    # Find submatchings and create full coupling
    if verbose:
        print('Matching Subgraphs and Constructing Coupling...')
    full_coup = coo_matrix((dists1.shape[0], dists2.shape[0]))

    matching_time = 0
    matching_and_expanding_time = 0
    num_local_matches = 0

    for (i_enum, i) in enumerate(node_subset_idx1):
        subgraph_i = list(membership1[:,i_enum].nonzero())[0] #get indices anchored to i
        for (j_enum, j) in enumerate(node_subset_idx2):
            start = time.time()
            w_ij = coup_compressed[i_enum,j_enum]
            if w_ij > 1e-10:
                num_local_matches += 1
                subgraph_j = list(membership2[:,j_enum].nonzero())[0] #get indices anchored to j
                p_i = (p1[subgraph_i]/np.sum(p1[subgraph_i])).reshape(-1)
                p_j = (p2[subgraph_j]/np.sum(p2[subgraph_j])).reshape(-1)
                # Compute submatching based on graph distances
                if beta > 0:
                    coup_sub_dist_ij = ot.emd_1d(dists1[subgraph_i,i_enum].toarray(),
                                              dists2[subgraph_j,j_enum].toarray(),
                                              p_i,p_j, p=2)
                else:
                    coup_sub_dist_ij = np.zeros([len(subgraph_i),len(subgraph_j)])
                # Compute submatching based on node features
                if beta < 1:
                    coup_sub_features_ij = ot.emd_1d(fdists1[subgraph_i,i_enum].toarray(),
                                              fdists2[subgraph_j,j_enum].toarray(),
                                              p_i,p_j,p=2)
                else:
                    coup_sub_features_ij = np.zeros([len(subgraph_i),len(subgraph_j)])
                # Take weighted average
                coup_sub_ij = (1-beta)*coup_sub_features_ij + beta*coup_sub_dist_ij
                matching_time += time.time()-start

                # Expand to correct size
                idx = np.argwhere(coup_sub_ij > 1e-10)
                idx_i = idx.T[0]
                idx_j = idx.T[1]
                row = np.array(subgraph_i)[idx_i]
                col = np.array(subgraph_j)[idx_j]
                data = w_ij*np.array([coup_sub_ij[p[0],p[1]] for p in list(idx)])
                expanded_coup_sub_ij = coo_matrix((data, (row,col)),
                                                  shape=(full_coup.shape[0], full_coup.shape[1]))
                # Update full coupling
                full_coup += expanded_coup_sub_ij
                matching_and_expanding_time += time.time()-start


    if verbose:
        print('Total Time for',num_local_matches,'local matches:')
        print('Local matching:', matching_time)
        print('Local Matching Plus Expansion:', matching_and_expanding_time)

    if return_dense:
        return full_coup.toarray()
    else:
        return full_coup



def partition_featurize_graphlist_fpdwl(graphs,k=100,dims=64,wl_steps=1,
                                    distribution_offset=0,distribution_exponent=0,verbose=True):
    """
    Preprocess list of graphs by creating partitions and computing information for locally linear FGW
    ----------
    Parameters:
    graphs : list of NetworkX graphs
    k : number of blocks in each partition
    dims : length of histogram used to bin degrees
    wl_steps : number of Weisfeiler-Lehman aggregations
    distribution_offset, distribution_exponent : offset and exponent parameters for probabilities
    ----------
    Returns:
    dataset : list of dicts, any pair can be consumed by compress_fgw_from_dicts
    """

    if verbose:
        print('Partitioning with',k,'blocks in each partition')

    dataset = []
    for idx,G in enumerate(graphs):
        if verbose:
            print('Starting with Graph',idx)

        start = time.time()
        nodes, features, p, partition, node_subset, node_subset_idx = partition_featurize_graph_fpdwl(G,
                                                                                                      k=k,dims=dims,
                                                                                                      wl_steps=wl_steps,
                                                                                                      distribution_offset=distribution_offset,
                                                                                                      distribution_exponent=distribution_exponent)

        if verbose:
            print('Partition+Featurize completed in', time.time() - start,'seconds')

        start = time.time()
        dists,fdists,membership,p_subset,dists_subset = compress_graph_from_hard_partition_ts(G,nodes,features,
                                                                                           p,partition,node_subset)
        if verbose:
            print('Distance primitives computed in', time.time() - start,'seconds')
        data = {}
        for i in ('nodes', 'features','p','partition','node_subset','node_subset_idx',
                 'dists','fdists','membership','p_subset','dists_subset'):
            data[i] = locals()[i]

        dataset.append(data)

    return dataset


def compress_fgw_from_dicts(data1,data2,alpha=0.1,beta=0.1,verbose = False, return_dense = True):
    """
    Apply compress_fgw to a pair of dicts containing precomputed data
    """
    dists1,dists2,fdists1,fdists2 = data1['dists'], data2['dists'],data1['fdists'], data2['fdists']
    membership1,membership2 = data1['membership'], data2['membership']
    features1,features2,p1,p2 = data1['features'], data2['features'],data1['p'], data2['p']
    node_subset_idx1,node_subset_idx2 = data1['node_subset_idx'], data2['node_subset_idx']
    dists_subset1,dists_subset2 = data1['dists_subset'], data2['dists_subset']
    p_subset1,p_subset2 = data1['p_subset'], data2['p_subset']

    start = time.time()
    full_coup12 = compressed_fgw(dists1,dists2,fdists1,fdists2,
                             membership1,membership2,
                   features1,features2,p1,p2,
                   node_subset_idx1,node_subset_idx2,
                  dists_subset1,dists_subset2,
                  p_subset1,p_subset2, alpha=alpha,beta=beta,verbose = verbose, return_dense = return_dense)

    print('Time for Matching:', time.time() - start,'seconds')

    return full_coup12


"""
The point cloud version (just assuming unique nearest neighbors).
"""


"""
--- quantized GW for Point Clouds ---

The code below uses the generic assumption that pairwise distances are unique.
This allows us to do certain steps more efficiently.
"""

def deterministic_coupling_point_cloud(Dist,p,node_subset):

    n = Dist.shape[0]

    # Get distance matrix from all nodes to the subset nodes
    D_subset = Dist[:,node_subset]

    # Find shortest distances to the subset
    dists_to_subset_idx = np.argmin(D_subset,axis = 1)

    # Construct the coupling
    row = list(range(n))
    col = [node_subset[j] for j in dists_to_subset_idx]
    data = p
    coup = coo_matrix((data,(row,col)),shape = (n,n))

    return coup


def compress_graph_from_subset_point_cloud(Dist,p,node_subset):
    """
    Update Feb 8, 2020: this is the version of `compress_graph_from_subset`
    that we're using for point cloud experiments -- sparse matrices help a lot
    """
    coup = deterministic_coupling_point_cloud(Dist,p,node_subset)
    p_compressed = renormalize_prob(np.squeeze(np.array(np.sum(coup, axis = 0))))


    return coup.toarray(), p_compressed

def compressed_gw_point_cloud(Dist1,Dist2,p1,p2,node_subset1,node_subset2, verbose = False, return_dense = True):

    # Compress Graphs
    start = time.time()
    if verbose:
        print('Compressing Graphs...')

    coup1, p_compressed1 = compress_graph_from_subset_point_cloud(Dist1,p1,node_subset1)
    coup2, p_compressed2 = compress_graph_from_subset_point_cloud(Dist2,p2,node_subset2)

    Dist_new1, p_new1 = compress_graph(Dist1,p_compressed1)
    Dist_new2, p_new2 = compress_graph(Dist2,p_compressed2)

    if verbose:
        print('Time for Compressing:', time.time() - start)

    # Match compressed graphs
    start = time.time()
    if verbose:
        print('Matching Compressed Graphs...')
    coup_compressed, log = ot.gromov.gromov_wasserstein(Dist_new1, Dist_new2, p_new1, p_new2,
                                                     'square_loss', verbose=False, log=True)

    # If coupling is dense, abort the algorithm and return a dense full size coupling.
    if np.sum(coup_compressed > 1e-10) > len(coup_compressed)**1.5:
        print('Dense Compressed Matching, returning dense coupling...')
        return p1[:,None]*p2[None,:]

    # coup_compressed, log = gwa.gromov_wasserstein(Dist_new1, Dist_new2, p_new1, p_new2)
    if verbose:
        print('Time for Matching Compressed:', time.time() - start)

    # Find submatchings and create full coupling
    if verbose:
        print('Matching Subgraphs and Constructing Coupling...')
    supp1 = find_support(p_compressed1)
    supp2 = find_support(p_compressed2)

    full_coup = coo_matrix((Dist1.shape[0], Dist2.shape[0]))

    matching_time = 0
    matching_and_expanding_time = 0
    num_local_matches = 0

    for (i_enum, i) in enumerate(supp1):
        subgraph_i = find_support(coup1[:,i])
        for (j_enum, j) in enumerate(supp2):
            start = time.time()
            w_ij = coup_compressed[i_enum,j_enum]
            if w_ij > 1e-10:
                num_local_matches += 1
                subgraph_j = find_support(coup2[:,j])
                # Compute submatching
                coup_sub_ij = find_submatching_locally_linear(Dist1,Dist2,coup1,coup2,i,j)
                matching_time += time.time()-start
                # Expand to correct size
                idx = np.argwhere(coup_sub_ij > 1e-10)
                idx_i = idx.T[0]
                idx_j = idx.T[1]
                row = np.array(subgraph_i)[idx_i]
                col = np.array(subgraph_j)[idx_j]
                data = w_ij*np.array([coup_sub_ij[p[0],p[1]] for p in list(idx)])
                expanded_coup_sub_ij = coo_matrix((data, (row,col)), shape=(full_coup.shape[0], full_coup.shape[1]))
                # Update full coupling
                full_coup += expanded_coup_sub_ij
                matching_and_expanding_time += time.time()-start

    if verbose:
        print('Total Time for',num_local_matches,'local matches:')
        print('Local matching:', matching_time)
        print('Local Matching Plus Expansion:', matching_and_expanding_time)

    if return_dense:
        return full_coup.toarray()
    else:
        return full_coup

"""
Loss of a GW coupling.

The code below is adapted from the Python Optimal Transport library to compute
the GW loss of a given coupling.
"""

def frobenius(A,B):
    return np.trace(np.matmul(np.transpose(A),B))

# Auxilliary function to implement the tensor product of [ref]
def init_matrix(C1, C2, T, p, q):

    def f1(a):
        return (a**2) / 2.0

    def f2(b):
        return (b**2) / 2.0

    def h1(a):
        return a

    def h2(b):
        return b

    constC1 = np.dot(np.dot(f1(C1), p.reshape(-1, 1)),
                     np.ones(len(q)).reshape(1, -1))
    constC2 = np.dot(np.ones(len(p)).reshape(-1, 1),
                     np.dot(q.reshape(1, -1), f2(C2).T))
    constC = constC1 + constC2
    hC1 = h1(C1)
    hC2 = h2(C2)

    return constC, hC1, hC2

# Define the tensor product from [ref]
def tensor_product(constC, hC1, hC2, T):
    A = -np.dot(hC1, T).dot(hC2.T)
    tens = constC + A
    return tens

# Define the loss function for GW distance.
def gwloss(constC, hC1, hC2, T):
    tens = tensor_product(constC, hC1, hC2,T)
    return frobenius(tens,T)

def gwloss_init(C1, C2, p, q, G0):
    constC, hC1, hC2 = init_matrix(C1,C2,G0,p,q)
    return gwloss(constC, hC1, hC2,G0)
