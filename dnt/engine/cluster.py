import numpy as np

def cluster_by_gap(arr:np.ndarray, gap:int)->list:
    '''
    Parameters:
    - arr: a 1d array for clustering
    - gap: the threshold to break (> gap)

    Returns:
    - list of clustered numbers (lists)
    '''    
    # Sort the array to process it in order
    arr.sort()
    clusters = []
    current_cluster = [arr[0]]

    # Iterate through the sorted list
    for i in range(1, len(arr)):
        if arr[i] - arr[i-1] <= gap:
            # If the gap condition is met, add to the current cluster
            current_cluster.append(arr[i])
        else:
            # If the gap condition is not met, start a new cluster
            clusters.append(current_cluster)
            current_cluster = [arr[i]]

    # Add the last cluster
    clusters.append(current_cluster)

    return clusters

