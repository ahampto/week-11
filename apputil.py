def kmeans(X, k):
    """
    Perform k-means clustering on a  NumPy array 

    Parameters
    ----------
    X : np.ndarray
        A 2D NumPy array of shape (n_samples, n_features).
    k : int
        The number of clusters.

    Returns
    -------
    tuple
        (centroids, labels)
        - centroids: (k, n_features) array of cluster centers
        - labels: (n_samples,) array of cluster labels
    """
    #k-means clustering
    model = KMeans(n_clusters=k, n_init='auto', random_state=42)
    model.fit(X)
    
    # Return centroids and labels
    return model.cluster_centers_, model.labels_

    # exercise 2

# Load the diamonds dataset 
diamonds = sns.load_dataset('diamonds')

# Select only numerical columns
diamonds_numeric = diamonds.select_dtypes(include=[np.number])

def kmeans_diamonds(n, k):
    """
    Run k-means clustering on the first n rows of the numeric diamonds data.

    Parameters
    ----------
    n : int
        Number of rows to use.
    k : int
        Number of clusters.

    Returns
    -------
    tuple
        (centroids, labels) from kmeans()
    """
    X = diamonds_numeric.iloc[:n].to_numpy()
    return kmeans(X, k)

def kmeans_timer(n, k, n_iter=5):
    """
    Run kmeans_diamonds multiple times and return average runtime.

    Parameters
    ----------
    n : int
        Number of rows from the dataset.
    k : int
        Number of clusters.
    n_iter : int, optional
        Number of iterations (default is 5).

    Returns
    -------
    float
        Average runtime in seconds.
    """
    times = []
    for _ in range(n_iter):
        start = time()
        _ = kmeans_diamonds(n, k)
        elapsed = time() - start
        times.append(elapsed)
    
    return np.mean(times)