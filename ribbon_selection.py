import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from joblib import load
from crispy import CRISP, CRISPNonU
from crispy.spectral import interp_fine

def db_score(min_k, max_k, ss):
    """
    Calculates the Davies-bouldin score for a range of cluster numbers for a specific set of standardised data.

    Parameters
    ----------
    min_k : int
        The minimum number of clusters to try.
    max_k : int
        The maximum number of clusters to try.
    ss : numpy.ndarray
        The standardised data to be clusetered (should be of shape (n_samples, n_features)).

    Returns
    -------
    dbs : list
        A list of the Davies-bouldin scores for each of the number of clusters in the cluster range.
    """

    cluster_range = range(min_k, max_k+1)
    dbs = []

    for k in cluster_range:
        km = KMeans(n_clusters=k, n_jobs=-1)
        l = km.fit_predict(ss)
        dbs.append(davies_bouldin_score(ss, l))

    return dbs

def kmeans_predict(data, model, nonu=False):
    """
    A function to use the pretrained KMeans model to predict cluster labels for standardised spectra.

    Parameters
    ----------
    data : str or crispy.CRISP or crispy.CRISPNonU
        The path to the data or the data object itself which is to be clustered.
    model : str or sklearn.cluster.KMeans
        The path to the KMeans model or the model itself.
    nonu : bool, optional
        Whether or not the wavelengths of the spectra are sampled non-uniformly. Default is False, it is sampled uniformly.

    Returns
    -------
    labels : numpy.ndarray
        An array of the cluster labels for each point reshaped to be the same dimensions as the input image.
    """
    if type(data) == str:
        if not nonu:
            data = CRISP(data)
        else:
            data = CRISPNonU(data)
    
    if type(model) == str:
        model = load(model)

    wavels = data.wvls
    interp_wavels, interp_spec = interp_fine(wavels, data.data[...], pts=model.cluster_centers_.shape[-1])

    spectra = interp_spec.reshape((interp_spec.shape[0],-1)).T # this reshapes into a (n_samples, n_features) shape array where the number of samples is equal to the number of pixels and the number of features is the number of interpolated wavelength points

    standard_spectra = StandardScaler().fit_transform(spectra)

    labels = model.predict(standard_spectra).reshape((data.shape[-2], data.shape[-1]))

    return labels

def kmeans_ribbons(labels, ribb_clust=0):
    """
    A function that takes the labels output by the trained KMeans and returns an array that contains 1 in the locations of the flare ribbons and 0 elsewhere.

    Parameters
    ----------
    labels : numpy.ndarray
        The cluster labels from the result of the trained KMeans clustering the data.
    ribb_clust : int, optional
        Which cluster in the trained KMeans model represents the flare ribbon class. Default is 0.

    Returns
    -------
    ribbon_labels : numpy.ndarray
        An array containing 1s where the flare ribbons have been identified by the KMeans model and 0 elsewhere.
    """
    ribbon_labels = np.zeros_like(labels.flatten())

    ribbon_labels[np.where(labels.flatten() == ribb_clust)] = 1

    return ribbon_labels.reshape(labels.shape)

def dbscan_ribbons(ribbon_labels, min_samples=100, eps=0.1):
    """
    A function to perform the DBSCAN on the flare ribbon locations to eliminate noise and locate exactly where flare ribbons are (and separation of multiple flare ribbons).

    NOTE: DBSCAN is deterministic. There is no trained model because it does not do predictions, it performs the DBSCAN algorithm for each data it sees. This can lead to different optimal DBSCAN parameters for different datasets but it is likely that for a single flare the same DBSCAN parameters should work for all data. tl;dr flares are slightly sensible.

    Parameters
    ----------
    ribbon_labels : numpy.ndarray
        An array containing 1s where the flare ribbons have been located by the KMeans model and 0 elsewhere.
    min_samples : int
        The `min_samples` DBSCAN parameter. Default is 100.
    eps : float
        The `eps` DBSCAN parameter. Default is 0.1.

    Returns
    -------
    dbribs_img : numpy.ndarray
        An array containing each KMeans-identified flare spectra with its associated DBSCAN cluster label in its rightful spatial location. Points that were not clustered by DBSCAN are represented by NaNs.
    """
    dbscan = DBSCAN(min_samples=min_samples, eps=eps)

    db_ribbons = []
    flat_ribbs = ribbon_labels.flatten()
    for j in tqdm(range(flat_ribbs.shape[0])):
        if flat_ribbs[j] == 1:
            db_ribbons.append([j % ribbon_labels.shape[-1], j // ribbon_labels.shape[-1], flat_ribbs[j]])
    db_ribbons = np.array(db_ribbons)
    print(db_ribbons.shape)

    ss_dbribbons = StandardScaler().fit_transform(db_ribbons)
    db_labels = dbscan.fit_predict(ss_dbribbons)

    dbribs_img = np.zeros((ribbon_labels.shape[-2], ribbon_labels.shape[-1]))
    for j in tqdm(range(dbribs_img.shape[-2])):
        for i in range(dbribs_img.shape[-1]):
            try:
                idx = np.where((db_ribbons[:,0] == i) & (db_ribbons[:,1] == j))[0][0]
                dbribs_img[j,i] = db_labels[idx]
            except IndexError:
                dbribs_img[j,i] = np.nan

    return dbribs_img