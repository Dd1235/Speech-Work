I ran DBSCAN on just 5 phoneme classes: `['s', 'ih', 'aa', 'iy', 'n']`, instead of the full set, to see whether a smaller and acoustically cleaner subset would give better clustering.

That gave a dataset of about `352k` frame-level examples.

I tried working with subsets between `10k` and `50k` frames, because that was enough to capture the density structure while still keeping the clustering runs manageable.

I standardized the MFCC features before clustering, because DBSCAN uses Euclidean distance [straight-line distance in feature space], and the `c0` coefficient [log energy] has a much larger numeric range than `c1-c11`, while `c1-c11` are more useful for distinguishing phonemes.

To choose the DBSCAN hyperparameters, I used k-distance quantiles [the distribution of distances to the k-th nearest neighbor, which helps estimate a reasonable neighborhood radius].

Then I did a grid-search style sweep over the main DBSCAN parameters: `epsilon` [the radius within which DBSCAN considers points to be neighbors] and `min_samples` [the minimum number of neighbors needed for a point to become a core point and start a cluster].

Even when DBSCAN produced `4-5` clusters, the cluster quality was still not good.

It was better than the earlier full-phone experiments, where around `90%` of frames were classified as noise [points that DBSCAN does not assign to any cluster]; in the 5-phone setup, the noise fraction came down to around `30%` in the better settings.

The main issue is the usual DBSCAN trade-off: if `epsilon` is too small [the neighborhood radius is too strict], the algorithm creates many tiny clusters and coverage drops because too many points become noise.

If `epsilon` is too large [the neighborhood radius is too loose], then many points get merged into one large cluster.

I also tried HDBSCAN [a hierarchical version of DBSCAN that can handle variable-density clusters better than a single global `epsilon`].

HDBSCAN gave `2` clusters with better quality: one extremely clean `s` cluster, and one mixed vowel/nasal cluster.

I also tried GMM [Gaussian Mixture Model, a probabilistic clustering method that models each cluster as a Gaussian distribution].

For GMM, the BIC [Bayesian Information Criterion, used for selecting the number of components] kept improving up to `11` components.

When I forced GMM to use `5` clusters for comparison, `s` and `aa` were modeled very cleanly.

However, `ih` and `iy` still overlapped heavily even in GMM.

So the main takeaway is that restricting to 5 classes helped compared to the full set, but the feature space is still not clean enough for DBSCAN-style clustering to recover all phoneme classes well.
