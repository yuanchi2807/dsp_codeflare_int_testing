# Source: https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html
# Modification made by Yuan-Chi Chang <yuanchi@us.ibm.com> to create test cases for OpenShift AI Data Science Pipeline
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Arturo Amor <david-arturo.amor-quiroz@inria.fr>
# License: BSD 3 clause

import ray

@ray.remote
class doc_clustering_actor:
    """
    Ray actor class to perform clustering and evaluate results
    """
    def __init__(
        self,
        vectorizer,
    ):
        self.vectorizer = vectorizer
        self.name = vectorizer.__class__.__name__

    def fit_and_evaluate(self, km, labels, X, name=None, n_runs=5):
        from collections import defaultdict
        import numpy as np
        from time import time
        from sklearn import metrics

        evaluations = []
        evaluations_std = []

        name = km.__class__.__name__ if name is None else name

        train_times = []
        scores = defaultdict(list)
        for seed in range(n_runs):
            km.set_params(random_state=seed)
            t0 = time()
            km.fit(X)
            train_times.append(time() - t0)
            scores["Homogeneity"].append(metrics.homogeneity_score(labels, km.labels_))
            scores["Completeness"].append(metrics.completeness_score(labels, km.labels_))
            scores["V-measure"].append(metrics.v_measure_score(labels, km.labels_))
            scores["Adjusted Rand-Index"].append(
                metrics.adjusted_rand_score(labels, km.labels_)
            )
            scores["Silhouette Coefficient"].append(
                metrics.silhouette_score(X, km.labels_, sample_size=2000)
            )
        train_times = np.asarray(train_times)

        print(f"clustering done in {train_times.mean():.2f} ± {train_times.std():.2f} s ")
        evaluation = {
            "vectorizer": self.name,
            "train_time": train_times.mean(),
        }
        evaluation_std = {
            "vectorizer": self.name,
            "train_time": train_times.std(),
        }
        for score_name, score_values in scores.items():
            mean_score, std_score = np.mean(score_values), np.std(score_values)
            # print(f"{score_name}: {mean_score:.3f} ± {std_score:.3f}")
            evaluation[score_name] = mean_score
            evaluation_std[score_name] = std_score
        evaluations.append(evaluation)
        evaluations_std.append(evaluation_std)

        return (evaluations, evaluations_std)

    def cluster_n_evaluate(self, dataset):
        import numpy as np
        from sklearn.decomposition import TruncatedSVD
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Normalizer
        from sklearn.feature_extraction.text import TfidfTransformer
        from sklearn.cluster import KMeans

        labels = dataset.target
        unique_labels, category_sizes = np.unique(labels, return_counts=True)
        true_k = unique_labels.shape[0]

        transform_pipeline = make_pipeline(
            self.vectorizer,
            TfidfTransformer(),
            TruncatedSVD(n_components=100),
            Normalizer(copy=False))

        X_vec = transform_pipeline.fit_transform(dataset.data)

        kmeans = KMeans(
            n_clusters=true_k,
            max_iter=100,
            n_init=5,
        )

        return self.fit_and_evaluate(kmeans, labels, X_vec)




