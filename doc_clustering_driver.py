# Source: https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html
# Modification made by Yuan-Chi Chang <yuanchi@us.ibm.com> to create test cases for OpenShift AI Data Science Pipeline
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Arturo Amor <david-arturo.amor-quiroz@inria.fr>
# License: BSD 3 clause

import ray

from doc_clustering_actor import doc_clustering_actor

@ray.remote
def document_clustering(args):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import HashingVectorizer
    from load_documents import load_sample_data

    vectorizers = [TfidfVectorizer(max_df=0.5, min_df=5, stop_words="english"),
                   HashingVectorizer(stop_words="english", n_features=50_000)]

    ray_actors = [doc_clustering_actor.options(scheduling_strategy="SPREAD").remote(v) for v in vectorizers]
    print(f"{len(ray_actors)} Ray actors launched")

    dataset = load_sample_data()

    clustering_results = {}
    result_refs = [ray_actor.cluster_n_evaluate.remote(dataset) for ray_actor in ray_actors]
    while result_refs:
        ready_refs, result_refs = ray.wait(result_refs, num_returns = 1)
        evaluations = ray.get(ready_refs)
        clustering_results[evaluations[0][0][0]['vectorizer']] = evaluations[0][0][0]['V-measure']

    return clustering_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-ns", "--namespace", default=None)
    args = parser.parse_args()
    if args.namespace is None:
        ray.init(address='auto')
    else:
        ray.init(address="auto", namespace=args.namespace)

    print(ray.get(document_clustering.remote(args)))

