## Integration testing for Red Hat Data Science Pipeline, CodeFlare and KubeRay
### Container image built from Dockerfile
quay.io/yuanchichang_ibm/integration_testing/dsp_codeflare_int_testing:0.1
### Data Science Pipeline workflow file
yet_another_ray_integration_test.py
### Ray application design
1. Design followed scikit learn example https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html
2. `load_sample_data.py` reads data from the 20newsgroups dataset.
3. `docker_clustering_driver.py` launches two Ray actors and configures a different vectorizer for each.
4. `docker_clustering_driver.py` submits newsgroup data to Ray actors, which executes sklearn pipeline to fit KMeans clustering
with vectorized data.
5. `doc_clustering_actor`.py returns the V-measure metric of resulting clusters.
### Pipeline definition
`yet_another_ray_integration_test.py` is modified from
https://github.com/diegolovison/ods-ci/blob/ray_integration/ods_ci/tests/Resources/Files/pipeline-samples/ray_integration.py
to point to the custom image and invokes docker_clustering_driver.py through Ray jobs API.