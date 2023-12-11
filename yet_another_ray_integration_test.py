from kfp import components, dsl
from ods_ci.libs.DataSciencePipelinesKfpTekton import DataSciencePipelinesKfpTekton


def ray_fn(openshift_server:str, openshift_token:str) -> int:
    from codeflare_sdk.cluster.cluster import Cluster, ClusterConfiguration
    from codeflare_sdk.cluster.auth import TokenAuthentication
    import ray


    auth = TokenAuthentication(
        token=openshift_token,
        server=openshift_server,
        skip_tls=True
    )
    auth.login()
    cluster = Cluster(ClusterConfiguration(
        name='raytest',
        namespace='integration-test-ray',
        num_workers=2,
        head_cpus='1',
        head_memory='2Gi',
        num_gpus=0,
        # image="quay.io/project-codeflare/ray:latest-py39-cu118",
        # Point to the integration testing image
        image="quay.io/yuanchichang_ibm/integration_testing/dsp_codeflare_int_testing:0.1",
        instascale=False
    ))
    # workaround for https://github.com/project-codeflare/codeflare-sdk/pull/412
    cluster_file_name = '/opt/app-root/src/.codeflare/appwrapper/raytest.yaml'
    # Read in the file
    with open(cluster_file_name, 'r') as file:
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace('busybox:1.28', 'quay.io/project-codeflare/busybox:latest')

    # Write the file out again
    with open(cluster_file_name, 'w') as file:
        file.write(filedata)
    # end workaround

    # always clean the resources
    cluster.down()
    print(cluster.status())
    cluster.up()
    cluster.wait_ready()
    print(cluster.status())
    print(cluster.details())

    ray_dashboard_uri = cluster.cluster_dashboard_uri()
    ray_cluster_uri = cluster.cluster_uri()
    print(ray_dashboard_uri)
    print(ray_cluster_uri)

    # before proceeding make sure the cluster exists and the uri is not empty
    assert ray_cluster_uri, "Ray cluster needs to be started and set before proceeding"

    # reset the ray context in case there's already one.
    ray.shutdown()
    # establish connection to ray cluster
    ray.init(address=ray_cluster_uri)
    print("Ray cluster is up and running: ", ray.is_initialized())

    # ray document clustering test
    import requests, json, time, sys
    from ray.job_submission import JobStatus

    # Ref: https://docs.ray.io/en/master/cluster/running-applications/job-submission/api.html#/paths/~1api~1jobs/post
    retries = 10
    job_id = -1
    invoke_script_in_image = {
            "entrypoint": f"python doc_clustering_driver.py",
        }
    i = retries
    while i > 0:
        try:
            resp = requests.post(f"{ray_dashboard_uri}/api/jobs/", json=invoke_script_in_image)
            if resp.status_code == 200:
                rst = json.loads(resp.text)
                job_id = rst["job_id"]
                print(f"Submitted job to Ray with the id: {job_id}")
                break
            else:
                print(f"Failed submitted job to Ray, status : {resp.status_code}")
                sys.exit(1)
        except Exception as e:
            print(f"Failed to submit Ray remote job, error {e}")
            sys.exit(1)
        time.sleep(1)
        i -= 1
        if i == 0:
            print(f"Failed to submit Ray remote job in {retries} retries")
            sys.exit(1)

    # Get job status
    def get_job_status(job_id: str) -> str:
        for i in range(retries):
            try:
                resp = requests.get(f"{ray_dashboard_uri}/api/jobs/{job_id}")
                if resp.status_code == 200:
                    rst = json.loads(resp.text)
                    return rst["status"]
                else:
                    print(f"Getting job execution status failed, status {resp.status_code}")
            except Exception as e:
                print(f"Failed to get job status, error {e}")
            time.sleep(1)
        print(f"Failed to get job status in {retries} retries")
        sys.exit(1)

    job_status = get_job_status(job_id)
    while job_status in [JobStatus.STOPPED, JobStatus.SUCCEEDED, JobStatus.RUNNING,
                         JobStatus.PENDING, JobStatus.FAILED]:
        job_status = get_job_status(job_id)
        if job_status == JobStatus.PENDING or job_status == JobStatus.RUNNING:
            time.sleep(1)
            continue
        elif job_status == JobStatus.STOPPED:
            result = 'STOPPED'
            break
        elif job_status == JobStatus.FAILED:
            result = 'FAILED'
            break
        elif job_status == JobStatus.SUCCEEDED:
            result = 'SUCCEEDED'
            break
        else:
            print(f"Unrecognized Ray JobStatus {job_status}")
            sys.exit(1)

    assert 'SUCCEEDED' == result

    '''
    @ray.remote
    def train_fn():
        return 100

    result = ray.get(train_fn.remote())
    assert 100 == result
    '''
    ray.shutdown()
    cluster.down()
    auth.logout()
    return result


@dsl.pipeline(
    name="Ray Integration Test",
    description="Ray Integration Test",
)
def ray_integration(openshift_server, openshift_token):
    ray_op = components.create_component_from_func(
        ray_fn, base_image=DataSciencePipelinesKfpTekton.base_image,
        packages_to_install=['codeflare-sdk']
    )
    ray_op(openshift_server, openshift_token)