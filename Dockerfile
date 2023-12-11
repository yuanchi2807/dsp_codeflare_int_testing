FROM python:3.10.12-slim-bullseye
ENV TZ=US \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /DSP
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN apt-get install -y bash

ENV PYTHONPATH "$PYTHONPATH:/DSP"
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install --no-cache-dir ray[default]==2.6.3
RUN pip install --no-cache-dir numpy==1.24.1
RUN pip install --no-cache-dir scikit-learn

COPY doc_clustering_driver.py /DSP/doc_clustering_driver.py
COPY doc_clustering_actor.py /DSP/doc_clustering_actor.py
COPY load_documents.py /DSP/load_documents.py

WORKDIR /DSP
CMD ["/bin/bash"]