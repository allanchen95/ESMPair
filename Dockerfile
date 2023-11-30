FROM ubuntu:18.04

RUN apt-get update

RUN apt-get install -y wget && apt-get install -y git build-essential libssl-dev

# install conda
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda install pytorch==2.0.1 torchvision pytorch-cuda=11.7 fair-esm absl-py dm-tree -c pytorch -c nvidia -c conda-forge

# installing google cloud SDK
RUN wget -q https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-386.0.0-linux-x86_64.tar.gz
RUN tar -xf google-cloud-cli-386.0.0-linux-x86_64.tar.gz
RUN ./google-cloud-sdk/install.sh -q



ARG username=$GIT_USERNAME
ARG password=$GIT_PASSWORD

WORKDIR /app

RUN git clone --single-branch -b si2312 https://github.com/diffuse-bio/ESMPair.git

WORKDIR /app/ESMPair

RUN cp msa_transformer.py /root/miniconda3/lib/python3.11/site-packages/esm/model/

WORKDIR /app/ESMPair/msa_pair

RUN python -m pip install -e .

WORKDIR /app/ESMPair

ENTRYPOINT ["python", "colattn_pair.py", "./dataset", "0" ]