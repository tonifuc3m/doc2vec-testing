#FROM ubuntu:18.04

# Anaconda
#RUN apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

#RUN  apt-get update \
#  && apt-get install -y wget \
#  && rm -rf /var/lib/apt/lists/*

#RUN wget https://repo.anaconda.com/archive/Anaconda2-2019.10-Linux-x86_64.sh

#RUN bash Anaconda2-2019.10-Linux-x86_64.sh -b -p anaconda

#RUN conda install scipy=0.10

FROM continuumio/anaconda2
RUN apt-get update \
	&& apt-get install -y gcc \
	&& rm -rf /var/lib/apt/lists/*
RUN conda create -n env python=2.7
RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH
ENV CONDA_DEFAULT_ENV env

#RUN conda init bash
#RUN conda activate env


RUN python -m pip install scipy==0.10
RUN python -m pip install numpy==1.8
RUN python -m pip install smart-open==1.10.0

RUN apt-get install git
RUN git clone --single-branch --branch develop https://github.com/jhlau/gensim.git
RUN python -m pip install gensim/

RUN python -c "import gensim.models as g"

RUN git clone https://github.com/jordiae/doc2vec_docker.git


#VOLUME /resources


# train
#ENTRYPOINT ["python", "doc2vec_docker/doc2vec_docker.py", "train", "model_prova", "/mount/output", "/mount/Scielo_wiki_FastText300.vec", "/mount/data/utf8_nofooter"]
# retrieve
ENTRYPOINT ["python", "doc2vec_docker/doc2vec_docker.py", "retrieve", "model_prova", "/mount/output", "/mount/Scielo_wiki_FastText300.vec", "/mount/data/utf8_nofooter"]
