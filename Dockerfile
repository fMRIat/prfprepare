# This Dockerfile constructs a docker image, based on the vistalab/freesurfer
# docker image to execute recon-all as a Flywheel Gear.
#
# Example build:
#   docker build --no-cache --tag scitran/freesurfer-recon-all `pwd`
#
# Example usage:
#   docker run -v /path/to/your/subject:/input scitran/freesurfer-recon-all
#
FROM ubuntu:xenial

# Install dependencies for FreeSurfer
#RUN apt-get update && apt-get -y install \
#        bc \
#        tar \
#        zip \
#        wget \
#        gawk \
#        tcsh \
#        python \
#        libgomp1 \
#        python-pip \
#        perl-modules 
# Download Freesurfer dev from MGH and untar to /opt
#RUN wget -N -qO- ftp://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.1.1/freesurfer-linux-centos6_x86_64-7.1.1.tar.gz | tar -xz -C /opt && chown -R root:root /opt/freesurfer && chmod -R a+rx /opt/freesurfer



# Make directory for flywheel spec (v0)
ENV FLYWHEEL /flywheel/v0
RUN mkdir -p ${FLYWHEEL}
WORKDIR ${FLYWHEEL}


RUN apt-get update --fix-missing \
 && apt-get install -y wget bzip2 ca-certificates \
      libglib2.0-0 libxext6 libsm6 libxrender1 \
      git mercurial subversion curl grep sed dpkg gcc g++ libeigen3-dev zlib1g-dev libqt4-opengl-dev libgl1-mesa-dev libfftw3-dev libtiff5-dev
RUN apt-get install -y libxt6 libxcomposite1 libfontconfig1 libasound2



############################
# Install dependencies
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    xvfb \
    xfonts-100dpi \
    xfonts-75dpi \
    xfonts-cyrillic \
    zip \
    unzip \
    python \
    imagemagick \
    wget \
    subversion\
    vim \
    bsdtar 

############################
# install conda env for neuropythy

FROM continuumio/miniconda3:latest

RUN conda update -n base -c defaults conda
 
# install conda env
COPY conda_config/scientific.yml .
RUN conda env create -f scientific.yml
 

RUN apt-get update && apt-get install -y jq



# Make directory for flywheel spec (v0)
ENV FLYWHEEL /flywheel/v0
RUN mkdir -p ${FLYWHEEL}

# Copy and configure run script and metadata code
COPY bin/run \
	scripts/stim_as_nii.py    \ 
	scripts/nii_to_surfNii.py \
	scripts/link_stimuli.py    \
      ${FLYWHEEL}/

# Handle file properties for execution
RUN chmod +x \
      ${FLYWHEEL}/run \
	${FLYWHEEL}/stim_as_nii.py    \ 
	${FLYWHEEL}/nii_to_surfNii.py \
	${FLYWHEEL}/link_stimuli.py    
WORKDIR ${FLYWHEEL}
# Run the run.sh script on entry.
ENTRYPOINT ["/flywheel/v0/run"]

#make it work under singularity 
# RUN ldconfig: it fails in BCBL, check Stanford 
#https://wiki.ubuntu.com/DashAsBinSh 
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
