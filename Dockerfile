ARG BASE_IMAGE=ubuntu:noble

#############################
# Download stages
#############################

# Utilities for downloading packages
FROM ${BASE_IMAGE} AS downloader

RUN apt update && \
    apt install -y --no-install-recommends \
                    binutils \
                    bzip2 \
                    ca-certificates \
                    curl \
                    unzip && \
    apt clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# FreeSurfer 7.3.2
FROM downloader AS freesurfer
COPY config_files/freesurfer7.3.2-exclude.txt /usr/local/etc/freesurfer7.3.2-exclude.txt
RUN curl -sSL https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.3.2/freesurfer-linux-ubuntu22_amd64-7.3.2.tar.gz \
     | tar zxv --no-same-owner -C /opt --exclude-from=/usr/local/etc/freesurfer7.3.2-exclude.txt

# Micromamba
FROM downloader AS micromamba
# Install a C compiler to build extensions when needed.
# traits<6.4 wheels are not available for Python 3.11+, but build easily.
RUN apt update && \
    apt install -y --no-install-recommends build-essential && \
    apt clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

ENV MAMBA_ROOT_PREFIX="/opt/conda"
COPY config_files/scientific.yml /tmp/scientific.yml
WORKDIR /tmp
RUN micromamba create -y -f /tmp/scientific.yml && \
    micromamba clean -y -a

# Put conda in path so we can use conda activate
ENV PATH="/opt/conda/envs/scientific/bin:$PATH" \
      UV_USE_IO_URING=0

#############################
# Main stage
#############################
FROM ${BASE_IMAGE} AS prfprepare

# Make directory for flywheel spec (v0)
ENV FLYWHEEL=/flywheel/v0
RUN mkdir -p ${FLYWHEEL}
WORKDIR ${FLYWHEEL}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update --fix-missing \
 && apt install -y --no-install-recommends \
      libglib2.0-0 \
      libxext6 \
      libsm6 \
      libxrender1 \
      git \
      mercurial \
      subversion \
      grep \
      sed \
      dpkg \
      gcc \
      g++ \
      libeigen3-dev \
      zlib1g-dev \
      libgl1-mesa-dev \
      libfftw3-dev \
      libtiff5-dev \
      libxt6 \
      libxcomposite1 \
      libfontconfig1 \
      libasound2t64 \
      bc \
      tcsh \
      libgomp1 \
      python3-pip \
      perl-modules \
      xvfb \
      xfonts-100dpi \
      xfonts-75dpi \
      xfonts-cyrillic \
      python-is-python3 \
      imagemagick \
      wget \
      subversion\
      vim && \
      apt clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# Install files from freesurfer stage
COPY --from=freesurfer /opt/freesurfer /opt/freesurfer

# Simulate SetUpFreeSurfer.sh
ENV OS="Linux" \
    FS_OVERRIDE=0 \
    FIX_VERTEX_AREA="" \
    FSF_OUTPUT_FORMAT="nii.gz" \
    FREESURFER_HOME="/opt/freesurfer"
ENV SUBJECTS_DIR="$FREESURFER_HOME/subjects" \
    FUNCTIONALS_DIR="$FREESURFER_HOME/sessions" \
    MNI_DIR="$FREESURFER_HOME/mni" \
    LOCAL_DIR="$FREESURFER_HOME/local" \
    MINC_BIN_DIR="$FREESURFER_HOME/mni/bin" \
    MINC_LIB_DIR="$FREESURFER_HOME/mni/lib" \
    MNI_DATAPATH="$FREESURFER_HOME/mni/data"
ENV PERL5LIB="$MINC_LIB_DIR/perl5/5.8.5" \
    MNI_PERL5LIB="$MINC_LIB_DIR/perl5/5.8.5" \
    PATH="$FREESURFER_HOME/bin:$FREESURFER_HOME/tktools:$MINC_BIN_DIR:$PATH"


# Install files from micromamba stage
COPY --from=micromamba /bin/micromamba /bin/micromamba
COPY --from=micromamba /opt/conda/envs/scientific /opt/conda/envs/scientific

ENV MAMBA_ROOT_PREFIX="/opt/conda"
RUN bash -c 'eval "$(micromamba shell hook --shell bash)"' && \
    echo "micromamba activate scientific" >> $HOME/.bashrc
ENV PATH="/opt/conda/envs/scientific/bin:$PATH" \
    CPATH="/opt/conda/envs/scientific/include:$CPATH" \
    LD_LIBRARY_PATH="/opt/conda/envs/scientific/lib:$LD_LIBRARY_PATH"

# Copy and configure run script and metadata code
COPY bin/run \
	bin/run.py \
	scripts/stim_as_nii.py    \
	scripts/nii_to_surfNii.py \
	scripts/link_stimuli.py    \
      ${FLYWHEEL}/

# Handle file properties for execution
RUN chmod +x \
      ${FLYWHEEL}/run \
      ${FLYWHEEL}/run.py \
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
