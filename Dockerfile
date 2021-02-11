# We have a very esoteric Python dependency resolution strategy here
# in the interest of finding pre-compiled binaries for PowerPC.
# Simpler x64-only version is represented by commit
# 66fad39af31d2aa3b2a58f9e857ae090c475a2ab
#
# There are several tools which we can use to get packages from different places:
#
# - the Docker base image
# - apt-get (out of date, deprecated numpy functions break python3-skimage-lib)
# - pip (poor availability of wheels for ppc64le)
# - conda (more options)
#
# First, we get tensorflow from the base image because it cannot be obtained
# from anywhere else.
# Next, we use conda to install numpy, opencv, and scikit-image.
# These packages are broken on ubuntu-18.04 (and bloated in ubuntu-20.04).
# The remaining packages, nibabel and Keras, can be safely installed with
# pip since the difficult packages which they depend on were installed
# previously using conda.

FROM fnndsc/tensorflow:1.15.3 as base

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:$PATH
ENV PYTHONPATH=/opt/conda/lib/python3.6/site-packages:/usr/local/lib/python3.6/dist-packages

FROM base as dependencies
WORKDIR /tmp

# download conda installer
RUN apt-get update
RUN apt-get install -y curl
RUN curl -so install-conda.sh \
    https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-$(uname -s)-$(uname -m).sh
# install conda
RUN bash install-conda.sh -b -p /opt/conda

# downgrade python to the same version which was present in base image
RUN conda install python=$(/usr/local/bin/python --version | awk '{print $2}')

# can't use conda env export -n base -f environment.yml to create
# a lock file because the build IDs differ cross-platform
RUN conda install numpy=1.19.2 scikit-image=0.17.2 opencv=3.4.1

# remaining non-binary dependencies are safe to get from pypi because the
# difficult libraries which they depend on were installed by conda
COPY requirements.txt .
RUN pip install --upgrade-strategy only-if-needed -r requirements.txt


FROM base

COPY --from=dependencies /opt/conda /opt/conda

WORKDIR /usr/local/src
COPY . .
# data (model and weights) can be deleted from source folder
# because they are copied  to a system managed directory by pip
RUN pip install . && rm -rv fetal_brain_mask/data

CMD ["fetal_brain_mask", "--help"]
