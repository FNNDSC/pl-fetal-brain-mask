# See https://github.com/fnndsc/conda-multiarch
FROM fnndsc/tensorflow:1.15.3-conda4.9.2 as base

FROM base as dependencies
WORKDIR /tmp

RUN conda install numpy=1.19.2 scikit-image=0.17.2 opencv=3.4.1

COPY requirements.txt .
RUN pip --no-cache-dir install --upgrade-strategy only-if-needed -r requirements.txt

WORKDIR /usr/local/src
COPY . .
RUN pip install .

FROM base
COPY --from=dependencies /opt/conda /opt/conda

CMD ["fetal_brain_mask", "--help"]
