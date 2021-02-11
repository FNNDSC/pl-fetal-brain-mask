FROM fnndsc/tensorflow:1.15.3

WORKDIR /usr/local/src

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# data (model and weights) can be deleted from source folder
# because they are copied  to a system managed directory by pip
RUN pip install . && rm -rv fetal_brain_mask/data

CMD ["fetal_brain_mask", "--help"]
