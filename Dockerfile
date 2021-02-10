FROM python:3.9.1-slim-buster

WORKDIR /usr/local/src

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install .

CMD ["fetal_brain_mask", "--help"]
