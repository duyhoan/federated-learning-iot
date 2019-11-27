FROM tensorflow/tensorflow:1.8.0-py3

RUN apt-get update

# Install additional packages
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app/ /usr/src/app
WORKDIR /usr/src/app
ENV PYTHONUNBUFFERED 1