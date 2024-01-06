FROM python:3.8-slim-buster

ENV PYENV_SHELL=/bin/bash
ENV GOOGLE_APPLICATION_CREDENTIALS=secrets/mega-pipeline.json


RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y --no-install-recommends build-essential ffmpeg

RUN pip install --no-cache-dir --upgrade pip
RUN pip install pipenv


RUN mkdir -p /app
WORKDIR /app

ADD Pipfile Pipfile.lock /app/

RUN pipenv sync

ADD . /app

ENTRYPOINT ["/bin/bash"]

CMD ["-c","pipenv shell"]