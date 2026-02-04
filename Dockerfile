FROM python:3.12

RUN apt-get update -y && \
    apt-get install ffmpeg -y

RUN pip install --upgrade pip poetry &&\
    poetry config virtualenvs.create false

WORKDIR /app
ADD pyproject.toml poetry.lock /app/
ADD README.md /app/README.md
RUN --mount=type=cache,target=/root/.cache/pip poetry install --without dev --no-root

WORKDIR /app/src
ADD ./src /app/src

CMD ["/bin/bash"]
