FROM python:3.9

RUN pip install youtube-dl yt-dlp ffmpeg-python && \
    apt-get update -y && \
    apt-get install ffmpeg -y
WORKDIR /app/src
ADD ./src /app/src

CMD ["/bin/bash"]
