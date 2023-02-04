import os
import re
from pathlib import Path
import logging
from datetime import timedelta
from textwrap import wrap
from functools import wraps

import ffmpeg
import yt_dlp
from yt_dlp.utils import MaxDownloadsReached


ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'data_workdir'
MEDIA_DIR = ROOT / 'media'
PODCAST_DIR = ROOT / 'podcasts'
# We add 3 more seconds after the sermon so we have time to do a fadeout
# to smoothly transition to the outro
SERMON_FADE_OVERTIME = 3


class MalformedDescription(ValueError):

    pass


def with_cwd(path):
    def deco(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            initial_dir = os.getcwd()
            os.chdir(path)
            try:
                result = func(*args, **kwargs)
            finally:
                os.chdir(initial_dir)

            return result

        return wrapper
    return deco


@with_cwd(DATA_DIR)
def download_descriptions():
    def f(info, **kwargs):
        '''!is_live & live_status!=is_upcoming & availability=public'''
        if info.get('live_status') == 'is_upcoming':
            return 'Is upcoming'

    ydl = yt_dlp.YoutubeDL(
        {
            "outtmpl": "%(id)s",
            "max_downloads": 100,
            "skip_download": True,
            "download_archive": "archive",
            "writedescription": True,
            'noplaylist': True,
            'match_filter': f,
        }
    )

    with ydl:
        try:
            ydl.extract_info("https://www.youtube.com/@bisericairis/streams")
        except MaxDownloadsReached:
            pass


def get_descriptions(data_dir=DATA_DIR):
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".description"):
            with open(DATA_DIR / file_name, "r") as f:
                content = f.read()

            id_ = file_name.split(".")[0]

            yield id_, content


def clean_descriptions(data_dir=DATA_DIR):
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".description"):
            os.remove(DATA_DIR / file_name)


def is_with_time(description):
    return re.match(r"^(\d*\d:)*\d*\d:\d\d - (\d*\d:)*\d*\d:\d\d.*", description.strip())


def is_sermon_time(description_line):
    return " Mesaj |" in description_line


def get_timedelta_from_desc_interval(desc_time):
    def compute_one(str_time):
        ls_parts = list(reversed(str_time.strip().split(":")))
        ls_parts_complete = ls_parts + ([0] * (3 - len(ls_parts)))
        return timedelta(
            seconds=int(ls_parts_complete[0]),
            minutes=int(ls_parts_complete[1]),
            hours=int(ls_parts_complete[2]),
        )

    the_time = compute_one(desc_time.strip())

    return the_time


def extract_times(description):
    lines_with_time = []
    for line in description.split("\n"):
        if is_with_time(line):
            lines_with_time.append((line, is_sermon_time(line)))

    times = []
    for index, (line, is_sermon) in enumerate(lines_with_time):
        if is_sermon:
            print('====== Found to be sermon')
            # this will be: "35:30 - 1:29:14"
            begin_and_end = line.split(" Mesaj ")[0]
            begin, end = begin_and_end.split('-')
            begin = get_timedelta_from_desc_interval(begin.strip())
            end = get_timedelta_from_desc_interval(end.strip())
            times.append((begin, end))

    return times


def mark_as_processed(video_id):
    with open("already_processed", "a+") as f:
        f.write(f"{video_id.strip()}\n")


def was_processed(video_id):
    with open("already_processed", "r") as f:
        video_ids = [line.strip() for line in f]
        return video_id in video_ids


@with_cwd(DATA_DIR)
def download_video(id_):
    ydl = yt_dlp.YoutubeDL(
        {
            "outtmpl": "%(id)s.video",
            "format": "vestvideo/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "prefer_ffmpeg": True,
            "keepvideo": True,
        }
    )

    with ydl:
        try:
            ydl.download(["https://www.youtube.com/watch?v={}".format(id_)])
        except MaxDownloadsReached:
            pass

    return f"{id_}.mp3"


def make_podcast(
    input_file: str,
    times: list[tuple[timedelta, timedelta]],
    data_dir: Path = DATA_DIR,
    media_dir: Path = MEDIA_DIR,
    podcasts_dir: Path = PODCAST_DIR
):
    file_path = Path(input_file)
    audio_input = ffmpeg.input(input_file)
    final_inputs = []
    for index, (begin, end) in enumerate(times):
        out_file_name = str(data_dir / f"{file_path.stem}.{index}.cut.mp3")
        print(f"=== Trimming from {begin.total_seconds()} to {end.total_seconds()}")
        trimmed = audio_input.filter_(
            "atrim", start=begin.total_seconds(), end=end.total_seconds() + SERMON_FADE_OVERTIME
        )
        trimmed_faded = trimmed.filter_(
            'afade', type='out', start_time=end.total_seconds(), duration=SERMON_FADE_OVERTIME
        )
        output = ffmpeg.output(trimmed_faded, out_file_name)

        ffmpeg.run(output)
        final_inputs.append(ffmpeg.input(out_file_name))

    final_inputs.insert(0, ffmpeg.input(str(media_dir / "intro.mp3")))
    final_inputs.append(ffmpeg.input(str(media_dir / "outro.mp3")))
    merged = ffmpeg.concat(*final_inputs, v=0, a=1).node
    audio = merged[1]
    output = ffmpeg.output(audio, str(podcasts_dir / f"{file_path.stem}.final.mp3"))
    ffmpeg.run(output)


def run():
    clean_descriptions()
    download_descriptions()
    descriptions = get_descriptions()

    for video_id, description in descriptions:
        if was_processed(video_id):
            continue

        # print("Description:", description)
        if video_id == 'xIBDSr7Jz8c':
            import pudb; pu.db
        try:
            video_times = extract_times(description)
        except MalformedDescription as e:
            logging.error(f"Failed for video [{video_id}]: {e}")
            continue

        if video_times:
            file_name = download_video(video_id)
            file_path = str(DATA_DIR / file_name)
            make_podcast(file_path, video_times)
            mark_as_processed(video_id)

    # clean_descriptions()


if __name__ == "__main__":
    run()
