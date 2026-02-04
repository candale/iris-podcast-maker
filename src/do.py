import os
import re
from contextlib import ContextDecorator
from pathlib import Path
from datetime import timedelta
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from loguru import logger
import json

import ffmpeg
import yt_dlp
from yt_dlp.utils import MaxDownloadsReached

import torch
from transformers import pipeline


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data_workdir"
MEDIA_DIR = ROOT / "media"
PODCAST_DIR = ROOT / "podcasts"
# We add 3 more seconds after the sermon so we have time to do a fadeout
# to smoothly transition to the outro
SERMON_FADE_OVERTIME = 3


class MalformedDescription(ValueError):
    pass


class with_cwd(ContextDecorator):
    def __init__(self, path, create_if_not_exists=False):
        self.path = path
        self.initial_dir = os.getcwd()
        if os.path.exists(path) is False:
            os.mkdir(path)

    def __enter__(self):
        os.chdir(self.path)

    def __exit__(self, *exc):
        os.chdir(self.initial_dir)


@with_cwd(DATA_DIR, create_if_not_exists=True)
def download_descriptions(target_video_id=None):
    def filter_(info, **kwargs):
        """!is_live & live_status!=is_upcoming & availability=public"""
        if info.get("live_status") == "is_upcoming":
            return "Is upcoming"

    ydl = yt_dlp.YoutubeDL(
        {
            "outtmpl": "%(id)s",
            "max_downloads": 100,
            "skip_download": True,
            "download_archive": "archive",
            "writedescription": True,
            "noplaylist": True,
            "match_filter": filter_,
            "quiet": False,
            "extract_flat": True,
            "force_generic_extractor": False,
        }
    )

    description = []
    with ydl:
        channel_info = ydl.extract_info(
            "https://www.youtube.com/@bisericairis/streams", download=False
        )
        if "entries" in channel_info:
            videos = channel_info["entries"]
        else:
            # Some channel URLs might need an extra level of extraction
            playlist = channel_info["entries"][0]
            videos = playlist["entries"]

        description = []
        for index, video in enumerate(videos):
            # Extract full video info to get description
            video_id = parse_qs(urlparse(video["url"]).query)["v"][0]

            if target_video_id and target_video_id != video_id:
                continue

            logger.info(f"Getting info for video {video_id}, {index}/{len(videos)}")
            if os.path.exists(os.path.join(DATA_DIR, f"{video_id}.spec.json")):
                logger.info("Already processed, skipping")
                continue

            try:
                video_info = ydl.extract_info(video["url"], download=False)
            except yt_dlp.utils.DownloadError:
                logger.exception(f"Failed to download video {video['url']}")
                continue

            description = {
                "video_id": video_id,
                "title": video_info["title"],
                "description": video_info["description"],
                "upload_date": video_info["upload_date"],
                "url": video["url"],
            }

            with open(os.path.join(DATA_DIR, f"{video_id}.spec.json"), "w") as f:
                f.write(json.dumps(description, indent=4, sort_keys=True))


def get_descriptions(data_dir=DATA_DIR):
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".spec.json"):
            with open(DATA_DIR / file_name, "r") as f:
                content = f.read()

            id_ = file_name.split(".")[0]

            yield id_, content


def clean_descriptions(data_dir=DATA_DIR):
    with with_cwd(data_dir, create_if_not_exists=True):
        for file_name in os.listdir(data_dir):
            if file_name.endswith(".description"):
                os.remove(DATA_DIR / file_name)


def is_with_time(description):
    return re.match(
        r"^(\d*\d:)*\d*\d:\d\d - (\d*\d:)*\d*\d:\d\d.*", description.strip()
    )


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
    """
    Extracts various timiing information from the video, like when the sermon
    starts, when it ends etc
    """
    lines_with_time = []
    for line in description.split("\n"):
        if is_with_time(line):
            lines_with_time.append((line, is_sermon_time(line)))

    times = []
    for index, (line, is_sermon) in enumerate(lines_with_time):
        if is_sermon:
            print("====== Found to be sermon")
            # this will be: "35:30 - 1:29:14"
            begin_and_end = line.split(" Mesaj ")[0]
            begin, end = begin_and_end.split("-")
            begin = get_timedelta_from_desc_interval(begin.strip())
            end = get_timedelta_from_desc_interval(end.strip())
            times.append((begin, end))

    return times


def extract_times_from_local(video_id):
    if not os.path.exists("local_video_specs.json"):
        return []

    local_video_specs = json.loads(Path("local_video_specs.json").read_text())
    times = []
    if video_id in local_video_specs:
        times = [
            (timedelta(seconds=section["start"]), timedelta(seconds=section["end"]))
            for section in local_video_specs[video_id]
        ]

    return times


def mark_as_processed(video_id):
    with open("already_processed", "a+") as f:
        f.write(f"{video_id.strip()}\n")


def was_processed(video_id):
    if os.path.exists("already_processed") is False:
        return False

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

    return f"{id_}.video.mp3"


def make_podcast(
    input_file: str,
    times: list[tuple[timedelta, timedelta]],
    data_dir: Path = DATA_DIR,
    media_dir: Path = MEDIA_DIR,
    podcasts_dir: Path = PODCAST_DIR,
):
    file_path = Path(input_file)
    audio_input = ffmpeg.input(input_file)
    final_inputs = []
    for index, (begin, end) in enumerate(times):
        out_file_name = str(data_dir / f"{file_path.stem}.{index}.cut.mp3")
        print(f"=== Trimming from {begin.total_seconds()} to {end.total_seconds()}")
        trimmed = audio_input.filter_(
            "atrim",
            start=begin.total_seconds(),
            end=end.total_seconds() + SERMON_FADE_OVERTIME,
        )
        trimmed_faded = trimmed.filter_(
            "afade",
            type="out",
            start_time=end.total_seconds(),
            duration=SERMON_FADE_OVERTIME,
        )
        output = ffmpeg.output(trimmed_faded, out_file_name)

        ffmpeg.run(output)
        final_inputs.append(ffmpeg.input(out_file_name))

    final_inputs.insert(0, ffmpeg.input(str(media_dir / "intro.v2.mp3")))
    final_inputs.append(ffmpeg.input(str(media_dir / "outro.v2.mp3")))
    merged = ffmpeg.concat(*final_inputs, v=0, a=1).node
    audio = merged[1]
    file_name = str(podcasts_dir / f"{file_path.stem}.final.mp3")
    output = ffmpeg.output(audio, file_name)
    ffmpeg.run(output)

    return file_name


def transcribe(file_name):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    pipe = pipeline(
        "automatic-speech-recognition",
        # model="gigant/whisper-medium-romanian",
        # model="intelsense/whisper-m-romanian-ct2",
        # model="readerbench/whisper-ro",
        model="openai/whisper-medium",
        chunk_length_s=30,
        device=device,
    )

    prediction = pipe(file_name, batch_size=8, generate_kwargs={"language": "ro"})[
        "text"
    ]

    return prediction


def run(target_video_id="Wgk5otvahNk"):
    # clean_descriptions()
    download_descriptions(target_video_id=target_video_id)
    descriptions = get_descriptions()

    for video_id, description in descriptions:
        should_skip = (
            target_video_id and video_id != target_video_id or was_processed(video_id)
        )
        if should_skip:
            continue

        print("Description:", description)
        video_times = extract_times(description)
        if not video_times:
            video_times = extract_times_from_local(video_id)

        if video_times:
            file_name = download_video(video_id)
            file_path = str(DATA_DIR / file_name)
            file_name = make_podcast(file_path, video_times)
            transcribe(file_name)
            mark_as_processed(video_id)
        else:
            logger.warning(f"No descriptions for video: {video_id}")

    # clean_descriptions()


if __name__ == "__main__":
    run()
