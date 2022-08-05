import os
import re
from pathlib import Path
import logging
from datetime import timedelta

import ffmpeg
import yt_dlp
from yt_dlp.utils import MaxDownloadsReached


class MalformedDescription(ValueError):

    pass


def download_descriptions():
    ydl = yt_dlp.YoutubeDL({
        'outtmpl': '%(id)s',
        'max_downloads': 10,
        'skip_download': True,
        'download_archive': 'archive',
        'writedescription': True
    })

    with ydl:
        try:
            ydl.extract_info('https://www.youtube.com/user/bisericairis/videos')
        except MaxDownloadsReached:
            pass


def get_descriptions():
    for file_name in os.listdir('.'):
        if file_name.endswith('.description'):
            with open(file_name, 'r') as f:
                content = f.read()

            id_ = file_name.split('.')[0]

            yield id_, content


def clean_descriptions():
    for file_name in os.listdir('.'):
        if file_name.endswith('.description'):
            os.remove(file_name)


def is_with_time(description):
    return re.match(r'^(\d*\d:)*\d*\d:\d\d.*', description.strip())


def is_sermon_time(description_line):
    return ' Mesaj |' in description_line


def get_timedelta_from_desc_interval(desc_time):
    def compute_one(str_time):
        ls_parts = list(reversed(str_time.strip().split(':')))
        ls_parts_complete = ls_parts + ([0] * (3 - len(ls_parts)))
        return timedelta(
            seconds=int(ls_parts_complete[0]),
            minutes=int(ls_parts_complete[1]),
            hours=int(ls_parts_complete[2])
        )

    the_time = compute_one(desc_time.strip())

    return the_time


def extract_times(description):
    lines_with_time = []
    for line in description.split('\n'):
        if is_with_time(line):
            lines_with_time.append((line, is_sermon_time(line)))

    times = []
    for index, (line, is_sermon) in enumerate(lines_with_time):
        if is_sermon:
            str_time = line.split(' Mesaj ')[0]
            begin = get_timedelta_from_desc_interval(str_time)
            next_line, _ = lines_with_time[index + 1]
            end = get_timedelta_from_desc_interval(next_line.split(' ')[0])
            times.append((begin, end))

    return times


def mark_as_processed(video_id):
    with open('already_processed', 'a+') as f:
        f.write(f'{video_id.strip()}\n')


def was_processed(video_id):
    with open('already_processed', 'r') as f:
        video_ids = [line.strip() for line in f]
        return video_id in video_ids


def download_video(id_):
    ydl = yt_dlp.YoutubeDL({
        'outtmpl': '%(id)s.video',
        'format': 'vestvideo/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192'
        }],
        'prefer_ffmpeg': True,
        'keepvideo': True
    })

    with ydl:
        try:
            ydl.download(['https://www.youtube.com/watch?v={}'.format(id_)])
        except MaxDownloadsReached:
            pass

    return f'{id_}.mp3'


def make_podcast(file_name, times: list[tuple[timedelta, timedelta]]):
    file_path = Path(file_name)
    audio_input = ffmpeg.input(file_name)
    final_inputs = []
    for index, (begin, end) in enumerate(times):
        print(f'=== Trimming from {begin.total_seconds()} to {end.total_seconds()}')
        trimmed = audio_input.filter_('atrim', start=begin.total_seconds(), end=end.total_seconds())
        out_file_name = f'{file_path.stem}.{index}.cut.mp3'
        output = ffmpeg.output(trimmed, out_file_name)

        ffmpeg.run(output)
        final_inputs.append(ffmpeg.input(out_file_name))

    final_inputs.insert(0, ffmpeg.input('intro.mp3'))
    final_inputs.append(ffmpeg.input('outro.mp3'))
    merged = ffmpeg.concat(*final_inputs, v=0, a=1).node
    audio = merged[1]
    output = ffmpeg.output(audio, f'{file_path.stem}.final.mp3')
    ffmpeg.run(output)


def run():
    clean_descriptions()
    download_descriptions()
    descriptions = get_descriptions()

    for video_id, description in descriptions:
        if was_processed(video_id):
            continue

        try:
            video_times = extract_times(description)
        except MalformedDescription as e:
            logging.error(f'Failed for video [{video_id}]: {e}')
            continue

        if video_times:
            file_name = download_video(video_id)
            make_podcast(file_name, video_times)
            mark_as_processed(video_id)

    clean_descriptions()


if __name__ == '__main__':
    run()
