# iris-podcast-maker
Create a podcast version of the sermons from Iris Church

# Quick start

```shell
# build the image
./run build

# run the image
./run -- python do.py
```


## Make outro/intro

```shell
: 1657458828:0;ffmpeg -i Ru9Mb3Mh14k.mp3 -ss 5718 -to 5746 outro.mp3
: 1657458863:0;ffmpeg -i outro.mp3 -af afade=t=in:ss=0:d=3 outro.fade.mp3
: 1657458898:0;ffmpeg -i outro.fade.mp3 -af afade=t=in:ss=24:d=28 outro.fade.1.mp3
: 1657458944:0;ffmpeg -i outro.fade.mp3 -af afade=t=out:ss=24:d=28 outro.fade.1.mp3
```

```shell
: 1657458439:0;ffmpeg -i Ru9Mb3Mh14k.mp3 -ss 250 -to 264 intro.mp3
: 1657458584:0;ffmpeg -i intro.mp3 afade=t=in:ss=0:d=3 intro.fade.mp3
: 1657458671:0;ffmpeg -i Ru9Mb3Mh14k.mp3 -ss 245 -to 264 intro.mp3
: 1657458677:0;ffmpeg -i intro.mp3 -af afade=t=in:ss=0:d=3 intro.fade.mp3
```
