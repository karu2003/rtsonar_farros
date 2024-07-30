# pyaudio_test
To test pyaudio in docker.

## Find the ALSA card name on your host system

Use `aplay -l` to list all devices available for playback. The audio card will appear some what like this:

```
card 2: Generic_1 [HD-Audio Generic], device 0: ALC285 Analog [ALC285 Analog]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
```

Before running the python file, verify once with: `aplay /usr/share/sounds/alsa/Front_Center.wav`.

Once at the shell, run `aplay Front_Center.wav` which should play the audio using alsa. 
Then run `python3 play.py Front_Center.wav` which will attempt to play the audio using python.

Play.py is from http://people.csail.mit.edu/hubert/pyaudio/docs/ and plays the audio on macOS.
