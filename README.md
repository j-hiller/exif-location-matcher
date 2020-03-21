# exif-location-matcher

A script that matches the GPS tags of one set of images to another set of images that doesn't have GPS tags.
This can be used if e.g. your phone has GPS tags, but your "real" camera doesn't.
Tested and developed with Python 3.7

## Install

Clone the repository.

Install the requirements

```bash
pip install -r requirements.txt
```

## Usage

The tool can be used from the command line.
So in the folder where you downloaded it to, simply run

```bash
python location_matcher.py E:\some\folder\with\images
```

The script brings some options for the command line:

```bash
python location_matcher.py E:\some\folder\with\images [-h] [--output OUTPUT]
                                   [--model_exclude MODEL_EXCLUDE]
                                   [--interpolate] [--deviation DEVIATION]
                                   [--time_diff TIME_DIFF] [--ext EXT]
                                   [--verbose]
```

|short|long|explanation|default value|
|:---:|:---:|:---|:---:|
| -h | | Display the help | |
| -o | --output | The folder to store the output images, i.e. the new images with the GPS tags. If left empty, the images are overwritten. | None |
| -m | --model_exclude | Camera model to exclude when looking for images with GPS tags | None |
| -i | --interpolate | Whether to interpolate between locations to fill up gaps where there was not much movement, but a lot of time went by. There might still be an issue here | False |
| -d | --deviation | Deviation between the two cameras in time. This can happen, if one the clocks is not correctly adjusted. Defined as time of device without GPS - time of device with GPS | 0 |
| -t | --time_diff | Time difference between a GPS tag and an image to still take the GPS tag for that image. | 300 |
| -e | --ext | The extension of the images to look for. | jpg |
| -v | --verbose | Verbosity flag | False |

## TODOs

- [ ] Find bug with interpolation  
- [ ] Add setuptools?  
- [ ] Investigate issue with piexif sometimes crashing  

## License

MIT License, see LICENSE for details
