#!/usr/bin/env python

""" Script for adding GPS locations to images from images with GPS tag """

__author__ = "Johannes Hiller"
__version__ = "0.1"
__maintainer__ = "Johannes Hiller"
__status__ = "development"

import piexif
from pathlib import Path
from PIL import Image
from datetime import datetime
import bisect
import math
import numpy as np
import argparse

# Dictionary for converting GPS values
direction = {'N': 1, 'S': -1, 'E': 1, 'W': -1}


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    computes Haversine distance (distance on sphere)

    :return: distance of origin and destination
    """
    radius = 6371000  # meters

    if lon1 is None or lat1 is None or lon2 is None or lat2 is None:
        d = math.inf
    else:
        d_lat = math.radians(lat2 - lat1)
        d_lon = math.radians(lon2 - lon1)
        a = (math.sin(d_lat / 2) * math.sin(d_lat / 2) + math.cos(math.radians(lat1))
             * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) * math.sin(d_lon / 2))
        c = math.atan2(math.sqrt(a), math.sqrt(1 - a))
        d = 2 * radius * c
    return d


def find_closest(timestamps, t):
    """
    Function that finds the nearest timestamp to t in timestamps
    :param timestamps: List of timestamps
    :param t: Timestamp to find nearest to
    :return: Index in the list of timestamps
    """
    # https://stackoverflow.com/questions/29525050/nearest-timestamp-price-ready-data-structure-in-python
    idx = bisect.bisect_left(timestamps, t)  # Find insertion point
    # Check which timestamp with idx or idx - 1 is closer
    if idx > 0 and abs(timestamps[idx] - t) > abs(timestamps[idx - 1] - t):
         idx -= 1
    return idx  # Return index


def convert_gps_minutes_to_degrees(lat_dms, lat_ref, lon_dms, lon_ref):
    """
    Convert the display of GPS coordinates in minutes to degrees
    :param lat_dms: Latitude in degrees, minutes and seconds
    :param lat_ref: Latitude reference, i.e. "N" or "S"
    :param lon_dms: Longitude in degrees, minutes and seconds
    :param lon_ref: Longitude reference, i.e. "W" or "E"
    :return:
    """
    lat = (lat_dms[0][0] / lat_dms[0][1] + lat_dms[1][0] / lat_dms[1][1] / 60 +
           lat_dms[2][0] / lat_dms[2][1] / 3600) * direction[lat_ref]
    lon = (lon_dms[0][0] / lon_dms[0][1] + lon_dms[1][0] / lon_dms[1][1] / 60 +
           lon_dms[2][0] / lon_dms[2][1] / 3600) * direction[lon_ref]
    return lat, lon


def convert_gps_degress_to_minutes(lat, lon):
    """
    Convert GPS in degrees to degrees, minutes, seconds as needed for the Exif
    :param lat: Latitude in degrees
    :param lon: Longitude in degrees
    :return: Latitude and longitude in degrees, minutes, seconds
    """
    lat_positive = lat >= 0
    lon_positive = lon >= 0
    lat = abs(lat)
    lat_min, lat_sec = divmod(lat * 3600, 60)
    lat_deg, lat_min = divmod(lat_min, 60)
    lat_dms = ((int(lat_deg), 1), (int(lat_min), 1), (int(lat_sec * 10000), 10000))
    lat_ref = 'N'.encode() if lat_positive else 'S'.encode()
    lon = abs(lon)
    lon_min, lon_sec = divmod(lon * 3600, 60)
    lon_deg, lon_min = divmod(lon_min, 60)
    lon_dms = ((int(lon_deg), 1), (int(lon_min), 1), (int(lon_sec * 10000), 10000))
    lon_ref = 'E'.encode() if lon_positive else 'W'.encode()
    return lat_dms, lat_ref, lon_dms, lon_ref


def fill_up_locations(gps_list, dist_tol=500, time_tol=86400, fill_time=60, verbose=False):
    """
    Interpolate between two gps list entries, as long as they're close enough. This takes time and location into
    account. If the distance in meters is still close, after the time tolerance in seconds has run out, an
    interpolation is no longer done.
    :param gps_list: List of GPS locations
    :param dist_tol: Distance tolerance up to which two locations are considered for interpolation
    :param time_tol: Time tolerance, after which distance tolerance is seen as exhausted
    :param fill_time: Timesteps in which locations are planted
    :param verbose: Verbosity flag
    :return: Return a filled up GPS list
    """
    filled_gps_list = []
    diffs = [(0, 0), *[(l[0] - k[0], haversine_distance(k[1][0], k[1][1], l[1][0], l[1][1]))
                       for k, l in zip(gps_list[:-1], gps_list[1:])]]
    for idx, val in enumerate(diffs):
        if idx > 0 and fill_time < val[0] < time_tol and val[1] < dist_tol:
            time = np.arange(gps_list[idx-1][0], gps_list[idx][0], fill_time)
            lat = np.interp(time, [gps_list[idx - 1][0], gps_list[idx][0]],
                            [gps_list[idx - 1][1][0], gps_list[idx][1][0]])
            lon = np.interp(time, [gps_list[idx - 1][0], gps_list[idx][0]],
                            [gps_list[idx - 1][1][1], gps_list[idx][1][1]])
            filled_gps_list.extend([(j, (k, l)) for j, k, l in zip(time, lat, lon)])
        else:
            filled_gps_list.append(gps_list[idx])
    return filled_gps_list


def extract_timestamp_gps(img_folder, ext='jpg', mod_exc=None, verbose=False) -> (list, list):
    """
    Extracts all GPS locations from the images in the folder. A certain camera model can be excluded via the mod_exc
    :param img_folder: The folder to look for GPS locations
    :param ext: The extension for the images
    :param mod_exc: Model or type of camera to exclude
    :param verbose: Verbosity flag
    :return: Return a list of images without GPS location and a list of GPS locations
    """
    # A list of all images in the folder without GPS information. This will just contain path and timestamp
    imgs = []
    # A list of all images containing GPS information. This will contain path, timestamp, latitude and longitude
    imgs_gps = []
    if ext[0] != '.':
        ext = '.' + ext
    # Extract information on all pictures from the folder
    for img in img_folder.glob('*' + ext):
        if img.suffix == ext.lower() or img.suffix == ext.upper():
            with Image.open(img) as i:
                # Load the exif info
                exif_dict = piexif.load(i.info['exif'])
                # Get the time the picture was taken
                time_taken = exif_dict['Exif'][piexif.ExifIFD.DateTimeOriginal]
                time_taken = str(time_taken).replace('b', '').replace('\'', '')
                time_taken = datetime.strptime(time_taken, '%Y:%m:%d %H:%M:%S')
                # Excluded model can be zero as based on field lengths also works
                if mod_exc and len(exif_dict['0th']) > 6:
                    try:
                        if mod_exc and mod_exc in str(exif_dict['0th'][piexif.ImageIFD.Model]):
                            imgs.append([time_taken.timestamp(), img])
                            continue
                    except KeyError:
                        # TODO: Not quick and dirty
                        # Sometimes the exif extract only works in the second try. Catch this case here
                        try:
                            if mod_exc and mod_exc in str(exif_dict['0th'][piexif.ImageIFD.Model]):
                                imgs.append([time_taken.timestamp(), img])
                                print('Had to use the "except"')
                        except KeyError:
                            print('Failed while extracting camera model')
                        continue
                # If there is GPS information, extract it
                if len(exif_dict['GPS']) > 2:
                    latitude = exif_dict['GPS'][piexif.GPSIFD.GPSLatitude]
                    latiude_ref = exif_dict['GPS'][piexif.GPSIFD.GPSLatitudeRef]
                    longitude = exif_dict['GPS'][piexif.GPSIFD.GPSLongitude]
                    longitude_ref = exif_dict['GPS'][piexif.GPSIFD.GPSLongitudeRef]
                    imgs_gps.append([time_taken.timestamp(), img, latitude, latiude_ref,
                                     longitude, longitude_ref])

                else:
                    imgs.append([time_taken.timestamp(), img])
    # Sort pictures by timestamp (better safe than sorry)
    imgs.sort(key=lambda l: l[0])
    # Sort GPS pictures by timestamp
    imgs_gps.sort(key=lambda l: l[0])
    # Create list of GPS timestamps
    gps_stamps = [(l[0], convert_gps_minutes_to_degrees(l[2], l[3].decode(), l[4], l[5].decode())) for l in imgs_gps]
    return imgs, gps_stamps


def add_gps_to_pictures(imgs, gps_stamps, result_path=None, deviation=0, time_diff=300, verbose=False):
    """
    Adds a GPS location to the images that don't have a location
    :param imgs: List of images to apply locations to
    :param gps_stamps: List of GPS locations together with timestamps
    :param result_path: The path to store the images to. If None, the images are overwritten
    :param verbose: Verbosity flag
    :param deviation: Deviation between clocks of cameras
    :param time_diff: Maximum time difference for a match
    :return:
    """
    if result_path:
        if isinstance(result_path, str):
            result_path = Path(result_path)
        result_path.mkdir(parents=True, exist_ok=True)
    times = [stamp[0] for stamp in gps_stamps]
    cnt = 0
    # Iterate through all images without GPS
    for si in imgs:
        # Find closest GPS picture
        f_idx = find_closest(times, si[0])
        # Check if the time difference between the timestamps is not too big
        if abs(si[0] + deviation - times[f_idx]) < time_diff:
            cnt += 1
            # Open the image
            im = Image.open(si[1])
            exif_dict = piexif.load(im.info['exif'])
            # Convert the GPS values
            loc = convert_gps_degress_to_minutes(*gps_stamps[f_idx][1])
            # Generate the exif data using the GPS from the close image
            exif_dict['GPS'][piexif.GPSIFD.GPSLatitude] = loc[0]
            exif_dict['GPS'][piexif.GPSIFD.GPSLatitudeRef] = loc[1]
            exif_dict['GPS'][piexif.GPSIFD.GPSLongitude] = loc[2]
            exif_dict['GPS'][piexif.GPSIFD.GPSLongitudeRef] = loc[3]
            # Convert it to binary data
            exif_bytes = piexif.dump(exif_dict)
            if result_path:
                sl = result_path.joinpath(si[1].name)
            else:
                sl = si[1]
            # Write it to the image
            im.save(sl, exif=exif_bytes)
            if verbose:
                print('Added gps to ' + str(sl) + '. Time difference was ' +
                      str(si[0] + deviation - times[f_idx]) + ' s')
    if verbose:
        print('Added GPS locations to ' + str(cnt) + ' images.')


def main():
    parser = argparse.ArgumentParser(prog='Exif Image Location Matcher', description='Small script that matches '
                                                                                     'locations of images from cameras '
                                                                                     'with and without GPS tags')
    parser.add_argument('folder', help='The path to the folder containing the images you want matched.', type=Path)
    parser.add_argument('--output', '-o', help='Folder to output the images with the new Exif path', default=None,
                        type=Path)
    parser.add_argument('--model_exclude', '-m', help='The model of camera to exclude when looking for images with GPS',
                        default=None)
    parser.add_argument('--interpolate', '-i', help='Whether to interpolate the GPS list', action='store_true')
    parser.add_argument('--deviation', '-d', help='Deviation in seconds between the clocks of the cameras. Measured as '
                                                  'time of device without GPS - time of device with GPS', default=0,
                        type=int)
    parser.add_argument('--time_diff', '-t', help='Time difference in seconds between two pictures in order to still '
                                                  'count them as match. Default is 300 s', default=300, type=int)
    parser.add_argument('--ext', help='The extension of the pictures to use. Default is "jpg".', default='jpg')
    parser.add_argument('--verbose', '-v', help='Verbosity flag', action='store_true')

    args = parser.parse_args()
    if not args.folder:
        print('Please specify a folder!')
        return
    normal_imgs , gps_stamp_list = extract_timestamp_gps(args.folder, ext=args.ext, mod_exc=args.model_exclude,
                                                         verbose=args.verbose)
    if args.interpolate:
        gps_stamp_list = fill_up_locations(gps_stamp_list, verbose=args.verbose)
    add_gps_to_pictures(normal_imgs, gps_stamp_list, result_path=args.output, deviation=args.deviation,
                        time_diff=args.time_diff, verbose=args.verbose)


if __name__ == '__main__':
    main()
