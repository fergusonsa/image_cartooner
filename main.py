import argparse
import datetime
import pathlib

import cv2
# import numpy as np


class Cartoonizer:
    """Cartoonizer effect
        A class that applies a cartoon effect to an image.
        The class uses a bilateral filter and adaptive thresholding to create
        a cartoon effect.

        Based on example from https://www.geeksforgeeks.org/cartooning-an-image-using-opencv-python/
    """
    source_image_path = None

    def __init__(self, source_image_path):
        self.source_image_path = pathlib.Path(source_image_path)

    def render(self, img_rgb):
        img_rgb = cv2.imread(img_rgb)
        img_rgb = cv2.resize(img_rgb, (1366, 768))
        numDownSamples = 2  # number of downscaling steps
        numBilateralFilters = 50  # number of bilateral filtering steps

        # -- STEP 1 --

        # downsample image using Gaussian pyramid
        img_color = img_rgb
        for _ in range(numDownSamples):
            img_color = cv2.pyrDown(img_color)

            # cv2.imshow("downcolor",img_color)
        # cv2.waitKey(0)
        # repeatedly apply small bilateral filter instead of applying
        # one large filter
        for _ in range(numBilateralFilters):
            img_color = cv2.bilateralFilter(img_color, 9, 9, 7)

            # cv2.imshow("bilateral filter",img_color)
        # cv2.waitKey(0)
        # upsample image to original size
        for _ in range(numDownSamples):
            img_color = cv2.pyrUp(img_color)
            # cv2.imshow("upscaling",img_color)
        # cv2.waitKey(0)

        # -- STEPS 2 and 3 --
        # convert to grayscale and apply median blur
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.medianBlur(img_gray, 3)
        # cv2.imshow("grayscale+median blur",img_color)
        # cv2.waitKey(0)

        # -- STEP 4 --
        # detect and enhance edges
        img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 9, 2)
        # cv2.imshow("edge",img_edge)
        # cv2.waitKey(0)

        # -- STEP 5 --
        # convert back to color so that it can be bit-ANDed with color image
        (x, y, z) = img_color.shape
        img_edge = cv2.resize(img_edge, (y, x))
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
        # cv2.imwrite("edge.png", img_edge)
        # cv2.imshow("step 5", img_edge)
        # cv2.waitKey(0)
        # img_edge = cv2.resize(img_edge,(i for i in img_color.shape[:2]))
        # print img_edge.shape, img_color.shape
        return cv2.bitwise_and(img_color, img_edge)

    def generate_cartoonized_images(self, destination_path, file_name_root,
                                    min_num_down_samples=None, max_num_down_samples=None,
                                    min_num_bilateral_filters=None, max_num_bilateral_filters=None,
                                    min_median_blur=1,
                                    max_median_blur=11, min_edge_block_size=9, max_edge_block_size=9):
        if not self.source_image_path:
            print("Must supply a valid image path!")
            return
        if not self.source_image_path.is_file():
            print("Must supply a valid image path! {} is not a valid file".format(self.source_image_path))
            return
        if not min_num_down_samples:
            min_num_down_samples = 0  # number of downscaling steps
        if not max_num_down_samples:
            max_num_down_samples = 8  # number of downscaling steps
        if not min_num_bilateral_filters:
            min_num_bilateral_filters = 0  # number of bilateral filtering steps
        if not max_num_bilateral_filters:
            max_num_bilateral_filters = 50  # number of bilateral filtering steps
        for num_down_samples in range(min_num_down_samples, max_num_down_samples, 2):  # number of downscaling steps
            for num_bilateral_filters in range(min_num_bilateral_filters, max_num_bilateral_filters, 5):  # number of downscaling steps
                for median_blur in range(min_median_blur, max_median_blur, 2):
                    for edge_block_size in range(min_edge_block_size, max_edge_block_size):
                        named_options = "downsamples-{}_bilaterals-{}_medianBlur-{}".format(num_down_samples,
                                                                                            num_bilateral_filters,
                                                                                            median_blur)
                        generated_image_file_path = destination_path.joinpath("{}_{}.jpg".format(file_name_root,
                                                                                                 named_options))
                        # print("Reading original file {}".format(str(self.source_image_path)))
                        img_rgb = cv2.imread(str(self.source_image_path))

                        # downsample image using Gaussian pyramid
                        img_color = img_rgb
                        for _ in range(num_down_samples):
                            img_color = cv2.pyrDown(img_color)

                        # repeatedly apply small bilateral filter instead of applying
                        # one large filter
                        for _ in range(num_bilateral_filters):
                            img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)

                        # upsample image to original size
                        for _ in range(num_down_samples):
                            img_color = cv2.pyrUp(img_color)

                        # convert to grayscale and apply median blur
                        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                        img_blur = cv2.medianBlur(img_gray, median_blur)

                        # -- STEP 4 --
                        # detect and enhance edges
                        img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                                         cv2.THRESH_BINARY, blockSize=edge_block_size, C=2)

                        # -- STEP 5 --
                        # convert back to color so that it can be bit-ANDed with color image
                        (x, y, z) = img_color.shape
                        img_edge = cv2.resize(img_edge, (y, x))
                        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
                        # cv2.imwrite("edge.png", img_edge)

                        res = cv2.bitwise_and(img_color, img_edge)

                        cv2.imwrite(str(generated_image_file_path), res)
                        print("Wrote out image file {} created with # down samples {} and # bilateral filters {}".format(
                            generated_image_file_path, num_down_samples, num_bilateral_filters))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", dest="verbose", action="store_true", help="Flag to print verbose log messages.")
    parser.add_argument("-f", "--file", dest="file_name", required=True, help="Image file to cartoonize.")
    parser.add_argument("-s", "--downScale", dest="down_scale", help="Downscale range.")
    parser.add_argument("-d", "--destination", dest="destination", help="Root directory to place generated images",
                        default="C:\\Users\\fergu\\Desktop\\cartoonizer_testers\\python")
    args = parser.parse_args()

    image_file_path = pathlib.Path(args.file_name)
    if not image_file_path.exists():
        print("Cannot find {}, so cannot cartoonize it!".format(image_file_path))
    image_file_stem = image_file_path.stem

    tmp_canvas = Cartoonizer(image_file_path)
    root_destination_path = pathlib.Path(args.destination)
    destination_path = root_destination_path.joinpath("{}_{}".format(
        image_file_stem, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))

    if not destination_path.exists():
        print("Creating destination directory {}".format(destination_path))
        destination_path.mkdir(parents=True)

    # tmp_canvas.generate_cartoonized_images(destination_path, image_file_stem)
    tmp_canvas.generate_cartoonized_images(destination_path, image_file_stem,
                                           min_num_down_samples=0, max_num_down_samples=10,
                                           min_num_bilateral_filters=0, max_num_bilateral_filters=10,
                                           min_median_blur=1, max_median_blur=11,
                                           min_edge_block_size=7, max_edge_block_size=11)

    print("Finished.")