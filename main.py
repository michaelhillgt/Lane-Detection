import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
#from IPython.display import HTML


'''
Helper Functions
'''


import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)



def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    #printing out some stats and plotting
    #print('This image is:', type(image), 'with dimensions:', image.shape)

    #plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
    #plt.show()


    # grayscale
    grayed_image = grayscale(image)
    #plt.imshow(grayed_image,cmap='gray')
    #plt.show()

    # gaussian blur
    kernel_size = 5
    blur_image = gaussian_blur(grayed_image, kernel_size)
    #plt.imshow(blur_image,cmap='gray')
    #plt.show()

    # canny
    low_threshold = 50
    high_threshold = 150
    canny_image = canny(blur_image, low_threshold, high_threshold)
    #plt.imshow(canny_image,cmap='gray')
    #plt.show()

    # mask / region of interest

    mask = np.zeros_like(canny_image)
    ignore_mask_color = 255

    # define a four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(120,imshape[0]),(440, 326), (530, 326), (920,imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(canny_image, vertices)

    # curr_points = np.copy(vertices[0])
    # x = [curr_points[0][0], curr_points[1][0], curr_points[2][0], curr_points[3][0]]
    # y = [curr_points[0][1], curr_points[1][1], curr_points[2][1], curr_points[3][1]]
    # plt.imshow(canny_image,cmap='gray')
    # plt.plot(x, y, 'b--', lw=4)
    # plt.show()

    # plt.imshow(masked_edges,cmap='gray')
    # plt.show()




    # hough transform

    # Define the Hough transform parameters
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 10 #minimum number of pixels making up a line
    max_line_gap = 10    # maximum gap in pixels between connectable line segments

    lines_image = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap) # masked_edges
    #plt.imshow(lines_image,cmap='gray')
    #plt.show()

    #color_edges = draw_lines(line_image, lines)
    w_image = weighted_img(lines_image, image)
    #plt.imshow(w_image)
    #plt.show()
    #print(STOP)

    return w_image #result



if __name__ == '__main__':


    '''
    Ideas for Lane Detection Pipeline
    Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:
        cv2.inRange() for color selection
        cv2.fillPoly() for regions selection
        cv2.line() to draw lines on an image given endpoints
        cv2.addWeighted() to coadd / overlay two images cv2.cvtColor() to grayscale or change color cv2.imwrite() to output images to file
        cv2.bitwise_and() to apply a mask to an image
    Check out the OpenCV documentation to learn about these and discover even more awesome functionality!
    '''


    #print( os.listdir("test_images/") )

    # for filename in os.listdir("test_images/"):
    #     #print('\nlooking at',filename)
    #     #reading in an image
    #     image = mpimg.imread('test_images/'+filename) # test_images/solidWhiteRight.jpg
    #     process_image(image)

    for filename in os.listdir("test_videos/"):

        output_filename = 'my_output/' + filename #'test_videos_output/solidWhiteRight.mp4'
        ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
        ## To do so add .subclip(start_second,end_second) to the end of the line below
        ## Where start_second and end_second are integer values representing the start and end of the subclip
        ## You may also uncomment the following line for a subclip of the first 5 seconds
        ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
        clip1 = VideoFileClip('test_videos/' + filename)
        white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!

        white_clip.write_videofile(output_filename, audio=False)














##
