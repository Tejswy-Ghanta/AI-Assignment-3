#!/usr/local/bin/python3
#
# Authors: Purnima Surve:pursurve Tejaswy Ghanta:lghanta Shruti Gutta: shrgutta
#
# Ice layer finder
# Based on skeleton code by D. Crandall, November 2021
#

from PIL import Image
from numpy import *
from scipy.ndimage import filters
import sys
import imageio
import numpy as np

# calculate "Edge strength map" of an image
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))                       #for converting to greyscale
    filtered_y = zeros(grayscale.shape)                               #pixel length as zero for the 2d matrix
    filters.sobel(grayscale,0,filtered_y)
    return sqrt(filtered_y**2)

# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
# compute edge strength mask
def bayes_net_airice(norm_edge_strength):
    return argmax(norm_edge_strength, axis=0)                        #returns the highest edge strength index columnwise

def bayes_net_icerock(norm_edge_strength):                           #returns the second highest edge strength index columnwise
    maxy=[]
    for j in range(len(norm_edge_strength[0])):
        max1 = 0
        max2 = 0
        for i in range(len(norm_edge_strength)):
            if norm_edge_strength[i][j] > max1:
                max1 = norm_edge_strength[i][j]
        for i in range(len(norm_edge_strength)):
            if norm_edge_strength[i][j] < max1 and max1-norm_edge_strength[i][j] > 10:
                if norm_edge_strength[i][j] > max2:
                    max2 = i
        maxy.append(max2)
    return maxy


def transprob(prev_row, col):                                        #returns the transition probabilities
    # dist = [abs(prev_row - row) if abs(row - prev_row) < 12 else 0.001 for row in range(len(col))]
    #
    # temp = [1 /(t + 1) if t != 0.001  else 0.001 for t in dist]
    temp = [0.9 if abs(row - prev_row) < 12 else 0.001 for row in range(len(col))]
    return asarray(reshape(temp, (-1, 1)), dtype='float32')


def emission_probability(col):                                      #returns emission probabilities
     eprob = [c/sum(col) for c in col]

     return asarray(eprob).reshape(len(eprob), 1)


def viterbii(norm_edge_strength,initial_row,initial_col):           #viterbi algorithm for air-ice boundary
    len_col = norm_edge_strength.shape[1]
    v1 = [0] * len_col
    for col in range(initial_col, len_col):
        if col == initial_col:
            v1[initial_col] = initial_row                           #Adding the initial probability using input from bayes net algorithm
        else:

            pij = transprob(v1[col-1], norm_edge_strength[:,col])
            vtr = norm_edge_strength[v1[col - 1], col-1]/sum(norm_edge_strength[:,col-1])
            v1[col] = argmax(emission_probability(norm_edge_strength[:,col])*vtr*pij)

    for col in range(initial_col - 1, -1, -1):

        pij = transprob(v1[col+1],norm_edge_strength[:,col])
        vtr = norm_edge_strength[v1[col + 1], col + 1]/sum(norm_edge_strength[:,col+1])
        v1[col] = argmax(emission_probability(norm_edge_strength[:,col])*vtr*pij)
    return v1


def viterbii_humanfeed(norm_edge_strength,initial_row,initial_col):      #viterbi algorithm for human feedback
    len_col=norm_edge_strength.shape[1]
    v1 = [0]*len_col
    if initial_row == gt_icerock[0]:
        v1[gt_airice[1]] = 0
    for col in range(initial_col,len_col):
        if col == initial_col:
            v1[initial_col] = 1
        else:

            pij = transprob(v1[col-1], norm_edge_strength[:,col])
            vtr = norm_edge_strength[v1[col - 1], col-1]/sum(norm_edge_strength[:,col-1])
            v1[col] = argmax(emission_probability(norm_edge_strength[:,col])*vtr*pij)

    for col in range(initial_col - 1, -1, -1):

        pij = transprob(v1[col+1],norm_edge_strength[:,col])
        vtr = norm_edge_strength[v1[col + 1], col + 1]/sum(norm_edge_strength[:,col+1])
        v1[col] = argmax(emission_probability(norm_edge_strength[:,col])*vtr*pij)
    return v1


def draw_boundary(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range( int(max(y-int(thickness/2), 0)), int(min(y+int(thickness/2), image.size[1]-1 )) ):
            image.putpixel((x, t), color)
    return image

def draw_asterisk(image, pt, color, thickness):
    for (x, y) in [ (pt[0]+dx, pt[1]+dy) for dx in range(-3, 4) for dy in range(-2, 3) if dx == 0 or dy == 0 or abs(dx) == abs(dy) ]:
        if 0 <= x < image.size[0] and 0 <= y < image.size[1]:
            image.putpixel((x, y), color)
    return image


# Save an image that superimposes three lines (simple, hmm, feedback) in three different colors
# (yellow, blue, red) to the filename
def write_output_image(filename, image, simple, hmm, feedback, feedback_pt):
    new_image = draw_boundary(image, simple, (255, 255, 0), 2)
    new_image = draw_boundary(new_image, hmm, (0, 0, 255), 2)
    new_image = draw_boundary(new_image, feedback, (255, 0, 0), 2)
    new_image = draw_asterisk(new_image, feedback_pt, (255, 0, 0), 2)
    imageio.imwrite(filename, new_image)


# main program
#
if __name__ == "__main__":

    if len(sys.argv) != 6:
        raise Exception("Program needs 5 parameters: input_file airice_row_coord airice_col_coord icerock_row_coord icerock_col_coord")

    input_filename = sys.argv[1]

    gt_airice = [ int(i) for i in sys.argv[2:4] ]

    gt_icerock = [ int(i) for i in sys.argv[4:6] ]

    # load in image
    input_image = Image.open(input_filename).convert('RGB')
    image_array = array(input_image.convert('L'))


    # compute edge strength mask -- in case it's helpful. Feel free to use this.
    edge_strength = edge_strength(input_image)
    norm_edge_strength = uint8(255 * edge_strength / (amax(edge_strength)))
    imageio.imwrite('edges.png', norm_edge_strength)


    # You'll need to add code here to figure out the results! For now,
    # just create some random lines.

    ridgeLength = bayes_net_airice(norm_edge_strength)
    ridgeLength2 = bayes_net_icerock(norm_edge_strength)
    # imageio.imwrite("output_airice.png", draw_boundary(input_image, ridgeLength, (255, 255, 0), 5))
    # imageio.imwrite("output_icerock.png", draw_boundary(input_image, ridgeLength2, (255, 255, 0), 5))

    initial_row = bayes_net_airice(norm_edge_strength)
    ridgeV = viterbii(norm_edge_strength,initial_row[0],0)
    #imageio.imwrite("output_airice.png", draw_boundary(input_image, ridgeV, (0, 0, 255), 5))

    initial_rowr = bayes_net_icerock(norm_edge_strength)
    ridgeVrock = viterbii(norm_edge_strength,initial_rowr[0],0)
    #imageio.imwrite("output_icerock.png", draw_boundary(input_image, ridgeVrock, (0, 0, 255), 5))



#human input part

    human_airice=viterbii_humanfeed(norm_edge_strength,gt_airice[0],gt_airice[1])
    # imageio.imwrite("lala.png", draw_boundary(input_image, final_path_part1, (255, 0, 0), 5))


    human_icerock = viterbii_humanfeed(norm_edge_strength, gt_icerock[0], gt_icerock[1])
    # imageio.imwrite("lala.png", draw_boundary(input_image, final_path_part2, (255, 0, 0), 5))


    # airice_simple = [ image_array.shape[0]*0.25 ] * image_array.shape[1]
    # airice_hmm = [ image_array.shape[0]*0.5 ] * image_array.shape[1]
    # airice_feedback= [ image_array.shape[0]*0.75 ] * image_array.shape[1]
    #
    # icerock_simple = [ image_array.shape[0]*0.25 ] * image_array.shape[1]
    # icerock_hmm = [ image_array.shape[0]*0.5 ] * image_array.shape[1]
    # icerock_feedback= [ image_array.shape[0]*0.75 ] * image_array.shape[1]


    # Now write out the results as images and a text file
    write_output_image("air_ice_output.png", input_image, ridgeLength, ridgeV, human_airice, gt_airice)
    write_output_image("ice_rock_output.png", input_image, ridgeLength2, ridgeVrock, human_icerock, gt_icerock)
    with open("layers_output.txt", "w") as fp:
        for i in (ridgeLength, ridgeV, human_airice, ridgeLength2, ridgeVrock, human_icerock):
            fp.write(str(i) + "\n")
