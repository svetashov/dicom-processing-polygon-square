import pydicom as dicom
import numpy as np
import cv2
from tkinter import filedialog
 
points = []
current_image = np.zeros((1000, 1000, 3), np.uint8)
original = np.zeros((1000, 1000, 3), np.uint8)
pixel_spacing = (1, 1)

# calculates square of any polygon
def square(points):
    square_value = 0
    for i in range(0, len(points)):
        p1 = points[i-1]
        p2 = points[i]
        square_value += (p1[0] - p2[0]) * (p1[1] + p2[1])
    return abs(square_value) / 2

# scale pixels to milimeters
def scale_points(points, scale_x, scale_y):
    new_points = []
    for p in points:
        new_points.append((p[0] * scale_x, p[1] * scale_y))
    return new_points

# adds point to image and draw line
def add_point(image, x, y):
    points.append((x, y))
    if len(points) > 1:
        cv2.line(image, points[-1], points[-2], (255, 0, 255))
    cv2.rectangle(image,(x-1,y-1), (x+1,y+1), (255,0,255))

# removes selection and refreshes the image
def clear_selection(image):
    global current_image
    current_image = original.copy()
    points.clear()

# draws line from last point to first and calculates the square of figure
def complete_selection(image):
    if len(points) > 2:
        cv2.line(image, points[-1], points[0], (255, 0, 255))
        mm_square = square(scale_points(points, pixel_spacing[0], pixel_spacing[1]))
        cm_square = mm_square / 100
        cv2.putText(image, 'S=' + '{0:.4g}'.format(cm_square) + 'cm^2', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 255))

def click_event(event, x, y, flags, params):
    global current_image
    # Left click - add a point
    if event == cv2.EVENT_LBUTTONDOWN:
        add_point(current_image, x, y)

    # Right click - complete polygon and calculate square
    if event == cv2.EVENT_RBUTTONDOWN:
        complete_selection(current_image)

    # Double click - remove square
    if event == cv2.EVENT_LBUTTONDBLCLK:
        clear_selection(current_image)


if __name__ == '__main__':
    image_path = filedialog.askopenfilename(initialdir="/", title="Select file",
                    filetypes=(("DICOM files", "*.dcm"),("all files", "*.*")))
    # reading the image
    dimage = dicom.dcmread(image_path)
    # extract density
    if dimage.get('PixelSpacing') != None:
        pixel_spacing = dimage.get('PixelSpacing')
    array = dimage.pixel_array.astype(float)
    # prepare image: convert to grayscale 0..255
    image_2d_scaled = ((np.maximum(array,0)) / array.max()) * 255.0
    image_2d_scaled = np.uint8(image_2d_scaled)

    # convert to RGB
    original = cv2.cvtColor(image_2d_scaled, cv2.COLOR_GRAY2RGB)
    current_image = original.copy()

    # show image
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', click_event)
    cv2.imshow('image', current_image)

    while(1):
        cv2.imshow('image', current_image)
        # press 'Esc' to exit
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()  

