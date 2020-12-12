import numpy as np
from scipy import ndimage
import cv2
import tensorflow as tf
import math
from custom.preprocessing.preprocessing_ops import ds_as_df
from custom.config.dataset_config import Dataset
import matplotlib.pyplot as plt
from custom.nn_util.detection_visualization import draw_boxes, draw_annotations
from custom.preprocessing.dataset_visualizer import parse_example


"""First implementation -  correct one"""
def augment_rotate(img, bboxes, angle):
    w, h = img.shape[1], img.shape[0]
    img = ndimage.rotate(img, -angle, reshape=False)
    bboxes = np.asarray([rotate_single_bbox(bbox, h, w, angle) for bbox in bboxes])

    return img, bboxes

def rotate_single_bbox(bbox, image_height, image_width, degrees):
    image_height = np.float32(image_height)
    image_width = np.float32(image_width)

    # Convert from degrees to radians.
    degrees_to_radians = math.pi / 180.0
    radians = degrees * degrees_to_radians

    # Translate the bbox to the center of the image and turn the normalized 0-1
    # coordinates to absolute pixel locations.
    # Y coordinates are made negative as the y axis of images goes down with
    # increasing pixel values, so we negate to make sure x axis and y axis points
    # are in the traditionally positive direction.
    min_y = -(np.int32(image_height * (bbox[0] - 0.5)))
    min_x = np.int32(image_width * (bbox[1] - 0.5))
    max_y = -(np.int32(image_height*(bbox[2] - 0.5)))
    max_x = np.int32(image_width * (bbox[3] - 0.5))

    coordinates = np.stack(
        [[min_y, min_x], [min_y, max_x], [max_y, min_x], [max_y, max_x]])
    coordinates = np.float32(coordinates)
    # Rotate the coordinates according to the rotation matrix clockwise if
    # radians is positive, else negative
    rotation_matrix = np.stack(
        [[np.cos(radians), np.sin(radians)],
         [-np.sin(radians), np.cos(radians)]])
    new_coords = np.int32(np.matmul(rotation_matrix, np.transpose(coordinates)))

    # Find min/max values and convert them back to normalized 0-1 floats.
    min_y = -(np.float32(np.amax(new_coords[0, :])) / image_height - 0.5)
    min_x = np.float32(np.amin(new_coords[1, :])) / image_width + 0.5
    max_y = -(np.float32(np.amin(new_coords[0, :])) /image_height - 0.5)
    max_x = np.float32(np.amax(new_coords[1, :]))/ image_width + 0.5

    # Clip the bboxes to be sure the fall between [0, 1].
    min_y, min_x, max_y, max_x = __clip_bbox(min_y, min_x, max_y, max_x)
    min_y, min_x, max_y, max_x = __check_bbox_area(min_y, min_x, max_y, max_x)

    return np.stack([min_y, min_x, max_y, max_x])


def __clip_bbox(min_y, min_x, max_y, max_x):
    """Clip bounding box coordinates between 0 and 1.
    Args:
      min_y: Normalized bbox coordinate of type float between 0 and 1.
      min_x: Normalized bbox coordinate of type float between 0 and 1.
      max_y: Normalized bbox coordinate of type float between 0 and 1.
      max_x: Normalized bbox coordinate of type float between 0 and 1.
    Returns:
      Clipped coordinate values between 0 and 1.
    """
    min_y = np.clip(min_y, 0.0, 1.0)
    min_x = np.clip(min_x, 0.0, 1.0)
    max_y = np.clip(max_y, 0.0, 1.0)
    max_x = np.clip(max_x, 0.0, 1.0)
    return min_y, min_x, max_y, max_x


def __check_bbox_area(min_y, min_x, max_y, max_x, delta=0.05):
    """Adjusts bbox coordinates to make sure the area is > 0.
    Args:
      min_y: Normalized bbox coordinate of type float between 0 and 1.
      min_x: Normalized bbox coordinate of type float between 0 and 1.
      max_y: Normalized bbox coordinate of type float between 0 and 1.
      max_x: Normalized bbox coordinate of type float between 0 and 1.
      delta: Float, this is used to create a gap of size 2 * delta between
        bbox min/max coordinates that are the same on the boundary.
        This prevents the bbox from having an area of zero.
    Returns:
      Tuple of new bbox coordinates between 0 and 1 that will now have a
      guaranteed area > 0.
    """
    height = max_y - min_y
    width = max_x - min_x

    def __adjust_bbox_boundaries(min_coord, max_coord):
        # Make sure max is never 0 and min is never 1.
        max_coord = np.maximum(max_coord, 0.0 + delta)
        min_coord = np.minimum(min_coord, 1.0 - delta)
        return min_coord, max_coord

    if height == 0.0:
        min_y, max_y = __adjust_bbox_boundaries(min_y, max_y)
    if width == 0.0:
        min_x, max_x = __adjust_bbox_boundaries(min_x, max_x)

    return min_y, min_x, max_y, max_x

""" Second implementation - works only on absolute coordinates"""
'''
def augment_rotate(img, bboxes, angle):
    #print('0', bboxes)
    # Rotate image, no expanding
    w, h = img.shape[1], img.shape[0]
    img = ndimage.rotate(img, angle, reshape=True)
    cx, cy = w // 2, h // 2

    # xcenter and y_center
    abs_coord = np.zeros(shape=np.shape(bboxes))
    abs_coord[:,0] = bboxes[:, 0] * w #xcenter
    abs_coord[:,1] = bboxes[:, 1] * h #ycenter
    abs_coord[:,2] = bboxes[:, 2] * w # width
    abs_coord[:,3] = bboxes[:, 3] * h #height

    # convert to absolute coordinates
    bboxes_abs = np.zeros(shape=np.shape(bboxes))
    bboxes_abs[:, 0] = abs_coord[:, 0] - abs_coord[:, 2] * 0.5
    bboxes_abs[:, 1] = abs_coord[:, 1] - abs_coord[:, 3] * 0.5
    bboxes_abs[:, 2] = abs_coord[:, 0] + abs_coord[:, 2] * 0.5
    bboxes_abs[:, 3] = abs_coord[:, 1] + abs_coord[:, 3] * 0.5

    corners = get_corners(bboxes_abs)
    corners = np.hstack((corners, bboxes_abs[:, 4:]))
    corners[:, :8] = rotate_box(corners[:, :8], angle, cx, cy, h, w)
    new_bbox = get_enclosing_box(corners)
    scale_factor_x = img.shape[1] / w
    scale_factor_y = img.shape[0] / h

    img = cv2.resize(img, (w, h))
    new_bbox[:, :4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]
    bboxes = new_bbox
    bboxes = clip_box(bboxes, [0, 0, w, h], 0.25)

    # convert back to Darknet coordinates
    bboxes[:, 0] = ((bboxes[:, 2] - bboxes[:, 0]) / 2 + bboxes[:, 0]) / w
    bboxes[:, 1] = ((bboxes[:, 3] - bboxes[:, 1]) / 2 + bboxes[:, 1]) / h
    bboxes[:, 2] = (bboxes[:, 2] - bboxes[:, 0]) / w
    bboxes[:, 3] = (bboxes[:, 3] - bboxes[:, 1]) / h
    # TODO: return to old format
    #print('1:', bboxes)
    return img, bboxes
    
    def get_enclosing_box(corners):
    """Get an enclosing box for ratated corners of a bounding box
    https://github.com/Paperspace/DataAugmentationForObjectDetection/blob/master/data_aug/bbox_util.py
    Parameters
    ----------

    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    Returns
    -------

    numpy.ndarray
        Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    """
    x_ = corners[:, [0, 2, 4, 6]]
    y_ = corners[:, [1, 3, 5, 7]]

    xmin = np.min(x_, 1).reshape(-1, 1)
    ymin = np.min(y_, 1).reshape(-1, 1)
    xmax = np.max(x_, 1).reshape(-1, 1)
    ymax = np.max(y_, 1).reshape(-1, 1)

    final = np.hstack((xmin, ymin, xmax, ymax, corners[:, 8:]))

    return final


def get_corners(bboxes):
    """Get corners of bounding boxes
    See https://github.com/Paperspace/DataAugmentationForObjectDetection/blob/master/data_aug/bbox_util.py
    Parameters
    ----------

    bboxes: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    returns
    -------

    numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    """
    width = (bboxes[:, 2] - bboxes[:, 0]).reshape(-1, 1)
    height = (bboxes[:, 3] - bboxes[:, 1]).reshape(-1, 1)

    x1 = bboxes[:, 0].reshape(-1, 1)
    y1 = bboxes[:, 1].reshape(-1, 1)

    x2 = x1 + width
    y2 = y1

    x3 = x1
    y3 = y1 + height

    x4 = bboxes[:, 2].reshape(-1, 1)
    y4 = bboxes[:, 3].reshape(-1, 1)

    corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))

    return corners


def bbox_area(bbox):
    return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])


def clip_box(bbox, clip_box, alpha):
    """Clip the bounding boxes to the borders of an image
    https://github.com/Paperspace/DataAugmentationForObjectDetection/blob/master/data_aug/bbox_util.py
    Parameters
    ----------

    bbox: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    clip_box: numpy.ndarray
        An array of shape (4,) specifying the diagonal co-ordinates of the image
        The coordinates are represented in the format `x1 y1 x2 y2`

    alpha: float
        If the fraction of a bounding box left in the image after being clipped is
        less than `alpha` the bounding box is dropped.

    Returns
    -------

    numpy.ndarray
        Numpy array containing **clipped** bounding boxes of shape `N X 4` where N is the
        number of bounding boxes left are being clipped and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    """
    ar_ = (bbox_area(bbox))
    x_min = np.maximum(bbox[:, 0], clip_box[0]).reshape(-1, 1)
    y_min = np.maximum(bbox[:, 1], clip_box[1]).reshape(-1, 1)
    x_max = np.minimum(bbox[:, 2], clip_box[2]).reshape(-1, 1)
    y_max = np.minimum(bbox[:, 3], clip_box[3]).reshape(-1, 1)

    bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:, 4:]))

    delta_area = ((ar_ - bbox_area(bbox)) / ar_)

    mask = (delta_area < (1 - alpha)).astype(int)

    bbox = bbox[mask == 1, :]

    return bbox


def rotate_box(corners, angle, cx, cy, h, w):
    """Rotate the bounding box.
    See https://github.com/Paperspace/DataAugmentationForObjectDetection/blob/master/data_aug/bbox_util.py

    Parameters
    ----------

    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    angle : float
        angle by which the image is to be rotated

    cx : int
        x coordinate of the center of image (about which the box will be rotated)

    cy : int
        y coordinate of the center of image (about which the box will be rotated)

    h : int
        height of the image

    w : int
        width of the image

    Returns
    -------

    numpy.ndarray
        Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    """

    corners = corners.reshape(-1, 2)
    corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M, corners.T).T

    calculated = calculated.reshape(-1, 8)

    return calculated

'''

""" Third implementation - weird colors"""
'''
def augment_rotate(img, bboxes, angle):
    height = img.shape[0]
    width = img.shape[1]
    # Rotation and Scale
    R = np.eye(3)

    R[:2] = cv2.getRotationMatrix2D(angle=angle, center=(img.shape[1] / 2, img.shape[0] / 2), scale=1)

    # Combined rotation matrix
    img = cv2.warpAffine(img, R[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(bboxes)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = bboxes[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ R.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        print(xy)

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        area0 = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 4] - bboxes[:, 2])
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.2) & (ar < 10)

        bboxes = bboxes[i]
        bboxes[:, 1:5] = xy[i]

    return img, bboxes
'''


if __name__ == '__main__':
    ds = Dataset('chess_12')
    nb_samples = 2

    ds_tfrecord = tf.data.TFRecordDataset(ds.tf_record_test).take(nb_samples)
    ds_tfrecord = ds_tfrecord.map(lambda x: parse_example(x, img_size=416))
    ax=[]
    fig = plt.figure(figsize=(8 * nb_samples, 8))

    for idx, (img, bboxes, _) in enumerate(ds_tfrecord):

        ax.append(fig.add_subplot(nb_samples, 2, idx*2+1))

        axis_font = {'fontname': 'DejaVu Sans', 'size': '18'}
        ax[-1].set_title('Original image', fontdict=axis_font)

        img = img.numpy()
        bboxes_original = bboxes.numpy()
        bboxes_original[:,0] *= ds.image_width
        bboxes_original[:,1] *= ds.image_height
        bboxes_original[:,2] *= ds.image_width
        bboxes_original[:,3] *= ds.image_height
        original_img = draw_boxes(img, bboxes_original, color=(0, 255, 0))
        plt.imshow(original_img)

        angle = np.random.uniform(-20, 20)
        #bboxes_augment = np.hstack([np.zeros(shape=(bboxes.numpy().shape[0], 1)) ,bboxes.numpy()])
        img_augment, bboxes_augment = augment_rotate(img, bboxes.numpy(), angle)
        bboxes_augment[:,0] *= ds.image_width
        bboxes_augment[:,1] *= ds.image_height
        bboxes_augment[:,2] *= ds.image_width
        bboxes_augment[:,3] *= ds.image_height
        ax.append(fig.add_subplot(nb_samples, 2, idx*2+2))
        ax[-1].set_title('Rotated image', fontdict=axis_font)
        img_augment = draw_boxes(img_augment, bboxes_augment, color=(0, 255, 0))
        plt.imshow(img_augment)

        axis_font = {'fontname': 'DejaVu Sans', 'size': '18'}
        ax[-1].set_title('Original image', fontdict=axis_font)

    plt.show()