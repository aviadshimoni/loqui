import face_alignment
import cv2
import numpy as np
from matplotlib import pyplot as plt


def plot_frames(frames_to_plot):
    num_frames = 29
    for i in range(num_frames):
        frame = frames_to_plot[i]  # Extract the frame
        if frame.ndim == 3:  # If the frame is 3D, reshape it to 2D
            frame = frame.squeeze()
        plt.imshow(frame, cmap='gray')
        plt.axis('off')
        plt.show()

def get_faces(frames):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2halfD, device='cpu', face_detector='sfd')
    plot_frames(frames)
    all_landmarks = []
    for frame in frames:
        points_list = fa.get_landmarks(frame)
        all_landmarks.append(points_list)

    return  all_landmarks


def get_position(size, padding=0.25):

    x = [0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
                    0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
                    0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
                    0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
                    0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
                    0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
                    0.553364, 0.490127, 0.42689]

    y = [0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
                    0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
                    0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
                    0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
                    0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
                    0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
                    0.784792, 0.824182, 0.831803, 0.824182]

    x, y = np.array(x), np.array(y)

    x = (x + padding) / (2 * padding + 1)
    y = (y + padding) / (2 * padding + 1)
    x = x * size
    y = y * size
    return np.array(list(zip(x, y)))

def cal_area(anno):
    return (anno[:,0].max() - anno[:,0].min()) * (anno[:,1].max() - anno[:,1].min())


def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])



def anno_img(images, annos):

    shapes = []
    for i in range(len(images)):
        anno = annos[i]
        anno = sorted(anno, key = cal_area, reverse=True)[0]
        shapes.append(anno[17:])


    front256 = get_position(256)
    M_prev = None
    frames=[]

    for (shape, img) in zip(shapes, images):
        M = transformation_from_points(np.matrix(shape), np.matrix(front256))
        img = cv2.warpAffine(img, M[:2], (256, 256))
        (x, y) = front256[-20:].mean(0).astype(np.int32)
        w = 160//2
        img = img[y-w//2:y+w//2,x-w:x+w,...]
        frames.append(img)

    return np.array(frames)