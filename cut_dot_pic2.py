# get the image first
from __future__ import division, absolute_import
import os
import cv2
import numpy as np
from copy import deepcopy

the_image_path = "/home/omnisky/PycharmProjects/yolo_snip/DOTA/PNGimages"
the_new_data_path = "/home/omnisky/PycharmProjects/yolo_snip/DOTA/new_PNGimage"
train_file = "/home/omnisky/PycharmProjects/yolo_snip/DOTA/train.txt"

if not os.path.exists(the_new_data_path):
    os.mkdir(the_new_data_path)
train_f = open(train_file, 'w')
for path in os.listdir(the_image_path):
    if path.split('.')[1] == 'png':
        f = open(os.path.join(the_image_path, path.split('.')[0] + '.txt'))
        # get the coordinate
        the_lines = f.readlines()
        all_lines = [[float(lt) for lt in line.strip().split()] for line in the_lines]
        img = cv2.imread(os.path.join(the_image_path, path))
        if img.shape[0] < 1024 and img.shape[1] < 1024:
            the_new_img_name = path.split('.')[0] + '.png'
            the_new_img_path = os.path.join(the_new_data_path, the_new_img_name)
            h, w, _ = img.shape
            if (len(all_lines) == 0):
                continue
            cv2.imwrite(the_new_img_path, img)
            all_lines = np.array(all_lines)
            the_new_name = path.split('.')[0] + '.txt'
            the_new_path = os.path.join(the_new_data_path, the_new_name)
            f = open(the_new_path, 'w')
            train_f.writelines(the_new_img_path)
            train_f.write('\n')
            for it in all_lines:
                str_it = [str(t) for t in it]
                f.writelines(' '.join(str_it))
                f.write('\n')
            f.close()

        else:
            # grid the coor
            h, w, _ = img.shape
            the_start_coor = [[0, 0, 1024, 1024]]
            print(the_start_coor[-1])
            if (the_start_coor[-1][2] < w):
                while the_start_coor[-1][2] < w:
                    the_new_coor = [the_start_coor[-1][0] + 512, the_start_coor[-1][1], the_start_coor[-1][2] + 512,
                                    the_start_coor[-1][3]]
                    the_start_coor.append(the_new_coor)
                if the_start_coor[-1][2] > w:
                    the_start_coor[-1][0] -= (the_start_coor[-1][2] - w)
                    the_start_coor[-1][2] = w
            else:
                the_start_coor = [[0, 0, w, 1024]]
            # h
            the_all_coor = []
            for coor in the_start_coor:
                the_h_coor = [coor]
                if (the_h_coor[-1][3] < h):
                    while the_h_coor[-1][3] < h:
                        the_new_coor = [the_h_coor[-1][0], the_h_coor[-1][1] + 512, the_h_coor[-1][2],
                                        the_h_coor[-1][3] + 512]
                        the_h_coor.append(the_new_coor)
                    if the_h_coor[-1][3] > h:
                        the_h_coor[-1][1] -= (the_h_coor[-1][3] - h)
                        the_h_coor[-1][3] = h
                else:
                    the_h_coor[-1][3] = h
                the_all_coor.append(the_h_coor)
            debug = 1
            the_img = {}
            # get the correspond image and the target
            for i, all_coor in enumerate(the_all_coor):
                for j, coor in enumerate(all_coor):
                    the_img[i * len(the_all_coor[0]) + j] = img[coor[1]:coor[3], coor[0]:coor[2], :]
            # write the image

            the_label_all = {}
            all_lines = [[line[0], line[1] * w, line[2] * h, line[3] * w, line[4] * h] for line in all_lines]
            all_lines = np.array(all_lines)
            for i, all_coor in enumerate(the_all_coor):
                # clip the bbox into the corresponding area
                for j, coor in enumerate(all_coor):
                    # x,y-the_offset of the coor_area
                    try:
                        tmp_lines = all_lines[:, 1:3] - np.array([[coor[0], coor[1]]])
                        the_class_tag = tmp_lines[:, 0]
                    except:
                        continue
                    # filter the proper tag
                    the_proper_tag = np.all(
                        [tmp_lines[:, 0] >= 0, tmp_lines[:, 1] >= 0, tmp_lines[:, 0] < 1024, tmp_lines[:, 1] < 1024],
                        axis=0)

                    # fil
                    the_proper_lines = all_lines[the_proper_tag].copy()
                    if the_proper_lines.shape[0] == 0:
                        continue
                    else:
                        the_proper_lines[:, 1:3] -= np.array([[coor[0], coor[1]]])
                        # whether has override the bounding area
                        the_proper_lines[:, 1] = the_proper_lines[:, 1] - the_proper_lines[:, 3] / 2
                        the_proper_lines[:, 2] = the_proper_lines[:, 2] - the_proper_lines[:, 4] / 2
                        # the_right_coordinate
                        the_proper_lines[:, 3] = the_proper_lines[:, 1] + the_proper_lines[:, 3]
                        the_proper_lines[:, 4] = the_proper_lines[:, 2] + the_proper_lines[:, 4]
                        # clip the orignal bbox
                        img = the_img[i * len(the_all_coor[0]) + j]
                        h, w, _ = img.shape
                        the_proper_lines[:, 1] = np.clip(the_proper_lines[:, 1], a_min=0, a_max=w - 1)
                        the_proper_lines[:, 3] = np.clip(the_proper_lines[:, 3], a_min=0, a_max=w - 1)
                        the_proper_lines[:, 2] = np.clip(the_proper_lines[:, 2], a_min=0, a_max=h - 1)
                        the_proper_lines[:, 4] = np.clip(the_proper_lines[:, 4], a_min=0, a_max=h - 1)

                        the_proper_lines[:, 1] = (the_proper_lines[:, 1] + the_proper_lines[:, 3]) / 2
                        the_proper_lines[:, 2] = (the_proper_lines[:, 2] + the_proper_lines[:, 4]) / 2

                        the_proper_lines[:, 3] = 2 * (the_proper_lines[:, 3] - the_proper_lines[:, 1])
                        the_proper_lines[:, 4] = 2 * (the_proper_lines[:, 4] - the_proper_lines[:, 2])

                        the_proper_lines[:, 1] /= w
                        the_proper_lines[:, 3] /= w
                        the_proper_lines[:, 2] /= h
                        the_proper_lines[:, 4] /= h

                        the_label_all[i * len(the_all_coor[0]) + j] = the_proper_lines

            for key, item in the_label_all.items():
                the_new_img_name = path.split('.')[0] + str(key) + '.png'
                the_new_img_path = os.path.join(the_new_data_path, the_new_img_name)
                cv2.imwrite(the_new_img_path, the_img[key])
                the_new_name = path.split('.')[0] + str(key) + '.txt'
                the_new_path = os.path.join(the_new_data_path, the_new_name)
                train_f.writelines(the_new_img_path)
                train_f.write('\n')
                f = open(the_new_path, 'w')
                for it in item:
                    # print(int(it[0]))
                    str_it = [str(int(it[0]))] + [str(np.array(t, np.float32)) for t in it[1:]]
                    f.writelines(' '.join(str_it))
                    f.write('\n')
                f.close()
train_f.close()