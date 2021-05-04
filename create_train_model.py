from openpose_model import openpose_helper
import json
import cv2
import os


class pose_classifier():
    '''
    Class for binary classification of images of people sitting
    Labels will be either 1(good) or 0(bad)
    Functions for creating data and labels
    '''

    def __init__(self, path_to_model):
        '''
        Initialize class and load in openpose model
        '''
        self.openpose = openpose_helper.load_model(path_to_model)

    def find_keypoints(self, image_path, visualize=False):
        '''
        Find keypoints on an image and return them
        '''
        points, frame = openpose_helper.find_keypoints(image_path, self.openpose, visualize=visualize)
        if visualize:
            cv2.imshow('keypoints', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return points

    def label_data(self, image_path, save_path, save_name='keypoint_data.json'):
        '''
        Label and save all data in a given path
        '''
        data, labels = [], []
        for image in os.listdir(image_path):
            if 'bad' in image:
                labels.append(0)
            elif 'good' in image:
                labels.append(1)
            data.append(self.find_keypoints(image_path))
        json_data = {'data': data, 'labels': labels}
        with open(save_name, 'w') as json_file:
            json.dump(json_data, json_file)


if __name__ == '__main__':
    x = pose_classifier('openpose_model')
    image_path = 'openpose_model/image_1.jpeg'
    points = x.find_keypoints(image_path)
    print(points)
