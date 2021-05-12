import openpose_helper
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

    def find_keypoints(self, image_path, visualize=False, lower_bound=5, threshold=0.5):
        '''
        Find keypoints on an image and return them
        '''
        points, frame, count = openpose_helper.find_keypoints(image_path, self.openpose, visualize=visualize, threshold=0.5)
        if visualize and count>lower_bound:
            cv2.imshow('keypoints', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return points, count

    def label_data(self, image_path, save_name='keypoint_data.json'):
        '''
        Label and save all data in a given path
        '''
        data, labels, images = [], [], []
        for image in os.listdir(image_path):
            keypoints, count = self.find_keypoints(f'{image_path}/{image}', visualize=False)
            if count>=5:
                images.append(image)
                data.append(keypoints)
                if 'bad' in image:
                    labels.append(0)
                elif 'good' in image:
                    labels.append(1)

        json_data = {'data': data, 'labels': labels, 'used_images': images}
        print(len(images))
        with open(save_name, 'w') as json_file:
            json.dump(json_data, json_file)


    def build_model(self):
        '''
        Building our model for posture analysis based on keypoints
        '''
        pass

if __name__ == '__main__':
    # print('begin')
    # x = pose_classifier('openpose_model')
    # image_path = 'data'
    # x.label_data(image_path)
    with open('keypoint_data.json') as f:
        data = json.load(f)
    print(len(data['used_images']))
