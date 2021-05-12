import cv2

# Load model
protoFile = 'pose_deploy_linevec.prototxt'
modelWeights = 'pose_iter_160000.caffemodel'

# Read the network into memory
net = cv2.dnn.readNetFromCaffe(protoFile, modelWeights)

frame = cv2.imread("test.jpeg")

inwidth = 256
inheight = 256
frame = cv2.resize(frame, (256, 256))
frameheight = frame.shape[0]
framewidth = frame.shape[1]

inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inwidth, inheight), (0, 0, 0), swapRB=False, crop=False)

net.setInput(inpBlob)

out = net.forward()

H = out.shape[2]
W = out.shape[3]
# print(H,W)
# Empty list to store the detected keypoints
points = []
threshold = .5

for i in range(16):
    # confidence map of corresponding body's part.
    probMap = out[0, i, :, :]
    print(i)

    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    # Scale the point to fit on the original image
    # print(point[0], point[1])
    x = (framewidth * point[0]) / W
    y = (frameheight * point[1]) / H

    if prob > threshold:
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frame, f"{i}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1, lineType=cv2.LINE_AA)

        # Add the point to the list if the probability is greater than the threshold
        points.append((int(x), int(y)))
    else:
        points.append(None)

# cv2.imshow('keypoints', frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
print(points)
