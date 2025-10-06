# https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097
# Adjusted base code found in the above link to work for a central rectangle in
# a video feed.
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def find_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist
    
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

cap = cv.VideoCapture(0)

while(1):
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w, _ = frame.shape
    rect_w, rect_h = 100, 100
    x1, y1 = w // 2 - rect_w // 2, h // 2 - rect_h // 2
    x2, y2 = x1 + rect_w, y1 + rect_h
    
    cv.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    
    cen = frame[y1:y2, x1:x2]
    cen_rgb = cv.cvtColor(cen, cv.COLOR_BGR2RGB)

    img = cen_rgb.reshape((cen_rgb.shape[0] * cen_rgb.shape[1],3)) #represent as row*column,channel number
    clt = KMeans(n_clusters=1) #cluster number
    clt.fit(img)

    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)
    
    bar_bgr = cv.cvtColor(bar, cv.COLOR_RGB2BGR)
    
    frame[0:50, w//2-150:w//2+150] = bar_bgr
    
    cv.imshow('Frame',frame)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()
