import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../img_asset/platnomor.jpg")
plt.imshow(img)
plt.show()

sr = cv2.dnn_superres.DnnSuperResImpl_create()
path = "models/ESPCN_x4.pb"
sr.readModel(path)
sr.setModel("espcn",4)
sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
result = sr.upsample(img)
# SR upscaled
plt.imshow(result)
plt.show()