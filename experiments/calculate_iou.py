# https://pytorch.org/vision/main/generated/torchvision.ops.box_iou.html#torchvision.ops.box_iou
def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou


# x1, y1, x2, y2 = [100, 200, 300, 400]
# x3, y3, x4, y4 = [400, 500, 600, 700]
x1, y1, x2, y2 = [100, 200, 300, 400]
x3, y3, x4, y4 = [200, 300, 300, 400]

intersection_x_min = max(x1, x3)
intersection_y_min = max(y1, y3)

intersection_x_max = min(x2, x4)
intersection_y_max = min(y3, y4)

print(intersection_x_min, intersection_y_min, intersection_x_max, intersection_y_max)

import torch
from torchvision import ops
bbox1 = torch.tensor([x1, y1, x2, y2])
bbox2 = torch.tensor([x3, y3, x4, y4])

intersection = ops.box_iou(bbox1.view(1, -1), bbox2.view(1, -1))
print(intersection.item())
intersection = ops.box_iou(bbox2.view(1, -1), bbox1.view(1, -1))
print(intersection.item())
intersection = ops.box_iou(bbox1.view(1, -1), bbox1.view(1, -1))
print(intersection.item())

print(bb_intersection_over_union(bbox1, bbox2))
print(bb_intersection_over_union(bbox2, bbox1))
print(bb_intersection_over_union(bbox1, bbox1))

