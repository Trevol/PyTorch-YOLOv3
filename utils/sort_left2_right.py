from trvo_utils.iter_utils import unzip


def sort_boxes(boxes, class_ids):
    def box_x1(box_class_id):
        return box_class_id[0][0]

    ordered = sorted(zip(boxes, class_ids), key=box_x1)
    ordered_boxes, ordered_class_ids = unzip(ordered, [], [])
    return ordered_boxes, ordered_class_ids


def sort_detections(yoloDetections):
    def x1(d):
        return d[0]

    return sorted(yoloDetections, key=x1)
