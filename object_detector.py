import cv2
import sys
import numpy as np
import tensorflow as tf
from drone_video import Video


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

##
from utils import label_map_util
from utils import visualization_utils as vis_util
print("Tensorflow version {}".format(tf.__version__))


def load_image_into_numpy_array(image):
  (im_width, im_height, oi) = image.shape
  #print (image.shape())
  return np.array(image).reshape((im_width, im_height, 3)).astype(np.uint8)

def run_inference(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

def show_image(image):
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference(image_np, detection_graph)
    # Visualization of the results of a detection.
    image_np = vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)
    cv2.imshow("detection", image_np)


## Loads the frozen tensorflow model
detection_graph = tf.Graph()
print("oi")
with detection_graph.as_default():
    print("oi")
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile("frozen_model.pb", 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

## Loads the label map GFile

category_index = label_map_util.create_category_index_from_labelmap("label_map.pbtxt", use_display_name=True)


#cap = cv2.VideoCapture(-1)
video = Video()
while True:
    print("Entrou no while")
    if not video.frame_available():
        continue
    image = video.frame()
    #obj = run_inference(image, detection_graph)
    #print(obj)
    #cv2.imshow("Objects", obj)
    show_image(image)
    #cv2.imshow("Camera", image)
    #### MAGICA ######
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    #################
cap.release()
cv2.destroyAllWindows()
# if __name__ == "__main__":
#     main()
