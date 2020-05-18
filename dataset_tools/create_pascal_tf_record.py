# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_pascal_tf_record.py \
        --data_dir=/home/user/VOCdevkit \
        --year=VOC2012 \
        --output_path=/home/user/pascal.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import cv2
import numpy as np

from lxml import etree
import PIL.Image
import tensorflow as tf
import sys
sys.path.append("./")

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

# validate label map

# def _validate_label_map(label_map):
#   for item in label_map.item:
#     if item.id < 0:
#       raise ValueError('Label map ids should be >= 0.')
#     if (item.id == 0 and item.name != 'background' and
#         item.display_name != 'background'):
#       raise ValueError('Label map id 0 is reserved for the background label')

# label_map_util._validate_label_map = _validate_label_map


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'trainval', 'Convert training set, validation set or '
                    'merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('year', 'VOC2007', 'Desired challenge year.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', '',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
FLAGS = flags.FLAGS

SETS = ['train', 'val', 'trainval', 'test']
YEARS = ['VOC2007', 'VOC2012', 'merged']

objects = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 
'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'pottedplant', 'sheep', 
'sofa', 'tvmonitor']

people = 0


def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=False,
                       image_subdirectory='JPEGImages', writer=None):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  # img_path = os.path.join(data['folder'], image_subdirectory, data['filename'])
  # full_path = os.path.join(dataset_directory, img_path)
  # with tf.gfile.GFile(full_path, 'rb') as fid:
  #   encoded_jpg = fid.read()
  # encoded_jpg_io = io.BytesIO(encoded_jpg)
  # image = PIL.Image.open(encoded_jpg_io)
  # if image.format != 'JPEG':
  #   raise ValueError('Image format not JPEG')
  # key = hashlib.sha256(encoded_jpg).hexdigest()

  global people

  if 'object' in data:
    for obj in data['object']:
      xmin = []
      ymin = []
      xmax = []
      ymax = []
      classes_text = []
      classes_label = []
      truncated = []
      poses = []
      difficult_obj = []

      difficult = bool(int(obj['difficult']))
      difficult_obj.append(int(difficult))
      # height
      # if np.asarray(image).shape[0] < int(obj['bndbox']['ymin']) or np.asarray(image).shape[0] < int(obj['bndbox']['ymax']):
      #   raise Exception("bounding box extension")
      # elif np.asarray(image).shape[1] < int(obj['bndbox']['xmin']) or np.asarray(image).shape[1] < int(obj['bndbox']['xmax']):
      #   raise Exception("bounding box extension")

      # if cv2.imread(os.path.join(FLAGS.data_dir, FLAGS.year, 'JPEGImages', data['filename'])).shape != np.asarray(image).shape:
      #   raise Exception("Shapes do not match")

      if obj['name'] == "person":
        people += 1

      xmin.append(int(obj['bndbox']['xmin']))
      ymin.append(int(obj['bndbox']['ymin']))
      xmax.append(int(obj['bndbox']['xmax']))
      ymax.append(int(obj['bndbox']['ymax']))
      classes_text.append(obj['name'].encode('utf8'))
      truncated.append(int(obj['truncated']))
      poses.append(obj['pose'].encode('utf8'))

      example = tf.train.Example(features=tf.train.Features(feature={
        'image/source_id': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/object/bbox/xmin': dataset_util.int64_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.int64_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.int64_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.int64_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj)
      }))
      writer.write(example.SerializeToString())
  return None


def main(_):
  if FLAGS.set not in SETS:
    raise ValueError('set must be in : {}'.format(SETS))
  if FLAGS.year not in YEARS:
    raise ValueError('year must be in : {}'.format(YEARS))

  data_dir = FLAGS.data_dir
  years = ['VOC2007', 'VOC2012']
  if FLAGS.year != 'merged':
    years = [FLAGS.year]

  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  # label map path

  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  for year in years:
    logging.info('Reading from PASCAL %s dataset.', year)
    examples_path = os.path.join(data_dir, year, 'ImageSets', 'Main',
                                 'person_' + FLAGS.set + '.txt')
    annotations_dir = os.path.join(data_dir, year, FLAGS.annotations_dir)
    examples_list = dataset_util.read_examples_list(examples_path)
    for idx, example in enumerate(examples_list):
      if idx % 100 == 0:
        logging.info('On image %d of %d', idx, len(examples_list))
      path = os.path.join(annotations_dir, example + '.xml')
      with tf.gfile.GFile(path, 'r') as fid:
        xml_str = fid.read()
      xml = etree.fromstring(xml_str)
      data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

      # writer writing the VOC2007 example
      dict_to_tf_example(data, FLAGS.data_dir, label_map_dict,
          FLAGS.ignore_difficult_instances, 
          'JPEGImages',
          writer)

  writer.close()
  print("Num of people: ", people)


if __name__ == '__main__':
  # tf.app.run()
  main(None)
