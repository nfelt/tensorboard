# -*- coding: utf-8 -*-
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
"""Tests for the image plugin summary generation functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os

import numpy as np
import six
import tensorflow as tf

from tensorboard.plugins.image import metadata
from tensorboard.plugins.image import summary

try:
  from tensorboard.compat import tf_v2
except ImportError:
  tf_v2 = None

try:
  tf.enable_eager_execution()
except AttributeError:
  # TF 2.0 doesn't have this symbol because eager is the default.
  pass


class SummaryBaseTest(object):

  def setUp(self):
    super(SummaryBaseTest, self).setUp()
    np.random.seed(0)
    self.image_width = 20
    self.image_height = 15
    self.image_count = 1
    self.image_channels = 3

  def _generate_images(self, **kwargs):
    size = [
        kwargs.get('n', self.image_count),
        kwargs.get('h', self.image_height),
        kwargs.get('w', self.image_width),
        kwargs.get('c', self.image_channels),
    ]
    return np.random.uniform(low=0, high=255, size=size).astype(np.uint8)

  def image(self, *args, **kwargs):
    raise NotImplementedError()

  def test_tag(self):
    data = np.array(1, np.uint8, ndmin=4)
    self.assertEqual('a', self.image('a', data).value[0].tag)
    self.assertEqual('a/b', self.image('a/b', data).value[0].tag)

  def test_metadata(self):
    data = np.array(1, np.uint8, ndmin=4)
    pb = self.image('mona_lisa', data)
    summary_metadata = pb.value[0].metadata
    plugin_data = summary_metadata.plugin_data
    self.assertEqual(plugin_data.plugin_name, metadata.PLUGIN_NAME)
    content = summary_metadata.plugin_data.content
    # There's no content, so successfully parsing is fine.
    metadata.parse_plugin_metadata(content)

  def test_image_count_zero(self):
    shape = (0, self.image_height, self.image_width, 3)
    data = np.array([], np.uint8).reshape(shape)
    pb = self.image('mona_lisa', data, max_outputs=3)
    self.assertEqual(1, len(pb.value))
    result = pb.value[0].tensor.string_val
    self.assertEqual(tf.compat.as_bytes(str(self.image_width)), result[0])
    self.assertEqual(tf.compat.as_bytes(str(self.image_height)), result[1])
    self.assertEqual(2, len(result))

  def test_image_count_less_than_max_outputs(self):
    max_outputs = 3
    data = self._generate_images(n=(max_outputs - 1))
    pb = self.image('mona_lisa', data, max_outputs=max_outputs)
    self.assertEqual(1, len(pb.value))
    result = pb.value[0].tensor.string_val
    image_results = result[2:]  # skip width, height
    self.assertEqual(len(data), len(image_results))

  def test_image_count_more_than_max_outputs(self):
    max_outputs = 3
    data = self._generate_images(n=(max_outputs + 1))
    pb = self.image('mona_lisa', data, max_outputs=max_outputs)
    self.assertEqual(1, len(pb.value))
    result = pb.value[0].tensor.string_val
    image_results = result[2:]  # skip width, height
    self.assertEqual(max_outputs, len(image_results))

  def test_requires_nonnegative_max_outputs(self):
    data = np.array(1, np.uint8, ndmin=4)
    with six.assertRaisesRegex(
        self, (ValueError, tf.errors.InvalidArgumentError), '>= 0'):
      self.image('mona_lisa', data, max_outputs=-1)

  def test_floating_point_data(self):
    # include truncation of values outside [0, 1)
    pass  # DO NOT SUBMIT

  def test_png_format_roundtrip(self):
    images = self._generate_images(c=1)
    pb = self.image('mona_lisa', images)
    encoded = pb.value[0].tensor.string_val[2]  # skip width, height
    self.assertAllEqual(images[0], tf.image.decode_png(encoded))

  def _test_dimensions(self, images):
    pb = self.image('mona_lisa', images)
    self.assertEqual(1, len(pb.value))
    result = pb.value[0].tensor.string_val
    # Check annotated dimensions.
    self.assertEqual(tf.compat.as_bytes(str(self.image_width)), result[0])
    self.assertEqual(tf.compat.as_bytes(str(self.image_height)), result[1])
    for i, encoded in enumerate(result[2:]):
      decoded = tf.image.decode_png(encoded)
      self.assertEqual(images[i].shape, decoded.shape)

  def test_dimensions(self):
    self._test_dimensions(self._generate_images(c=1))
    self._test_dimensions(self._generate_images(c=2))
    self._test_dimensions(self._generate_images(c=3))
    self._test_dimensions(self._generate_images(c=4))

  def test_dimensions_when_not_statically_known(self):
    # only works w/ graph fn now?
    pass  # DO NOT SUBMIT

  def test_requires_rank_4(self):
    with six.assertRaisesRegex(self, ValueError, 'must have rank 4'):
      self.image('mona_lisa', [[[1], [2]], [[3], [4]]])


class SummaryV1PbTest(SummaryBaseTest, tf.test.TestCase):
  def image(self, *args, **kwargs):
    return summary.pb(*args, **kwargs)

  def test_tag(self):
    data = np.array(1, np.uint8, ndmin=4)
    self.assertEqual('a/image_summary', self.image('a', data).value[0].tag)
    self.assertEqual('a/b/image_summary', self.image('a/b', data).value[0].tag)


class SummaryV1OpTest(SummaryBaseTest, tf.test.TestCase):
  def image(self, *args, **kwargs):
    args = list(args)
    # Force first argument to tf.uint8 since the V1 version requires this.
    args[1] = tf.cast(tf.constant(args[1]), tf.uint8)
    return tf.Summary.FromString(summary.op(*args, **kwargs).numpy())

  def test_tag(self):
    data = np.array(1, np.uint8, ndmin=4)
    self.assertEqual('a/image_summary', self.image('a', data).value[0].tag)
    self.assertEqual('a/b/image_summary', self.image('a/b', data).value[0].tag)

  def test_scoped_tag(self):
    data = np.array(1, np.uint8, ndmin=4)
    with tf.name_scope('scope'):
      self.assertEqual('scope/a/image_summary',
                       self.image('a', data).value[0].tag)


class SummaryV2OpTest(SummaryBaseTest, tf.test.TestCase):
  def setUp(self):
    super(SummaryV2OpTest, self).setUp()
    if tf_v2 is None:
      self.skipTest('TF v2 summary API not available')

  def image(self, *args, **kwargs):
    kwargs.setdefault('step', 1)
    writer = tf_v2.summary.create_file_writer(self.get_temp_dir())
    with writer.as_default():
      summary.image(*args, **kwargs)
    writer.close()
    return self.read_single_event_from_eventfile().summary

  def read_single_event_from_eventfile(self):
    event_files = glob.glob(os.path.join(self.get_temp_dir(), '*'))
    self.assertEqual(len(event_files), 1)
    events = list(tf.compat.v1.train.summary_iterator(event_files[0]))
    # Expect a boilerplate event for the file_version, then the summary one.
    self.assertEqual(len(events), 2)
    return events[1]

  def test_scoped_tag(self):
    data = np.array(1, np.uint8, ndmin=4)
    with tf.name_scope('scope'):
      self.assertEqual('scope/a', self.image('a', data).value[0].tag)

  def test_step(self):
    data = np.array(1, np.uint8, ndmin=4)
    self.image('a', data, step=333)
    event = self.read_single_event_from_eventfile()
    self.assertEqual(333, event.step)


if __name__ == '__main__':
  tf.test.main()
