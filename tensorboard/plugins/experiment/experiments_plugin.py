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
"""The TensorBoard Experiments plugin.

See `http_api.md` in this directory for specifications of the routes for
this plugin.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv

import six
from six import StringIO
from werkzeug import wrappers

import numpy as np
import tensorflow as tf
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.plugins import base_plugin


class ExperimentsPlugin(base_plugin.TBPlugin):
  """Experiments Plugin for TensorBoard."""

  plugin_name = 'experiments'

  def __init__(self, context):
    """Instantiates ExperimentsPlugin via TensorBoard core.

    Args:
      context: A base_plugin.TBContext instance.
    """
    self._db_connection_provider = context.db_connection_provider

  def get_plugin_apps(self):
    return {
        '/experiments': self.experiments_route,
    }

  def is_active(self):
    """The experiments plugin is active iff we detect more than one experiment."""
    if not self._db_connection_provider:
      return False
    # The plugin is active if 2+ experiments are found in the database.
    db = self._db_connection_provider()
    cursor = db.execute('''
      SELECT
        experiment_id
      FROM Experiments
      LIMIT 2
    ''')
    return len(list(cursor)) >= 2

  def experiments_impl(self):
    """Returns the body as a JSON-serializable python structure."""
    if not self._db_connection_provider:
      return []
    return [{'name': 'alpha'}, {'name': 'beta'}]
    """
    db = self._db_connection_provider()
    # We select for steps greater than -1 because the writer inserts
    # placeholder rows en masse. The check for step filters out those rows.
    cursor = db.execute('''
      SELECT
        Tensors.step,
        Tensors.computed_time,
        Tensors.data,
        Tensors.dtype
      FROM Tensors
      JOIN Tags
        ON Tensors.series = Tags.tag_id
      JOIN Runs
        ON Tags.run_id = Runs.run_id
      WHERE
        Runs.run_name = ?
        AND Tags.tag_name = ?
        AND Tags.plugin_name = ?
        AND Tensors.shape = ''
        AND Tensors.step > -1
      ORDER BY Tensors.step
    ''', (run, tag, metadata.PLUGIN_NAME))
    values = [(wall_time, step, self._get_value(data, dtype_enum))
              for (step, wall_time, data, dtype_enum) in cursor]
    """

  @wrappers.Request.application
  def experiments_route(self, request):
    """Return array of experiment data."""
    # TODO: return HTTP status code for malformed requests
    # tag = request.args.get('tag')
    # run = request.args.get('run')
    body = self.experiments_impl()
    return http_util.Respond(request, body, 'application/json')
