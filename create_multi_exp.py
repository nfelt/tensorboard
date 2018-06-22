#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf

def build_graph(db_path, exp_name, run_name, value):
  tf.reset_default_graph()
  writer = tf.contrib.summary.create_db_writer(
      db_path,
      experiment_name=exp_name,
      run_name=run_name,
      user_name='jeff')
  global_step = tf.get_variable(
      'global_step',
      shape=[],
      dtype=tf.int64,
      initializer=tf.ones_initializer,
      trainable=False,
      caching_device='/cpu:0',
      use_resource=True)
  train_op = global_step.assign_add(1, read_value=False)
  with writer.as_default(), tf.contrib.summary.always_record_summaries():
    summary_op = tf.contrib.summary.scalar('loss', value, step=global_step)
    flush_op = tf.contrib.summary.flush(writer._resource)
  return summary_op, train_op, flush_op

def run_experiment(db_path, exp_name, value):
  for run_name in ('train', 'eval'):
    summary_op, train_op, flush_op = build_graph(db_path, exp_name, run_name, value)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.contrib.summary.summary_writer_initializer_op())
    for i in range(100):
      sess.run([summary_op, train_op])
    sess.run(flush_op)

def main():
  args = sys.argv[1:]
  db_path = args[0] if args else '/tmp/tb_multiexp.sqlite'
  run_experiment(db_path, 'alpha', 1.0)
  run_experiment(db_path, 'beta', 2.0)
  run_experiment(db_path, 'ga', 3.0)

if __name__ == '__main__':
  main()
