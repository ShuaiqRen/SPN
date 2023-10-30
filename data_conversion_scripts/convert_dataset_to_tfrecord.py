# coding=utf-8
# Copyright 2023 .
# Author : Shuaiqi Ren.
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

import os
 
from absl import app
from absl import flags
import imageio
import tensorflow as tf

from spn.data_conversion_scripts import conversion_utils


FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/data/SPN/spn_dataset', 'Dataset folder.') 
flags.DEFINE_string('output_dir', '/data/SPN/spn_datasets', 'Location to export to.')
flags.DEFINE_string('train_split_file', 'spn/files/train_eval.txt', 'location of the datasets_train_eval.txt')
flags.DEFINE_integer('shard', 0, 'which shard this is')
flags.DEFINE_integer('num_shards', 1, 'How many total shards there are')


def convert_dataset():
    """Convert the data to the TFRecord format"""

    # make s directory to save the tfrecords to
    if not tf.io.gfile.exists(FLAGS.output_dir):
        tf.io.gfile.mkdir(FLAGS.output_dir)
    
    # create the directory for train ane test dataset
    train_dir = os.path.join(FLAGS.output_dir, 'train')
    test_dir = os.path.join(FLAGS.output_dir, 'test')
    if not tf.io.gfile.exists(train_dir):
        tf.io.gfile.mkdir(train_dir)
    if not tf.io.gfile.exists(test_dir):
        tf.io.gfile.mkdir(test_dir)
    
    # load the path set
    inputs_list = sorted(tf.io.gfile.glob(FLAGS.data_dir + '/*NxNy_log.txt'), key=lambda x:int(x.split('_')[2]))
    outputs_list = sorted(tf.io.gfile.glob(FLAGS.data_dir + '/*1D_Nx_outputlog.txt'), key=lambda x:int(x.split('_')[2]))
    assert len(inputs_list) == len(outputs_list)

    # Reading the txt file can fial on network filesystem, so copy to tmpdir first.
    tmpdir = '/tmp/spn_dataset'
    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)

    train_filenames = conversion_utils.generate_sharded_filenames(
        os.path.join(train_dir, 'spn@{}'.format(FLAGS.num_shards)))
    test_filenames = conversion_utils.generate_sharded_filenames(
        os.path.join(test_dir, 'spn@{}'.format(FLAGS.num_shards)))
    train_record_writer = tf.io.TFRecordWriter(train_filenames[FLAGS.shard])
    test_record_writer = tf.io.TFRecordWriter(test_filenames[FLAGS.shard])

    total = len(inputs_list)
    input_per_shard = total // FLAGS.num_shards
    start = input_per_shard * FLAGS.shard
    filepath = FLAGS.train_split_file
    with open(filepath, mode='r') as f:
        train_val = f.readlines()
        train_val = [int(x.strip()) for x in train_val]
    if FLAGS.shard == FLAGS.num_shards - 1:
        end = len(inputs_list)
    else:
        end = start + input_per_shard
    assert len(train_val) == len(inputs_list)
    assert len(outputs_list) == len(train_val)
    inputs_list = inputs_list[start:end]
    outputs_list = outputs_list[start:end]
    train_val = train_val[start:end]

    tf.compat.v1.logging.info('Writing %d images per shrd', input_per_shard)
    tf.compat.v1.logging.info('Writing range %d to %d of %d total', start, end, total)

    input_path = os.path.join(tmpdir, 'input.txt')
    output_path = os.path.join(tmpdir, 'output.txt')

    for i ,(input, output, assignment) in enumerate(zip(inputs_list, outputs_list, train_val)):
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)
        
        tf.io.gfile.copy(input, input_path)
        tf.io.gfile.copy(output, output_path)

        input_data = conversion_utils.read_input(input_path, FLAGS.dim_x, FLAGS.dim_y)
        output_data = conversion_utils.read_output(output_path, FLAGS.dim_x)

        assert input_data.shape[1] == output_data.shape[1]

        feature = {
            'input_shape_0': conversion_utils.int64_feature(input_data.shape[0]),
            'input_shape_1': conversion_utils.int64_feature(input_data.shape[1]),
            'output_shape_0': conversion_utils.int64_feature(output_data.shape[0]),
            'output_shape_1': conversion_utils.int64_feature(output_data.shape[1]),
            'input_path': conversion_utils.bytes_feature(str.encode(input)),
            'output_path': conversion_utils.bytes_feature(str.encode(output)),
            'input_data': conversion_utils.bytes_feature(input_data.tobytes()),
            'output_data': conversion_utils.bytes_feature(output_data.tobytes()),
        }
        example = tf.train.SequenceExample(
            context=tf.train.Features(feature=feature)
        )

        if i % 10 == 0:
            tf.compat.v1.logging.info('Writing %d out of %d total', i, len(inputs_list))

        if assignment == 1:
            train_record_writer.write(example.SerializeToString())
        elif assignment == 2:
            test_record_writer.write(example.SerializeToString())
        else:
            assert False, 'There is an error in the train_eval.txt'
    train_record_writer.close()
    test_record_writer.close()
    tf.compat.v1.logging.info('Saved result to %s', FLAGS.output_dir)


def main(_): 
    convert_dataset()


if __name__ == '__main__':
    print('Let start the test')
    app.run(main)


    