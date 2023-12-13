import subprocess
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import shutil
import os

WORKSPACE_PATH = './workspace'
SCRIPTS_PATH = 'workspace/scripts'
APIMODEL_PATH = WORKSPACE_PATH + '/models'
ANNOTATION_PATH = WORKSPACE_PATH + '/images/annotations'
IMAGE_PATH = WORKSPACE_PATH + '/images'
MODEL_PATH = WORKSPACE_PATH + '/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH + '/pre-trained-models'
CONFIG_PATH = MODEL_PATH + '/my_ssd_mobnet/pipeline.config'

labels = [{'name': 'a', 'id': 1}, {'name': 'b', 'id': 2}, {'name': 'c', 'id': 3}, {'name': 'd', 'id': 4}, {'name': 'e', 'id': 5}]

""" 1. Create Label Map """
with open(ANNOTATION_PATH + '/label_map.pbtxt', 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

""" 2. Create TF records """
train_command = [
    'python', f'{SCRIPTS_PATH}/generate_tfrecord.py',
    '-x', f'{IMAGE_PATH}/train',
    '-l', f'{ANNOTATION_PATH}/label_map.pbtxt',
    '-o', f'{ANNOTATION_PATH}/train.record'
]

test_command = [
    'python', f'{SCRIPTS_PATH}/generate_tfrecord.py',
    '-x', f'{IMAGE_PATH}/test',
    '-l', f'{ANNOTATION_PATH}/label_map.pbtxt',
    '-o', f'{ANNOTATION_PATH}/test.record'
]


subprocess.run(train_command, check=True)
subprocess.run(test_command, check=True)

""" Update Config For Transfer Learning """
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
# Create a directory and handle feedback
model_directory = os.path.join('workspace', 'models', CUSTOM_MODEL_NAME)

try:
    os.makedirs(model_directory, exist_ok=True)
    print(f"Directory '{CUSTOM_MODEL_NAME}' created successfully.")
except OSError as error:
    print(f"Directory '{CUSTOM_MODEL_NAME}' creation failed: {error}")

# Copy a file from one location to another and handle feedback
PRETRAINED_MODEL_PATH = './workspace/pre-trained-models'
MODEL_PATH = './workspace/models'

source_config_path = os.path.join(PRETRAINED_MODEL_PATH, 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', 'pipeline.config')
destination_config_path = os.path.join(MODEL_PATH, CUSTOM_MODEL_NAME, 'pipeline.config')

try:
    shutil.copyfile(source_config_path, destination_config_path)
    print("File copied successfully.")
except FileNotFoundError as error:
    print(f"File copy failed: {error}")
except shutil.SameFileError as error:
    print(f"File copy failed: Destination path '{destination_config_path}' already exists.")
except Exception as error:
    print(f"File copy failed: {error}")

    
    
config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

with tf.io.gfile.GFile(CONFIG_PATH, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

    pipeline_config.model.ssd.num_classes = 5
    pipeline_config.train_config.batch_size = 4
    pipeline_config.train_config.fine_tune_checkpoint = PRETRAINED_MODEL_PATH + '/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0'
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_input_reader.label_map_path = ANNOTATION_PATH + '/label_map.pbtxt'
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/train.record']
    pipeline_config.eval_input_reader[0].label_map_path = ANNOTATION_PATH + '/label_map.pbtxt'
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/test.record']

config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(CONFIG_PATH, "w") as f:  # Use "w" for writing mode
    f.write(config_text)  # Write the modified configuration back to the file

print("Execution completed successfully! Configuration updated.")

print("Command to train your model: ")
print("""python {}/research/object_detection/model_main_tf2.py --model_dir={}/{} --pipeline_config_path={}/{}/pipeline.config --num_train_steps=10000""".format(APIMODEL_PATH, MODEL_PATH,CUSTOM_MODEL_NAME,MODEL_PATH,CUSTOM_MODEL_NAME))