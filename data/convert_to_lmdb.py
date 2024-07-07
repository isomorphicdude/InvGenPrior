"""Convert a folder of images in the same directory to lmdb."""

from absl import app, flags
import sys
import os


script_dir = os.path.dirname(__file__)
project_dir = os.path.join(script_dir, '..')
sys.path.append(project_dir)

from datasets import lmdb_dataset

# see https://github.com/Fangyh09/Image2LMDB/blob/master/README.md for structure
flags.DEFINE_string("dpath", None, "Path to the folder of images.")
flags.DEFINE_string("split", "train", "Split of the dataset.")
flags.DEFINE_integer("num_workers", 1, "Number of workers.")

flags.mark_flags_as_required(["dpath"])

FLAGS = flags.FLAGS

def main(argv):
    lmdb_dataset.folder2lmdb(FLAGS.dpath, FLAGS.split, FLAGS.num_workers)
    
if __name__ == "__main__":
    app.run(main)