"""Convert a folder of images in the same directory to lmdb."""

from absl import app, flags

from datasets import lmdb_dataset

# see https://github.com/Fangyh09/Image2LMDB/blob/master/README.md for structure
flags.DEFINE_string("dpath", None, "Path to the folder of images.")
flags.DEFINE_string("split", "train", "Split of the dataset.")
flags.DEFINE_integer("num_workers", 1, "Number of workers.")

FLAGS = flags.FLAGS

def main(argv):
    lmdb_dataset.folder2lmdb(FLAGS.dpath, FLAGS.split, FLAGS.num_workers)
    
if __name__ == "__main__":
    app.run(main)