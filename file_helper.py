import codecs
import os
import errno
import shutil


def mkdirs(file_name):
    if not os.path.exists(os.path.dirname(file_name)):
        try:
            os.makedirs(os.path.dirname(file_name))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def dir_exists(dir_name):
    return os.path.exists(os.path.dirname(dir_name))


def delete_dir(dir_name):
    shutil.rmtree(dir_name)


def open_file_for_writing(file_name, encoding=None, mode='w', buffering=1):
    mkdirs(file_name)
    return codecs.open(file_name, mode, encoding, buffering=buffering)


def open_file_for_reading(file_name, encoding=None, mode='rb'):
    return codecs.open(file_name, mode, encoding)


def silent_remove(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise
