import torch
from torch import nn
import logging
import subprocess


def run_cmd(cmd):
    """
    Run a shell command and capture its output.

    Args:
        cmd (str): The shell command to run.

    Returns:
        Tuple[int, bytes, bytes]: Return code, standard output, and standard error.
    """
    logging.info("Running SubProcess")
    proc = subprocess.Popen(cmd, shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    s_output, s_err = proc.communicate()
    s_return = proc.returncode

    return s_return, s_output, s_err


def prepare_dataset():
    """
    Prepare the dataset based on user input (download or use local).

    Returns:
        None
    """
    logging.info("Give Url or local path of the dataset")
    method = int(input("Press 1 to download, press 2 to use a local dataset: "))

    if method == 1:
        url = str(input())
        logging.info("Creating the dataset")
        create_dataset = f"python3 load_dataset.py -url {url} -m download"
    else:
        directory = str(input())
        logging.info("Checking the dataset")
        create_dataset = f"python3 load_dataset.py -directory {directory} -m local"

    ret, out, err = run_cmd(create_dataset)

    if ret == 0:
        logging.info("Dataset Created")
    else:
        logging.info("Dataset creation failed")
        logging.error(err.decode("UTF-8"))


if __name__ == "__main__":

    logging.info("Train Your Own Model or Predict the image")
    logging.info("Press 1 to train your Own Model, press 2 to predict the Image ")
    request = int(input())

    if request == 1:
        prepare_dataset()
    else:
        logging.info("Predicting the image")
        logging.info("Choosing the best Model")
