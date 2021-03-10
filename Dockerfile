FROM nvcr.io/nvidia/pytorch:20.12-py3

WORKDIR /app


RUN apt-get update
RUN apt-get install libgl1-mesa-glx -y
RUN python -m pip install -U pip
RUN pip install numpy h5py scikit-learn scikit-image matplotlib imageio torch torchvision torchaudio tqdm opencv-python numpyencoder tensorboardX yacs pandas
RUN pip install hdbscan --no-cache-dir --no-binary :all:

CMD ["python", "main.py", "--cfg", "configs/vae.yaml"]
