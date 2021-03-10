FROM nvcr.io/nvidia/pytorch:20.12-py3

WORKDIR /temp


RUN apt-get update
RUN apt-get install libgl1-mesa-glx -y
RUN python -m pip install -U pip
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install numpy==1.20.0
RUN pip install hdbscan --no-cache-dir --no-binary :all:

WORKDIR /app

CMD ["python", "main.py", "--cfg", "configs/vae.yaml"]
