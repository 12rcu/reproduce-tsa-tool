FROM python:3.7
WORKDIR /app
COPY . .
WORKDIR /app/TSA
RUN pip install docker  \
scapy \
numpy \
scipy \
matplotlib \
scikit-learn

RUN pip install . --user

ENTRYPOINT ["python", "./examples/run_stac_examples.py"]