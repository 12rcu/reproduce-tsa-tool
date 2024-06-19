FROM python:3.7
WORKDIR /app
COPY . .
WORKDIR /app/TSA
RUN pip install docker \
scapy \
numpy \
scipy \
matplotlib \
scikit-learn \
lxml \
beautifulsoup4

RUN pip install . --user

WORKDIR /app/TSA/examples
ENTRYPOINT ["python", "-m", "run_stac_examples"]