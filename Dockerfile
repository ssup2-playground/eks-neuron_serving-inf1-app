FROM ubuntu:22.04
WORKDIR /app

# Install ubuntu packages and python
RUN apt-get update -y \
 && apt-get install -y --no-install-recommends \
    gnupg2 \
    wget \
    python3-pip \
    python3-setuptools \
    && cd /usr/local/bin \
    && pip3 --no-cache-dir install --upgrade pip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install neuron tools
RUN echo "deb https://apt.repos.neuron.amazonaws.com bionic main" > /etc/apt/sources.list.d/neuron.list
RUN wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | apt-key add -
RUN apt-get update -y && apt-get install -y \
    aws-neuronx-tools
ENV PATH="/opt/bin/:/opt/aws/neuron/bin:${PATH}"

# Install python packages
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --extra-index-url https://pip.repos.neuron.amazonaws.com

# Copy application files
COPY models/resnet50_neuron.pt models/resnet50_neuron.pt
COPY indexes/imagenet_index_name.txt indexes/imagenet_index_name.txt
COPY run.py run.py

CMD ["uvicorn", "run:app", "--host", "0.0.0.0", "--port", "80"]
