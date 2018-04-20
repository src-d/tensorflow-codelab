FROM ubuntu:latest

RUN apt-get update && \
    apt-get install -y --no-install-suggests --no-install-recommends ca-certificates curl locales git libc6-dev gcc python3 && \
    curl https://bootstrap.pypa.io/get-pip.py | python3 && \
    pip3 install --no-cache-dir 'tensorflow>=1.7' tensorflow_hub keras 'h5py>=2.8.0rc1' jupyter && \
    curl -o /usr/local/bin/hub2graph https://raw.githubusercontent.com/vmarkovtsev/hub/master/examples/hub2graph.py && \
    chmod +x /usr/local/bin/hub2graph && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

ENV PATH="/root/go/bin:/usr/local/go/bin:${PATH}"

RUN curl https://dl.google.com/go/go1.10.1.linux-amd64.tar.gz | tar -C /usr/local -xz && \
    curl -L "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.6.0.tar.gz" | tar -C /usr/local -xz && \
    ldconfig && \
    go get -v github.com/tensorflow/tensorflow/tensorflow/go
