FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update 

RUN apt-get install -y python3-pip python3-dev git wget unzip poppler-utils python3-opencv \ 
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip


# Setup a spot for the api code
WORKDIR  /usr/local/src/vila

COPY requirements.txt .
# Install Python dependencies (vila is the sole dependency now)
RUN pip install --upgrade pip setuptools wheel
# For now, we need to install a version of vila from github directly  
RUN pip install "git+https://github.com/allenai/vila.git#egg=vila" 
# Install additional requirements for the api
RUN pip install --no-cache-dir -r requirements.txt

COPY . /usr/local/src/vila
EXPOSE 8080

ENTRYPOINT [ "uvicorn" ]
CMD ["vila_pdf_parsing_service:app", "--host", "0.0.0.0", "--reload", "--port", "8080"]
