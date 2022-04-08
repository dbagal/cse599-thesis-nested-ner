FROM nvcr.io/nvidia/pytorch:22.02-py3
RUN pip3 install transformers==4.16.2 && \
pip3 install beautifulsoup4==4.10.0 \
&& pip3 install lxml==4.7.1