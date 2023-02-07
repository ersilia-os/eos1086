FROM bentoml/model-server:0.11.0-py37
MAINTAINER ersilia

RUN conda install -c conda-forge 
RUN pip install joblib==1.1.0
RUN pip install transformers
RUN pip install upsetplot==0.6.0 
RUN pip install pandas==1.1.5 matplotlib==3.3.4 pubchempy==1.0.4
RUN pip install transformers
RUN pip install torch
RUN pip install bs4
RUN pip install tqdm
RUN pip install numpy



WORKDIR /repo
COPY . /repo