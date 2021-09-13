FROM python:3.9

WORKDIR /app

RUN apt-get update

COPY requirements.txt .
RUN pip install -r requirements.txt 
RUN rm requirements.txt

COPY model/ ./model/
COPY utils.py ./utils.py
COPY app.py ./app.py

CMD [ "python", "app.py" ]
