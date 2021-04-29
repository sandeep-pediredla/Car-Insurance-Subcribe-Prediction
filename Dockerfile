FROM python:3
RUN apt-get update -y && apt-get install -y build-essential


COPY ./requirements.txt /app/requirements.txt

WORKDIR /app
RUN mkdir models
RUN mkdir dataset
COPY dataset /app/dataset
COPY src /app

RUN ls -al
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["app.py"]