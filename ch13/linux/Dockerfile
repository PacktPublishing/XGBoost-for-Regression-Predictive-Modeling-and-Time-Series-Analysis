#inherit from python 3.9
FROM python:3.9-slim

RUN apt-get update \
&& apt-get install -y --no-install-recommends python3\
&& apt-get clean

RUN mkdir /rest_api
WORKDIR /rest_api
COPY . /rest_api/

COPY ./requirements.txt /rest_api/requirements.txt

RUN python3 -m pip --no-cache-dir install wheel
RUN python3 -m pip --no-cache-dir install -r requirements.txt

CMD ["gunicorn", "--bind", ":8000", "--workers", "3" "rest_api.wsgi"]