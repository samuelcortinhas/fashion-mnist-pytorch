FROM python:3.9.16
# FROM --platform=linux/amd64 python:3.9.16

WORKDIR /usr/src/app

COPY ./requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

COPY . .

RUN /usr/sbin/useradd -m docker \
    && chown -R docker:docker /usr/src/app

USER docker

CMD ["uvicorn", "app.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8080"]
