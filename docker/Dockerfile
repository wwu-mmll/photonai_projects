FROM python:3.9.18-bookworm

RUN pip install photonai

COPY ../requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt

COPY ../photonai_projects/cli.py /src/main.py

ENTRYPOINT ["python", "/src/main.py"]
CMD ["--help"]
