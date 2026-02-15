FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN pip install poetry

COPY pyproject.toml poetry.lock* ./

ENV POETRY_VIRTUALENVS_CREATE=false

RUN poetry install

COPY . .
CMD ["bash"]