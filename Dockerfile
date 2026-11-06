# CPU verification image using the recorded Python package versions.
# The base image and operating-system packages are not byte-pinned.
#
#   make docker-verify                      # make verify test inside the image
#   make docker-review REPORT_ROOT=../r     # make review REVIEW_OFFLINE=1, report under ../r/review
#   docker run --rm -it cross-morse:verify make test PYTHON=python
#
# The image installs the recorded CPU environment (environment/requirements-cpu.lock) on the
# Python 3.12 that the lock was observed with. It contains no benchmark data and no official run
# archive; mount a deposits directory and run `make fetch-deposits` to examine official evidence.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends make git ca-certificates libcairo2 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/cross-morse

# Dependencies first so that source edits do not invalidate the environment layer.
COPY environment/ environment/
COPY requirements.txt requirements-review.txt pyproject.toml ./
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      "$(python -B environment/check_environment.py --bootstrap)" \
 && python -m pip install --no-cache-dir -r environment/requirements-cpu.lock \
 && python -m pip check

COPY . .
RUN python -B environment/check_environment.py

CMD ["make", "verify", "PYTHON=python"]
