FROM odsai/ecup26-quality-baseline:1.0

RUN apt-get update && apt-get install -y --no-install-recommends python3-venv && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv --system-site-packages

RUN /opt/venv/bin/pip install --no-cache-dir lightgbm==4.5.0 scikit-learn==1.6.1

ENV PATH="/opt/venv/bin:$PATH"
