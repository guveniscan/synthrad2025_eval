# The organisers' evaluation container, rebuilt: python:3.11-slim, CPU only.
#   docker build -t synthrad-eval .
#   docker run --rm -v /data:/data synthrad-eval python -m synthrad_eval.evaluate score --manifest ... 
FROM python:3.11-slim
WORKDIR /opt/synthrad_eval
COPY requirements.txt requirements-nodeps.txt ./
COPY scripts scripts
RUN PIP_NO_CACHE_DIR=1 scripts/setup_env.sh /opt/ts_results
ENV SYNTHRAD_EVAL_TS_RESULTS=/opt/ts_results MPLBACKEND=Agg
COPY synthrad_eval synthrad_eval
