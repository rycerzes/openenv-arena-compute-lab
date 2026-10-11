# SPDX-License-Identifier: BSD-3-Clause
#
# Build with scripts/build_image.sh, which first installs the locked linux/amd64
# dependencies into build/site-packages. The image build itself needs no package index.

# python:3.12-slim (3.12.15, Debian trixie, 2026-10-09), pinned by index digest
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

LABEL org.opencontainers.image.title="Verified Compute Lab (Workbench v2)" \
      org.opencontainers.image.description="OpenEnv environment: 14 procedural tool-use task families across eight domains" \
      org.opencontainers.image.source="https://github.com/rycerzes/openenv-arena-compute-lab" \
      org.opencontainers.image.licenses="BSD-3-Clause"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/deps:/opt/app

COPY build/site-packages /opt/deps
COPY __init__.py models.py client.py openenv.yaml /opt/app/
COPY server /opt/app/server

# The environment source is root-only: the python tool runs as nobody.
# Precompile bytecode so a cold start does not compile every dependency.
RUN python -m compileall -q -j 0 /opt/deps /opt/app && chmod -R go-rwx /opt/app && cd /opt/app && python -c "import server.app"

WORKDIR /opt/app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
