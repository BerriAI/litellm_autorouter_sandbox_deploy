ARG BASE_IMAGE=ghcr.io/berriai/litellm:main-latest
FROM ${BASE_IMAGE}

ARG SITE_PACKAGES=/app/.venv/lib/python3.13/site-packages
COPY litellm/ ${SITE_PACKAGES}/litellm/
COPY enterprise/litellm_enterprise/ ${SITE_PACKAGES}/litellm_enterprise/
COPY enterprise/ /app/enterprise/
COPY litellm-proxy-extras/litellm_proxy_extras/ ${SITE_PACKAGES}/litellm_proxy_extras/
COPY litellm-proxy-extras/ /app/litellm-proxy-extras/
COPY schema.prisma /app/schema.prisma
COPY deploy/autorouter-sandbox/proxy_config.yaml /app/config.yaml
# Every module the config resolves by dotted path must sit next to config.yaml, since
# get_instance_fn looks relative to the config file's directory. Copying the directory
# wholesale rather than naming files individually: the previous per-file list silently went
# stale every time an experiment added a module, and the failure only shows up at deploy time
# as "Could not import X from X", never locally (where the repo dir already has everything).
COPY deploy/autorouter-sandbox/*.py /app/

CMD ["--config", "/app/config.yaml", "--port", "4000"]
