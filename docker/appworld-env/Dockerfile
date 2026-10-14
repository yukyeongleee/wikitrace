# The AppWorld episode image: the simulator and our environment server, nothing of the host.
# scripts/setup/appworld_env_image.sh stages the install's own appworld package,
# its frozen dependency set and the server files into the build context.
FROM python:3.11.15-slim

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --no-deps -r /tmp/requirements.txt && rm /tmp/requirements.txt
COPY site/ /usr/local/lib/python3.11/site-packages/
COPY srv/ /srv/

# The episode runs as the host user, who has no home here; IPython keeps its profile under HOME.
ENV HOME=/tmp PYTHONDONTWRITEBYTECODE=1
WORKDIR /appworld
