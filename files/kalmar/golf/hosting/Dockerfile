FROM sagemath/sagemath:latest@sha256:7f553b1e4f2ea9f3ffff4b6eff33bda9f435d61844523f61d49d2cab29dd75d6

USER root

# We extract the correct directory names on two levels for the Python installation that comes
# with SageMath.
# There will other dirs in this directory like "installed", but there should be exactly one
# containing venv-python, and which also contains a version.
# Due to the version we can't hardcode it.
RUN SAGE_PYTHON_VENV_PATH=$(ls /home/sage/sage/local/var/lib/sage/ | grep venv-python) && \
    echo "Using SageMath Python virtual environment dir name: $SAGE_PYTHON_VENV_PATH" && \
    ln -s "/home/sage/sage/local/var/lib/sage/${SAGE_PYTHON_VENV_PATH}/bin/" /home/sage/bin && \
    SAGE_PYTHON_VENV_SITE_PACKAGES_VERSION=$(ls "/home/sage/sage/local/var/lib/sage/${SAGE_PYTHON_VENV_PATH}/lib/" | grep python | grep -v libpython) && \
    echo "The dir for the site-packages of SageMath's Python venv is: ${SAGE_PYTHON_VENV_SITE_PACKAGES_VERSION}" && \
    ln -s "/home/sage/sage/local/var/lib/sage/${SAGE_PYTHON_VENV_PATH}/lib/${SAGE_PYTHON_VENV_SITE_PACKAGES_VERSION}/site-packages" /home/sage/site-packages

# We include the sage bin directory for Python with highest priority.
# This is so that all our Python setup uses the same version, the sage one. 
ENV PATH="/home/sage/bin/:${PATH}"


RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade pwntools && \
    python3 -m pip install --upgrade pycryptodome

RUN apt-get -y update && apt-get -y install \
  socat \
  && apt-get clean


RUN mkdir /app
COPY chal-warmup.py /app/
COPY lib.py /app/
COPY flag.txt /app/
COPY requirements.txt /app/

RUN pip install -r /app/requirements.txt

WORKDIR /app
USER sage
EXPOSE 1337
ENTRYPOINT []
CMD socat tcp-l:3470,reuseaddr,fork exec:/app/chal-warmup.py,pty,echo=0,raw,iexten=0
