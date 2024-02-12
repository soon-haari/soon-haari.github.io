FROM sagemath/sagemath:latest

USER root
RUN apt-get update && apt-get install socat -y
USER sage

WORKDIR /app
COPY ./task.sage /app
COPY ./flag.txt /app

EXPOSE 17776

CMD "socat \"TCP-LISTEN:17776,reuseaddr,fork\" \"EXEC:sage chal.sage,nofork,stderr\""