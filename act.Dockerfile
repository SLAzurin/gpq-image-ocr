FROM mcr.microsoft.com/powershell:latest
RUN apt-get update \
    && TZ=America/New_York DEBIAN_FRONTEND=noninteractive \
    apt-get install --no-install-suggests --no-install-recommends -y \
    git python3-full python3-dev python3-pip python-is-python3 binutils curl \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get update && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*