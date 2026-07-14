FROM swift:6.1-noble@sha256:e1cdaf7ddc9de37d8561da7a260535236694fca8c1b67d3129d47d8b180a9394

ARG DEBIAN_FRONTEND=noninteractive
ARG DOTNET_VERSION=8.0.412
ARG KOTLIN_VERSION=2.1.21
ARG TYPESCRIPT_VERSION=5.8.3

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates=20260601~24.04.1 \
        clang=1:18.0-59~exp2 \
        cargo=1.75.0+dfsg0ubuntu1-0ubuntu7.4 \
        curl=8.5.0-2ubuntu10.11 \
        g++=4:13.2.0-7ubuntu1 \
        golang-go=2:1.22~2build1 \
        openjdk-21-jdk-headless=21.0.11+10-1~24.04.2 \
        nodejs=18.19.1+dfsg-6ubuntu5 \
        npm=9.2.0~ds1-2 \
        php-cli=2:8.3+93ubuntu2 \
        python3=3.12.3-0ubuntu2.1 \
        ruby-full=1:3.2~ubuntu1 \
        rustc=1.75.0+dfsg0ubuntu1-0ubuntu7.4 \
        unzip=6.0-28ubuntu4.1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl --retry 5 --retry-all-errors -fsSL https://dot.net/v1/dotnet-install.sh -o /tmp/dotnet-install.sh \
    && bash /tmp/dotnet-install.sh --version "${DOTNET_VERSION}" --install-dir /opt/dotnet \
    && ln -s /opt/dotnet/dotnet /usr/local/bin/dotnet \
    && rm /tmp/dotnet-install.sh

RUN curl --retry 5 --retry-all-errors -fsSL \
        "https://github.com/JetBrains/kotlin/releases/download/v${KOTLIN_VERSION}/kotlin-compiler-${KOTLIN_VERSION}.zip" \
        -o /tmp/kotlin.zip \
    && unzip -q /tmp/kotlin.zip -d /opt \
    && ln -s /opt/kotlinc/bin/kotlinc /usr/local/bin/kotlinc \
    && rm /tmp/kotlin.zip

RUN npm install --global "typescript@${TYPESCRIPT_VERSION}" \
    && ln -s /usr/bin/python3 /usr/local/bin/python

ENV DOTNET_ROOT=/opt/dotnet
ENV PATH="/opt/dotnet:/opt/kotlinc/bin:${PATH}"

RUN python --version \
    && node --version \
    && tsc --version \
    && go version \
    && rustc --version \
    && javac -version \
    && clang++ --version \
    && dotnet --version \
    && ruby --version \
    && php --version \
    && swiftc --version \
    && kotlinc -version

WORKDIR /workspace
