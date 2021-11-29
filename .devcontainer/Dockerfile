
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

ARG USERNAME=cisl

USER root
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        sudo \
        zsh && \
    apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
        autopep8 \
        pylint \
        pytest
RUN useradd \
        --shell $(which zsh) \
        -G sudo \
        -m -d /home/${USERNAME} -k /etc/skel \
        ${USERNAME} && \
    sed -i -e 's/%sudo.*/%sudo\tALL=(ALL:ALL)\tNOPASSWD:ALL/g' /etc/sudoers
USER ${USERNAME}
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" && \
    sed -i -e 's/ZSH_THEME=.*/ZSH_THEME="simple"/g' ~/.zshrc
