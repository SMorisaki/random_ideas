FROM node:20

ARG TZ
ENV TZ="$TZ"

ARG CLAUDE_CODE_VERSION=latest

# Install basic development tools and iptables/ipset
RUN apt-get update && apt-get install -y --no-install-recommends \
  less \
  git \
  procps \
  sudo \
  fzf \
  zsh \
  man-db \
  unzip \
  gnupg2 \
  gh \
  iptables \
  ipset \
  iproute2 \
  dnsutils \
  aggregate \
  jq \
  nano \
  vim \
  tmux \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Ensure default node user has access to /usr/local/share
RUN mkdir -p /usr/local/share/npm-global && \
  chown -R node:node /usr/local/share

ARG USERNAME=node

# Persist bash history.
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
  && mkdir /commandhistory \
  && touch /commandhistory/.bash_history \
  && chown -R $USERNAME /commandhistory

# Set `DEVCONTAINER` environment variable to help with orientation
ENV DEVCONTAINER=true

# Create workspace and config directories and set permissions
# (.codex holds Codex's auth.json / config.toml; mounted as a volume)
RUN mkdir -p /workspace /home/node/.claude /home/node/.codex && \
  chown -R node:node /workspace /home/node/.claude /home/node/.codex

WORKDIR /workspace

ARG GIT_DELTA_VERSION=0.18.2
RUN ARCH=$(dpkg --print-architecture) && \
  wget "https://github.com/dandavison/delta/releases/download/${GIT_DELTA_VERSION}/git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
  sudo dpkg -i "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
  rm "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb"

# Set up non-root user
USER node

# Install global packages
ENV NPM_CONFIG_PREFIX=/usr/local/share/npm-global
ENV PATH=$PATH:/usr/local/share/npm-global/bin

# Set the default shell to zsh rather than sh
ENV SHELL=/bin/zsh

# Set the default editor and visual
ENV EDITOR=nano
ENV VISUAL=nano

# Default powerline10k theme
ARG ZSH_IN_DOCKER_VERSION=1.2.0
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v${ZSH_IN_DOCKER_VERSION}/zsh-in-docker.sh)" -- \
  -p git \
  -p fzf \
  -a "source /usr/share/doc/fzf/examples/key-bindings.zsh" \
  -a "source /usr/share/doc/fzf/examples/completion.zsh" \
  -a "export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
  -x

# Install Claude
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}

ARG CODEX_VERSION=latest
RUN npm install -g @openai/codex@${CODEX_VERSION}

# Copy and set up firewall script
COPY init-firewall.sh /usr/local/bin/
USER root
RUN chmod +x /usr/local/bin/init-firewall.sh && \
  echo "node ALL=(root) NOPASSWD: /usr/local/bin/init-firewall.sh" > /etc/sudoers.d/node-firewall && \
  chmod 0440 /etc/sudoers.d/node-firewall
USER node

USER root
# Install Miniforge for Linux ARM64/AMD64
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl bzip2 && \
    rm -rf /var/lib/apt/lists/*

RUN ARCH="$(uname -m)" && \
    case "${ARCH}" in \
        aarch64) MINIFORGE_ARCH="aarch64" ;; \
        x86_64)  MINIFORGE_ARCH="x86_64" ;; \
        *) echo "Unsupported architecture: ${ARCH}" && exit 1 ;; \
    esac && \
    curl -L \
      "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${MINIFORGE_ARCH}.sh" \
      -o /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh && \
    chown -R node:node /opt/conda

# The project env (/workspace/.conda-linux) is the default Python for EVERY process,
# including the non-interactive shells Claude Code and Codex use to run commands:
#  - its bin/ comes first in PATH, so `python`, `pip`, `pytest` resolve into it
#  - CONDA_PREFIX makes `conda install` / `mamba install` target it without `-p`
# /opt/conda/bin (base) stays on PATH so `conda` / `mamba` themselves are found.
ENV CONDA_PREFIX=/workspace/.conda-linux \
    CONDA_DEFAULT_ENV=/workspace/.conda-linux \
    PATH="/workspace/.conda-linux/bin:/opt/conda/bin:${PATH}"

# Make `conda activate` available in bash and zsh (interactive shells)
RUN for rc in /home/node/.bashrc /home/node/.zshrc; do \
      echo 'source /opt/conda/etc/profile.d/conda.sh' >> "$rc" && \
      echo 'if [ -d /workspace/.conda-linux ]; then' >> "$rc" && \
      echo '  conda activate /workspace/.conda-linux' >> "$rc" && \
      echo 'fi' >> "$rc"; \
    done && \
    chown node:node /home/node/.bashrc /home/node/.zshrc
USER node
