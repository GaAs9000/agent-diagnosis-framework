FROM zli12321/lhtb-sokoban@sha256:11ae8cbb0a45d8cad145278caf1dc6cde99dd6bc7479dd486696759e80d1c950
LABEL org.l1.compatibility="sokoban-public-cli-history-replay-v1" \
      org.l1.native-base.digest="sha256:11ae8cbb0a45d8cad145278caf1dc6cde99dd6bc7479dd486696759e80d1c950" \
      org.l1.native-base.image-id="sha256:b2a9f13cab4b017a58e2f9dd9abd1db475f52f17bae290e3dc464dfeb0668fb0"
RUN mv /usr/local/bin/sokoban /usr/local/bin/sokoban-native
COPY sokoban_cold.py /usr/local/bin/sokoban
RUN chmod 755 /usr/local/bin/sokoban
ENV SOKOBAN_RUN_DIR=/workspace/runs \
    SOKOBAN_DAEMON_LOG=/workspace/runs/daemon.log \
    SOKOBAN_SOCKET=/tmp/sokoban.sock
