#!/usr/bin/env bash
# run like `./send.sh` on local, to prepare remote to continue setup


PORTS=$(runpodctl get pod -a | tail -n 1 | awk -F\t '{print $12}')
echo "Ports: $PORTS"
SSH_LINE=$(echo "$PORTS" | tr ',' '\n' | fgrep "pub" | fgrep "22")
echo "SSH line: $SSH_LINE"
IP_ADDR=$(echo $SSH_LINE | cut -d: -f1)
echo "IP_addr: $IP_ADDR"
PORT=$(echo $SSH_LINE | cut -d: -f2 | cut -d- -f1)
echo "Port: $PORT"
REMOTE="mlx"

# Remove old mlx entry
awk -v host="$REMOTE" '
BEGIN { in_block = 0 }
/^Host / {
    if (in_block && $2 != host) {
        in_block = 0
        print
    } else if ($2 == host) {
        in_block = 1
    } else {
        print
    }
    next
}
/^[[:space:]]/ && in_block { next }
/^$/ && in_block { next }
!in_block { print }
' ~/.ssh/config > ~/.ssh/config.tmp && mv ~/.ssh/config.tmp ~/.ssh/config

# Add new entry
cat >> ~/.ssh/config << EOF
Host $REMOTE
    HostName $IP_ADDR
    IdentityFile ~/.ssh/id_ed25519
    Port $PORT
    User root

EOF

# Add the host key to known_hosts to avoid interactive prompts
echo "Adding host key to known_hosts..."
ssh-keyscan -p $PORT $IP_ADDR >> ~/.ssh/known_hosts

scp init_pod.sh "$REMOTE:init_pod.sh"
scp .env "$REMOTE:.env"
scp .tmux.conf "$REMOTE:.tmux.conf"

# Execute ssh.sh on the remote server
echo "Executing setup script on remote server..."
ssh "$REMOTE" "chmod +x init_pod.sh && ./init_pod.sh"

./copyfiles.sh
