#!/bin/bash
# Creates "Start gait session" / "Stop gait session" icons on the Pi desktop.
R="$(cd "$(dirname "$0")/.." && pwd)"
chmod +x "$R/scripts/start_session.sh" "$R/scripts/stop_session.sh"
for n in start stop; do
  F="$HOME/Desktop/gait-$n.desktop"
  T=$([ $n = start ] && echo "Start gait session" || echo "Stop gait session")
  I=$([ $n = start ] && echo "media-playback-start" || echo "media-playback-stop")
  printf '[Desktop Entry]\nType=Application\nName=%s\nExec=%s/scripts/%s_session.sh\nIcon=%s\nTerminal=false\n' "$T" "$R" "$n" "$I" > "$F"
  chmod +x "$F"
  gio set "$F" metadata::trusted true 2>/dev/null
done
echo "icons created in $HOME/Desktop"
