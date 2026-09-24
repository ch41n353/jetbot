# Rules for this repo

## Run artifacts go on the USB drive, never the SD card

The JetBot's previous SD card was destroyed by this repo's own logging. Telemetry,
camera frames and plan JSON were written to `local_nav/goals/` on the root
filesystem — 8,658 files / 712 MB in five days, much of it `fsync`'d per record.
The card's 4 MB allocation unit turned every 850-byte telemetry line into a 4 MB
erase cycle. It degraded to ~12 MB/s, started timing out (`mmcblk0: error -110`),
and ext4 remounted `/` read-only mid-run. See `git log` around 2026-09-18.

**Rule: nothing that grows with runtime may be written to the SD card.**

### Where things live

| what | where |
|---|---|
| OS, code, venvs, model weights | SD card (`/`) — read-mostly |
| run artifacts: `*.jsonl`, `*.jpg`, `*.json` plans, `*.log` | USB drive at `/mnt/robotlogs` |

`local_nav/goals` is a **symlink** to `/mnt/robotlogs/goals`. Every writer in
`local_nav/` defaults to `os.path.join(ROOT, 'local_nav/goals')`, so the symlink
redirects all of them with no code change. Keep it that way — do not "fix" a
path by pointing it back at a real directory under the repo.

### When adding a new writer

- Default its output under `local_nav/goals/`. Never under `/var/log`,
  `/tmp`, or anywhere else on `/`.
- **Do not `fsync` per record.** `output.flush()` is enough for telemetry and
  replay logs. Per-record `os.fsync()` is what killed the last card; existing
  offenders are `sol_search.py`, `search_graph.py` and `dashboard/server.py`.
- Do not open streaming logs with `buffering=1` at sensor rates. `battery.py`
  samples at ~5 Hz and now buffers with a timed flush (`LOG_BUFFER_BYTES`,
  `LOG_FLUSH_SECONDS`) — copy that pattern. A block buffer on its own is worse
  than it looks: 64 KB holds ~15 s of 5 Hz telemetry, and the seconds before a
  brownout are the part of a power log anyone reads back. Flush on a timer.
- Add a `.gitignore` entry for the new artifact type.

### If `/mnt/robotlogs` is not mounted

Fail loudly. Do **not** fall back to writing on the SD card — a silent fallback
is exactly how the last card filled up unnoticed. Preflight check:

    mountpoint -q /mnt/robotlogs || { echo "robot log volume not mounted"; exit 1; }

The drive is removable and mounted `nofail`, so boot will succeed without it.
That is deliberate: the robot should boot and report the problem, not brick.

### The log volume's mount options are deliberate

    LABEL=robotlogs /mnt/robotlogs ext4 defaults,nofail,noatime,commit=60 0 2

- `ext4`, not the vfat it shipped with — the robot gets hard power-cycled and
  FAT32 has no journal to recover from that.
- `nofail` — a missing drive must not block boot.
- `noatime` — no write amplification just for reading a log back.
- `commit=60` — flush the journal every 60 s instead of 5. Trades up to a
  minute of telemetry on a power cut for ~12x fewer journal writes. This is a
  log volume; that trade is correct here. Do not "fix" it.
- Mounted by `LABEL`, not `/dev/sda1` — USB device letters move.

The drive is a Verbatim thumb drive with no TRIM support, not an SSD. It has
less endurance than the SD card it replaced. It is here so a worn-out log
volume is an annoyance instead of an unbootable robot — not as a licence to
write carelessly.

### Health check

`/` going read-only is silent and looks like random permission errors. When
something fails inexplicably with I/O or permission errors, check this first:

    mount | grep -q ' / .*[(,]ro[,)]' && echo "ROOT FILESYSTEM IS READ-ONLY"

The kernel ring buffer is lost across a reboot, but ramoops survives — read
`/sys/fs/pstore/console-ramoops-0` for the dmesg from just before a reset.

### Keep the SD card under ~70% full

A full card has no free erase blocks for wear levelling, which is what wore the
last one out. `df -h /` before adding anything bulky.
