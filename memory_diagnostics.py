"""RunPod cgroup v2 RAM diagnostics for H3 generation jobs.

Measures the whole container (handler and ComfyUI), including file-backed cache.
The metrics are log-only and never alter the handler output.
"""
import functools
import os
import threading
import time

_CGROUP_DIR = "/sys/fs/cgroup"
_GIB = 1024 ** 3
_MONITORED_ACTIONS = frozenset({
    "generate",
    "h3_generate",
    "h3_t2v_generate",
    "h3_extend_generate",
    "h3_ref2va_generate",
})


def _read_int_file(name):
    try:
        with open(os.path.join(_CGROUP_DIR, name), encoding="ascii") as handle:
            value = handle.read().strip()
        return None if value == "max" else int(value)
    except (OSError, ValueError):
        return None


def _read_fields(name):
    fields = {}
    try:
        with open(os.path.join(_CGROUP_DIR, name), encoding="ascii") as handle:
            for line in handle:
                key, _, value = line.strip().partition(" ")
                if value:
                    fields[key] = int(value)
    except (OSError, ValueError):
        pass
    return fields


def _snapshot():
    current = _read_int_file("memory.current")
    if current is None:
        return None
    stat = _read_fields("memory.stat")
    events = _read_fields("memory.events")
    return {
        "current": current,
        "limit": _read_int_file("memory.max"),
        "highwater": _read_int_file("memory.peak"),
        "anon": stat.get("anon"),
        "file": stat.get("file"),
        "oom": events.get("oom"),
        "oom_kill": events.get("oom_kill"),
    }


def _gib(value):
    return "n/a" if value is None else f"{value / _GIB:.2f}"


def _log(action, phase, snapshot, sampled_peak, started, initial):
    if snapshot is None:
        print(f"[H3 RAM] action={action} phase={phase} cgroup_unavailable=true", flush=True)
        return
    start_oom = initial.get("oom") if initial else None
    start_kill = initial.get("oom_kill") if initial else None
    oom = snapshot.get("oom")
    killed = snapshot.get("oom_kill")
    oom_delta = oom - start_oom if oom is not None and start_oom is not None else None
    kill_delta = killed - start_kill if killed is not None and start_kill is not None else None
    # memory.peak includes prior jobs on the same worker; sampled_peak is per-job.
    print(
        f"[H3 RAM] action={action} phase={phase} elapsed_s={time.monotonic()-started:.1f} "
        f"current_gib={_gib(snapshot['current'])} "
        f"sampled_peak_gib={_gib(sampled_peak)} "
        f"cgroup_lifetime_peak_gib={_gib(snapshot['highwater'])} "
        f"limit_gib={_gib(snapshot['limit'])} "
        f"anon_gib={_gib(snapshot['anon'])} file_gib={_gib(snapshot['file'])} "
        f"oom_delta={oom_delta} oom_kill_delta={kill_delta}",
        flush=True,
    )


def _sample_until_done(stop, state, action, started, initial):
    next_report = started + 30
    while not stop.wait(5):
        try:
            snapshot = _snapshot()
            if snapshot is not None:
                state["sampled_peak"] = max(state["sampled_peak"], snapshot["current"])
            if time.monotonic() >= next_report:
                _log(action, "running", snapshot, state["sampled_peak"], started, initial)
                next_report = time.monotonic() + 30
        except Exception as error:
            print(f"[H3 RAM] monitor_error={type(error).__name__}", flush=True)


def monitor_generation_memory(func):
    """Monitor generation actions; preserve the original handler behavior."""
    @functools.wraps(func)
    def wrapped(event):
        input_data = event.get("input", {}) if isinstance(event, dict) else {}
        action = input_data.get("action", "generate") if isinstance(input_data, dict) else None
        if action not in _MONITORED_ACTIONS:
            return func(event)

        started = time.monotonic()
        try:
            initial = _snapshot()
        except Exception:
            initial = None
        state = {"sampled_peak": initial["current"] if initial else 0}
        _log(action, "start", initial, state["sampled_peak"], started, initial)
        stop = threading.Event()
        monitor = threading.Thread(
            target=_sample_until_done,
            args=(stop, state, action, started, initial),
            name="h3-ram-monitor",
            daemon=True,
        )
        monitor.start()
        try:
            return func(event)
        finally:
            stop.set()
            monitor.join(timeout=2)
            try:
                final = _snapshot()
                if final is not None:
                    state["sampled_peak"] = max(state["sampled_peak"], final["current"])
                _log(action, "end", final, state["sampled_peak"], started, initial)
            except Exception as error:
                print(f"[H3 RAM] end_monitor_error={type(error).__name__}", flush=True)
    return wrapped
