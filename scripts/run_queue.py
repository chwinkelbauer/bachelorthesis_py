"""
Experiment queue runner for unattended nights: runs the jobs of a queue file with at most `--parallel` jobs at a time.
Start the runner itself detached (CLAUDE.md §3), so that it survives editor/session restarts.

    python -m scripts.run_queue configs/night_queue.txt --parallel 2

Queue file, one job per line ("#" comments and blank lines are ignored):
    <name>: <command> [&& <command> ...]     a job; its commands run in sequence, a failing command ends the job
    wait                                     barrier: wait until all running jobs have finished
Commands are split like a POSIX shell (quotes group), so write paths with forward slashes: a line containing a
backslash is rejected, because the splitting would silently turn C:\dir\file into the relative path C:dirfile.
Commands starting with "python" use the interpreter running this script. A job's output goes to <logs>/<name>.log /
<name>.err; afterwards both are moved to <logs>/<name>/train.log / train.err if the run directory exists. Progress is
appended to <logs>/queue.log. Creating the file <logs>/queue.stop stops the runner from starting further jobs.
"""

import argparse
import os
import shlex
import subprocess
import sys
import threading
import time
from typing import List, Tuple

Job = Tuple[str, List[List[str]]]


def parse_queue(path: str) -> List[object]:
    """Returns a list of jobs (name, [argv, ...]) and "wait" barriers."""
    items: List[object] = []
    with open(path, encoding="utf-8") as f:
        for number, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line == "wait":
                items.append("wait")
                continue
            if "\\" in line:
                raise ValueError(f"{path}:{number}: use forward slashes, backslashes are not allowed: {line!r}")
            name, sep, command = line.partition(":")
            if not sep or not name.strip() or " " in name.strip():
                raise ValueError(f"{path}:{number}: expected '<name>: <command>', got {line!r}")
            commands = []
            for part in command.split("&&"):
                argv = shlex.split(part.strip())
                if argv and argv[0] == "python":
                    argv[0] = sys.executable
                commands.append(argv)
            items.append((name.strip(), commands))
    return items


def _stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Runner:
    def __init__(self, logs: str, parallel: int):
        self.logs, self.parallel = logs, parallel
        self.running: List[threading.Thread] = []
        self.lock = threading.Lock()
        self.failed: List[str] = []
        os.makedirs(logs, exist_ok=True)

    def note(self, message: str) -> None:
        with self.lock, open(os.path.join(self.logs, "queue.log"), "a", encoding="utf-8") as f:
            f.write(f"{_stamp()}  {message}\n")

    def _run_job(self, name: str, commands: List[List[str]]) -> None:
        out_path, err_path = (os.path.join(self.logs, f"{name}.{ext}") for ext in ("log", "err"))
        ok = True
        with open(out_path, "a", encoding="utf-8") as out, open(err_path, "a", encoding="utf-8") as err:
            for argv in commands:
                self.note(f"start {name}: {' '.join(argv)}")
                code = subprocess.call(argv, stdout=out, stderr=err)
                if code != 0:
                    ok = False
                    self.note(f"FAILED {name} (exit {code}); remaining commands of this job skipped")
                    break
        run_dir = os.path.join(self.logs, name)
        if os.path.isdir(run_dir):
            for src, dst in ((out_path, "train.log"), (err_path, "train.err")):
                target = os.path.join(run_dir, dst)
                if os.path.exists(target):
                    target = os.path.join(run_dir, f"queue_{dst}")
                os.replace(src, target)
        if ok:
            self.note(f"done {name}")
        else:
            self.failed.append(name)

    def _reap(self) -> None:
        self.running = [t for t in self.running if t.is_alive()]

    def run(self, items: List[object]) -> int:
        self.note(f"queue started ({len([i for i in items if i != 'wait'])} jobs, parallel {self.parallel})")
        for item in items:
            if item == "wait":
                for t in self.running:
                    t.join()
                self._reap()
                continue
            while True:
                self._reap()
                if len(self.running) < self.parallel:
                    break
                time.sleep(1.0)
            if os.path.exists(os.path.join(self.logs, "queue.stop")):
                self.note("queue.stop found: no further jobs are started")
                break
            name, commands = item
            thread = threading.Thread(target=self._run_job, args=(name, commands))
            thread.start()
            self.running.append(thread)
        for t in self.running:
            t.join()
        self.note(f"queue finished; failed jobs: {', '.join(self.failed) if self.failed else 'none'}")
        return 1 if self.failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("queue")
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--logs", default="results")
    args = parser.parse_args()
    sys.exit(Runner(args.logs, args.parallel).run(parse_queue(args.queue)))


if __name__ == "__main__":
    main()
