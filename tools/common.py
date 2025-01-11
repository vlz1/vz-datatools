import os
from colorama import Fore

trace_enabled = (os.environ.get("TRACE", "") == "1")
num_proc = min(os.cpu_count(), 8)

def log_trace(object):
    if not trace_enabled:
        return
    print(f"{Fore.LIGHTCYAN_EX}[TRACE]{Fore.RESET} {Fore.WHITE}{object}{Fore.RESET}")

def log_info(object):
    print(f"{Fore.WHITE}[INFO] {object}{Fore.RESET}")

def log_ok(object):
    print(f"{Fore.WHITE}[{Fore.GREEN}OK{Fore.WHITE}] {object}{Fore.RESET}")

def log_failed(object):
    print(f"{Fore.LIGHTRED_EX}[FAILED] {object}{Fore.RESET}")