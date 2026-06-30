import psutil
import time
import os

def write_memory_usage():

    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / (1024 * 1024)  # in MB

    with open('memory_usage.txt', 'a') as f:
        f.write(f"{time.time()} {mem_usage} MB\n")
